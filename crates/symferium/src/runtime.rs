//! The runtime: loaded instances, their supervision, the governor gate, restore and watchdog.

use crate::config::{self, Config};
use crate::engines::{self, LaunchSpec};
use crate::estimate::{self, Estimate, EstimateInput};
use crate::events::{now_ms, EventBus};
use crate::governor::{self, Budget, Decision};
use crate::library::{Library, ModelEntry};
use crate::probe::SharedProbe;
use crate::proxy;
use crate::state::{LoadRequest, StateStore};
use crate::symbiosis::Chain;
use anyhow::{anyhow, bail, Result};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Default)]
pub struct Status {
    /// starting, ready, restarting, failed, stopping
    pub state: String,
    pub pid: Option<u32>,
    pub restarts: u32,
    pub started_ms: u64,
    pub ready_ms: Option<u64>,
    pub load_seconds: Option<f64>,
    pub last_error: Option<String>,
    pub measured_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Metrics {
    pub requests: u64,
    pub errors: u64,
    pub in_flight: i64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub last_decode_tps: f64,
    pub avg_decode_tps: f64,
    pub last_prefill_tps: f64,
    pub last_ttft_ms: f64,
    pub last_latency_ms: f64,
    pub last_request_ms: u64,
}

pub struct Instance {
    pub id: String,
    pub model: ModelEntry,
    pub request: LoadRequest,
    pub spec: LaunchSpec,
    pub class: String,
    pub priority: i32,
    pub ctx: u64,
    pub port: u16,
    pub internal_port: u16,
    pub mmproj: Option<PathBuf>,
    pub estimate: Estimate,
    pub available_before: u64,
    pub status: RwLock<Status>,
    pub metrics: Mutex<Metrics>,
    pub logs: Mutex<VecDeque<String>>,
    pub ready_tx: watch::Sender<bool>,
    pub stop: CancellationToken,
    pub done: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstanceView {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub path: PathBuf,
    pub format: String,
    pub kind: String,
    pub class: String,
    pub engine: String,
    pub adapter: String,
    pub port: u16,
    pub url: String,
    pub internal_port: u16,
    pub ctx: u64,
    pub mmproj: Option<PathBuf>,
    pub vision: bool,
    pub priority: i32,
    pub estimate: Estimate,
    pub status: Status,
    pub metrics: Metrics,
    pub command: String,
}

impl Instance {
    pub fn view(&self, host: &str) -> InstanceView {
        InstanceView {
            id: self.id.clone(),
            model_id: self.model.id.clone(),
            model_name: self.model.name.clone(),
            path: self.model.path.clone(),
            format: self.model.format.clone(),
            kind: self.model.kind.clone(),
            class: self.class.clone(),
            engine: self.spec.engine.clone(),
            adapter: self.spec.adapter.clone(),
            port: self.port,
            url: format!("http://{host}:{}/v1", self.port),
            internal_port: self.internal_port,
            ctx: self.ctx,
            mmproj: self.mmproj.clone(),
            vision: self.mmproj.is_some() || (self.model.format == "mlx" && self.model.flags.vision),
            priority: self.priority,
            estimate: self.estimate.clone(),
            status: self.status.read().clone(),
            metrics: self.metrics.lock().clone(),
            command: self.spec.command_line(),
        }
    }
    pub fn state(&self) -> String {
        self.status.read().state.clone()
    }
    fn set_state(&self, s: &str) {
        self.status.write().state = s.into();
    }
    pub fn log(&self, line: String, keep: usize) {
        let mut l = self.logs.lock();
        l.push_back(line);
        while l.len() > keep {
            l.pop_front();
        }
    }
    pub fn committed_bytes(&self) -> u64 {
        let st = self.status.read();
        self.estimate.total.max(st.measured_bytes.unwrap_or(0))
    }
}

#[derive(Debug)]
pub enum LoadError {
    Refused(Box<Decision>),
    Invalid(anyhow::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Refused(d) => write!(f, "refused by the governor: {}", d.reasons.join("; ")),
            LoadError::Invalid(e) => write!(f, "{e:#}"),
        }
    }
}

impl From<anyhow::Error> for LoadError {
    fn from(e: anyhow::Error) -> Self {
        LoadError::Invalid(e)
    }
}

pub struct Runtime {
    pub config: RwLock<Config>,
    pub config_path: Option<PathBuf>,
    pub probe: SharedProbe,
    pub library: Library,
    pub events: EventBus,
    pub state: StateStore,
    pub instances: Mutex<BTreeMap<String, Arc<Instance>>>,
    pub middleware: RwLock<Chain>,
    pub http: reqwest::Client,
    pub downloads: crate::download::Downloads,
    pub catalog_cache: crate::catalog::Cache,
    load_lock: tokio::sync::Mutex<()>,
}

pub struct Planned {
    pub entry: ModelEntry,
    pub engine: String,
    pub params: BTreeMap<String, String>,
    pub class: String,
    pub priority: i32,
    pub ctx: u64,
    pub mmproj: Option<PathBuf>,
    pub estimate: Estimate,
    pub decision: Decision,
}

impl Runtime {
    pub fn new(config: Config, config_path: Option<PathBuf>, probe: SharedProbe, state: StateStore) -> Arc<Self> {
        let rt = Arc::new(Self {
            config: RwLock::new(config),
            config_path,
            probe,
            library: Library::new(),
            events: EventBus::new(500),
            state,
            instances: Mutex::new(BTreeMap::new()),
            middleware: RwLock::new(Chain::new()),
            http: reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).pool_idle_timeout(Duration::from_secs(30)).build().expect("http client"),
            downloads: Default::default(),
            catalog_cache: Default::default(),
            load_lock: tokio::sync::Mutex::new(()),
        });
        rt.library.scan(&rt.config.read());
        rt
    }

    pub fn cfg(&self) -> Config {
        self.config.read().clone()
    }

    pub fn set_config(&self, cfg: Config) -> Result<()> {
        if let Some(p) = &self.config_path {
            config::save(p, &cfg)?;
        }
        *self.config.write() = cfg;
        self.events.emit("settings.changed", serde_json::json!({}));
        Ok(())
    }

    pub fn list(&self) -> Vec<InstanceView> {
        let host = self.cfg().server.host;
        self.instances.lock().values().map(|i| i.view(&host)).collect()
    }

    pub fn get(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().get(id).cloned()
    }

    fn committed_and_pending(&self, exclude: Option<&str>) -> (u64, u64, Vec<String>) {
        let mut committed = 0;
        let mut pending = 0;
        let mut classes = vec![];
        for i in self.instances.lock().values() {
            if Some(i.id.as_str()) == exclude {
                continue;
            }
            let st = i.state();
            if st == "failed" || st == "stopping" {
                continue;
            }
            committed += i.committed_bytes();
            if st == "starting" || st == "restarting" {
                pending += i.estimate.total;
            }
            classes.push(i.class.clone());
        }
        (committed, pending, classes)
    }

    pub fn budget(&self) -> Budget {
        let (c, p, _) = self.committed_and_pending(None);
        governor::budget(&self.cfg(), &self.probe.snapshot(), c, p)
    }

    /// Resolve the request into engine, settings and an estimate, and ask the governor.
    pub fn plan(&self, req: &LoadRequest) -> Result<Planned> {
        let cfg = self.cfg();
        let entry = match self.library.get(&req.model) {
            Some(e) => e,
            None => {
                self.library.scan(&cfg);
                self.library.get(&req.model).ok_or_else(|| anyhow!("model {} is not in the library (rescan or check paths)", req.model))?
            }
        };
        if entry.class == "oversize" {
            bail!("{} is larger than catalog.large_max_b", entry.name);
        }
        let class_name = req.params.get("class").cloned().unwrap_or_else(|| entry.class.clone());
        let class = cfg.class(&class_name);
        let engine = engines::choose_engine(&cfg, &entry, req, &class)?;
        let adapter = cfg.engines.get(&engine).map(|p| p.adapter.clone()).unwrap_or_default();
        let mut params = engines::params_for(&class, req);
        let ctx_trained = entry.ctx_train.unwrap_or(0);
        let mut ctx: u64 = params.get("ctx").and_then(|v| v.parse().ok()).unwrap_or(class.ctx as u64);
        if req.ctx.is_none() && ctx_trained > 0 && ctx > ctx_trained {
            ctx = ctx_trained; // class default never exceeds what the model was trained for
            params.insert("ctx".into(), ctx.to_string());
        }
        let mmproj = match req.mmproj.as_deref() {
            None | Some("auto") => entry.mmproj.clone(),
            Some("none") | Some("") => None,
            Some(p) => Some(config::expand(p)),
        };
        let mmproj_bytes = mmproj.as_ref().and_then(|p| p.metadata().ok()).map(|m| m.len()).unwrap_or(0);
        let num = |k: &str| params.get(k).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let est = estimate::estimate(&EstimateInput {
            adapter: adapter.clone(),
            kind: entry.kind.clone(),
            weights_bytes: entry.size_bytes,
            projector_bytes: mmproj_bytes,
            geometry: entry.geometry.clone(),
            ctx,
            parallel: num("parallel").max(1),
            ubatch: num("ubatch").max(1),
            cache_type_k: params.get("cache_type_k").cloned().unwrap_or("f16".into()),
            cache_type_v: params.get("cache_type_v").cloned().unwrap_or("f16".into()),
            ctx_checkpoints: num("ctx_checkpoints"),
            cache_ram_mb: num("cache_ram_mb"),
            overhead_mb: cfg.governor.engine_overhead_mb,
            correction: self.state.correction(&adapter),
        });
        let alias = req.alias.clone().unwrap_or_else(|| entry.id.clone());
        let (committed, pending, classes) = self.committed_and_pending(Some(&alias));
        let decision = governor::check(&cfg, &self.probe.snapshot(), est.clone(), committed, pending, &classes, &class_name);
        Ok(Planned {
            priority: params.get("priority").and_then(|v| v.parse().ok()).unwrap_or(class.priority),
            entry,
            engine,
            params,
            class: class_name,
            ctx,
            mmproj,
            estimate: est,
            decision,
        })
    }

    fn pick_port(&self, cfg: &Config, wanted: Option<u16>, internal: bool) -> Result<u16> {
        let used: Vec<u16> = self.instances.lock().values().flat_map(|i| [i.port, i.internal_port]).collect();
        let (lo, hi) = if internal {
            (cfg.server.internal_port_start, cfg.server.internal_port_end)
        } else {
            (cfg.server.model_port_start, cfg.server.model_port_end)
        };
        let free = |p: u16| !used.contains(&p) && std::net::TcpListener::bind((cfg.server.host.as_str(), p)).is_ok();
        if let Some(p) = wanted {
            if p == cfg.server.control_port || !free(p) {
                bail!("port {p} is in use");
            }
            return Ok(p);
        }
        (lo..=hi).find(|p| free(*p)).ok_or_else(|| anyhow!("no free port in {lo}..={hi}"))
    }

    pub async fn load(self: &Arc<Self>, req: LoadRequest) -> Result<InstanceView, LoadError> {
        let _guard = self.load_lock.lock().await;
        let cfg = self.cfg();
        let plan = self.plan(&req)?;
        let alias = req.alias.clone().unwrap_or_else(|| plan.entry.id.clone());
        if self.instances.lock().contains_key(&alias) {
            return Err(anyhow!("{alias} is already loaded (unload it first or pass another alias)").into());
        }
        if !plan.decision.allowed {
            self.events.emit("governor.refused", serde_json::json!({"model": req.model, "reasons": plan.decision.reasons, "estimate": plan.estimate}));
            return Err(LoadError::Refused(Box::new(plan.decision)));
        }
        let port = self.pick_port(&cfg, req.port, false)?;
        let internal_port = self.pick_port(&cfg, None, true)?;
        let spec = engines::build(&cfg, &plan.engine, &plan.entry, &plan.params, plan.mmproj.as_deref(), "127.0.0.1", internal_port, &alias)?;
        let listener = tokio::net::TcpListener::bind((cfg.server.host.as_str(), port)).await.map_err(|e| anyhow!("binding port {port}: {e}"))?;
        let (ready_tx, _) = watch::channel(false);
        let inst = Arc::new(Instance {
            id: alias.clone(),
            model: plan.entry.clone(),
            request: req.clone(),
            spec,
            class: plan.class.clone(),
            priority: plan.priority,
            ctx: plan.ctx,
            port,
            internal_port,
            mmproj: plan.mmproj.clone(),
            estimate: plan.estimate.clone(),
            available_before: self.probe.snapshot().available_bytes,
            status: RwLock::new(Status { state: "starting".into(), started_ms: now_ms(), ..Default::default() }),
            metrics: Mutex::new(Metrics::default()),
            logs: Mutex::new(VecDeque::new()),
            ready_tx,
            stop: CancellationToken::new(),
            done: Mutex::new(None),
        });
        self.instances.lock().insert(alias.clone(), inst.clone());
        proxy::serve(self.clone(), inst.clone(), listener);
        let rt = self.clone();
        let i2 = inst.clone();
        *inst.done.lock() = Some(tokio::spawn(async move { rt.supervise(i2).await }));
        if req.persist.unwrap_or(true) {
            let mut saved = req.clone();
            saved.wait = false;
            saved.port = Some(port);
            saved.alias = Some(alias.clone());
            self.state.update(|d| {
                d.model_set.retain(|r| r.alias.as_deref() != Some(alias.as_str()));
                d.model_set.push(saved);
            });
        }
        self.events.emit("model.loading", serde_json::json!({"id": alias, "port": port, "engine": plan.engine, "estimate_bytes": plan.estimate.total}));
        drop(_guard);
        if req.wait {
            let mut rx = inst.ready_tx.subscribe();
            let timeout = Duration::from_secs(cfg.supervisor.startup_timeout_s + 5);
            let res = tokio::time::timeout(timeout, async {
                loop {
                    if *rx.borrow() {
                        return Ok(());
                    }
                    if inst.state() == "failed" {
                        return Err(inst.status.read().last_error.clone().unwrap_or_else(|| "failed".into()));
                    }
                    tokio::select! {
                        r = rx.changed() => { if r.is_err() { return Err("stopped".to_string()) } }
                        _ = tokio::time::sleep(Duration::from_millis(200)) => {}
                    }
                }
            })
            .await;
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(anyhow!("{alias} failed to start: {e}").into()),
                Err(_) => return Err(anyhow!("{alias} did not become ready in time").into()),
            }
        }
        Ok(inst.view(&cfg.server.host))
    }

    pub async fn unload(&self, id: &str, forget: bool) -> Result<()> {
        let inst = self.instances.lock().get(id).cloned().ok_or_else(|| anyhow!("{id} is not loaded"))?;
        inst.set_state("stopping");
        inst.stop.cancel();
        let handle = inst.done.lock().take();
        if let Some(h) = handle {
            let _ = tokio::time::timeout(Duration::from_secs(self.cfg().supervisor.stop_timeout_s + 5), h).await;
        }
        self.instances.lock().remove(id);
        if forget {
            self.state.update(|d| d.model_set.retain(|r| r.alias.as_deref() != Some(id)));
        }
        self.events.emit("model.unloaded", serde_json::json!({"id": id}));
        Ok(())
    }

    pub async fn unload_all(&self) {
        let ids: Vec<String> = self.instances.lock().keys().cloned().collect();
        for id in ids {
            let _ = self.unload(&id, false).await;
        }
    }

    // ------------------------------------------------------------ supervision

    async fn supervise(self: Arc<Self>, inst: Arc<Instance>) {
        let mut attempt: u32 = 0;
        loop {
            let cfg = self.cfg();
            let sup = cfg.supervisor.clone();
            match self.run_once(&inst, &cfg).await {
                Exit::Stopped => break,
                Exit::Crashed(why) => {
                    let _ = inst.ready_tx.send(false);
                    inst.status.write().last_error = Some(why.clone());
                    attempt += 1;
                    if attempt > sup.max_restarts {
                        inst.set_state("failed");
                        self.events.emit("model.failed", serde_json::json!({"id": inst.id, "error": why, "restarts": attempt - 1}));
                        // Stay listed as failed (the port answers 503 with the error) until unloaded.
                        inst.stop.cancelled().await;
                        break;
                    }
                    inst.status.write().restarts = attempt;
                    inst.set_state("restarting");
                    self.events.emit("model.restarting", serde_json::json!({"id": inst.id, "error": why, "attempt": attempt}));
                    let backoff = (sup.restart_backoff_ms.saturating_mul(1 << (attempt - 1).min(6))).min(60_000);
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(backoff)) => {}
                        _ = inst.stop.cancelled() => break,
                    }
                }
            }
        }
        let _ = inst.ready_tx.send(false);
    }

    async fn run_once(&self, inst: &Arc<Instance>, cfg: &Config) -> Exit {
        let sup = &cfg.supervisor;
        let mut cmd = tokio::process::Command::new(&inst.spec.program);
        cmd.args(&inst.spec.args)
            .envs(&inst.spec.env)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        let started = Instant::now();
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Exit::Crashed(format!("spawn {}: {e}", inst.spec.program.display())),
        };
        let pid = child.id();
        {
            let mut st = inst.status.write();
            st.pid = pid;
            st.started_ms = now_ms();
        }
        let log_file = open_log(cfg, &inst.id);
        if let Some(out) = child.stdout.take() {
            spawn_log_reader(inst.clone(), out, sup.log_lines, log_file.clone());
        }
        if let Some(err) = child.stderr.take() {
            spawn_log_reader(inst.clone(), err, sup.log_lines, log_file);
        }
        inst.log(format!("[symferium] started pid {:?}: {}", pid, inst.spec.command_line()), sup.log_lines);

        let health_url = format!("http://127.0.0.1:{}{}", inst.internal_port, inst.spec.health_path);
        let interval = Duration::from_millis(sup.health_interval_ms.max(50));
        // Wait until healthy.
        let deadline = started + Duration::from_secs(sup.startup_timeout_s);
        loop {
            tokio::select! {
                _ = inst.stop.cancelled() => { stop_child(&mut child, pid, sup.stop_timeout_s).await; return Exit::Stopped; }
                st = child.wait() => { return Exit::Crashed(format!("exited during startup: {}", st.map(|s| s.to_string()).unwrap_or_default())); }
                _ = tokio::time::sleep(interval.min(Duration::from_millis(250))) => {}
            }
            if self.healthy(&health_url).await {
                break;
            }
            if Instant::now() > deadline {
                stop_child(&mut child, pid, sup.stop_timeout_s).await;
                return Exit::Crashed(format!("not healthy after {} s", sup.startup_timeout_s));
            }
        }
        let load_s = started.elapsed().as_secs_f64();
        {
            let mut st = inst.status.write();
            st.state = "ready".into();
            st.ready_ms = Some(now_ms());
            st.load_seconds = Some(load_s);
        }
        let _ = inst.ready_tx.send(true);
        self.events.emit("model.ready", serde_json::json!({"id": inst.id, "port": inst.port, "load_seconds": load_s, "pid": pid}));
        let measure_at = Instant::now() + Duration::from_secs(cfg.governor.measure_after_s);
        let mut measured = false;
        let mut failures = 0u32;
        loop {
            tokio::select! {
                _ = inst.stop.cancelled() => { stop_child(&mut child, pid, sup.stop_timeout_s).await; return Exit::Stopped; }
                st = child.wait() => { return Exit::Crashed(format!("exited: {}", st.map(|s| s.to_string()).unwrap_or_default())); }
                _ = tokio::time::sleep(interval) => {}
            }
            if self.healthy(&health_url).await {
                failures = 0;
            } else {
                failures += 1;
                if failures >= sup.health_failures_before_restart {
                    stop_child(&mut child, pid, sup.stop_timeout_s).await;
                    return Exit::Crashed(format!("{failures} failed health checks"));
                }
            }
            if !measured && Instant::now() >= measure_at {
                measured = true;
                self.measure(inst, pid, cfg);
            }
        }
    }

    async fn healthy(&self, url: &str) -> bool {
        matches!(self.http.get(url).timeout(Duration::from_secs(5)).send().await, Ok(r) if r.status().is_success())
    }

    /// Measure the engine's memory and fold the error into the adapter's correction.
    fn measure(&self, inst: &Arc<Instance>, pid: Option<u32>, cfg: &Config) {
        let footprint = pid.and_then(|p| self.probe.process_footprint(p)).unwrap_or(0);
        let now_avail = self.probe.snapshot().available_bytes;
        let delta = inst.available_before.saturating_sub(now_avail);
        let measured = footprint.max(delta);
        inst.status.write().measured_bytes = Some(measured);
        let mut data = serde_json::json!({"id": inst.id, "footprint_bytes": footprint, "available_delta_bytes": delta, "estimate_bytes": inst.estimate.total});
        if cfg.governor.learn_corrections && measured > 0 && inst.estimate.total > 0 {
            let raw = inst.estimate.total as f64 / inst.estimate.correction.max(0.01);
            let c = self.state.learn(&inst.spec.adapter, measured as f64 / raw);
            data["new_correction"] = serde_json::json!(c);
        }
        self.events.emit("model.measured", data);
    }

    // ------------------------------------------------------------ background tasks

    /// Restore the saved model set after the autostart delay, one model at a time.
    pub fn spawn_restore(self: &Arc<Self>) {
        let rt = self.clone();
        tokio::spawn(async move {
            let cfg = rt.cfg();
            if !cfg.autostart.restore_model_set {
                return;
            }
            tokio::time::sleep(Duration::from_secs(cfg.autostart.delay_s)).await;
            let set = rt.state.data.lock().model_set.clone();
            for mut req in set {
                req.wait = true;
                let name = req.alias.clone().unwrap_or(req.model.clone());
                match rt.load(req).await {
                    Ok(_) => rt.events.emit("restore.loaded", serde_json::json!({"id": name})),
                    Err(e) => rt.events.emit("restore.skipped", serde_json::json!({"id": name, "error": e.to_string()})),
                }
                tokio::time::sleep(Duration::from_secs(rt.cfg().autostart.stagger_s)).await;
            }
        });
    }

    /// Memory-pressure watchdog.
    pub fn spawn_watchdog(self: &Arc<Self>) {
        let rt = self.clone();
        tokio::spawn(async move {
            let mut last_unload = Instant::now() - Duration::from_secs(3600);
            let mut last_pressure = String::new();
            loop {
                let cfg = rt.cfg();
                tokio::time::sleep(Duration::from_secs(cfg.governor.watchdog_interval_s.max(1))).await;
                let m = rt.probe.snapshot();
                if m.pressure != last_pressure {
                    rt.events.emit("machine.pressure", serde_json::json!({"pressure": m.pressure, "available_bytes": m.available_bytes}));
                    last_pressure = m.pressure.clone();
                }
                if m.pressure == "critical" && cfg.governor.pressure_policy == "unload_lowest" && last_unload.elapsed() > Duration::from_secs(30) {
                    let victim = rt.instances.lock().values().filter(|i| i.state() == "ready").min_by_key(|i| i.priority).map(|i| i.id.clone());
                    if let Some(id) = victim {
                        rt.events.emit("governor.unload_on_pressure", serde_json::json!({"id": id}));
                        let _ = rt.unload(&id, false).await;
                        last_unload = Instant::now();
                    }
                }
            }
        });
    }
}

enum Exit {
    Stopped,
    Crashed(String),
}

type LogFile = Option<Arc<Mutex<std::fs::File>>>;

fn open_log(cfg: &Config, id: &str) -> LogFile {
    let dir = cfg.logs_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{id}.log"))).ok()?;
    Some(Arc::new(Mutex::new(f)))
}

fn spawn_log_reader<R: tokio::io::AsyncRead + Unpin + Send + 'static>(inst: Arc<Instance>, r: R, keep: usize, file: LogFile) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(r).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(f) = &file {
                let _ = writeln!(f.lock(), "{line}");
            }
            inst.log(line, keep);
        }
    });
}

async fn stop_child(child: &mut tokio::process::Child, pid: Option<u32>, grace_s: u64) {
    #[cfg(unix)]
    if let Some(p) = pid {
        // The engine runs in its own process group; signal the whole group.
        unsafe {
            libc::killpg(p as i32, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
    if tokio::time::timeout(Duration::from_secs(grace_s), child.wait()).await.is_err() {
        #[cfg(unix)]
        if let Some(p) = pid {
            unsafe {
                libc::killpg(p as i32, libc::SIGKILL);
            }
        }
        let _ = child.kill().await;
    }
}
