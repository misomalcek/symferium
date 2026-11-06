//! One TOML config holds every limit, default, port, path, timeout and engine flag.
//!
//! The typed [`Config`] is what the code reads. Every leaf of it is also described by a
//! [`SettingSpec`] (see [`schema`]) so the UI can render it and the API can validate a change.
//! Defaults that depend on the machine (threads, memory reserve) are filled from the probe the
//! first time the file is created, then written out so the user sees and owns them.

use crate::probe::MachineInfo;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub server: ServerConfig,
    pub paths: PathsConfig,
    pub governor: GovernorConfig,
    pub supervisor: SupervisorConfig,
    pub autostart: AutostartConfig,
    pub catalog: CatalogConfig,
    pub download: DownloadConfig,
    pub bench: BenchConfig,
    /// Per-class defaults: engine, context, KV cache type, batch sizes, threads, priority.
    pub classes: BTreeMap<String, ClassDefaults>,
    /// Engine profiles. Adding a profile needs no code: it is a binary plus argument templates.
    pub engines: BTreeMap<String, EngineProfile>,
    pub symbiosis: SymbiosisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub control_port: u16,
    pub model_port_start: u16,
    pub model_port_end: u16,
    pub internal_port_start: u16,
    pub internal_port_end: u16,
    /// Bearer token for the whole control API. Empty = no auth (localhost only).
    pub admin_token: String,
    /// Bearer token that may call /api/tools only. Empty = no auth for tools.
    pub agent_token: String,
    /// Seconds a request waits on a model port while its engine (re)starts.
    pub proxy_wait_ready_s: u64,
    /// Largest JSON body the proxy buffers for model-field rewrite and middleware (bytes).
    pub proxy_max_json_body: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PathsConfig {
    pub models_dir: String,
    /// Extra folders scanned for models, read-only (never written or deleted).
    pub extra_model_dirs: Vec<String>,
    pub state_dir: String,
    pub logs_dir: String,
    /// Python env (created with uv) that holds mlx-lm, mlx-vlm and the TTS sidecar deps.
    pub python_env: String,
    /// Folder with Symferium's sidecar scripts (TTS).
    pub sidecar_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GovernorConfig {
    /// Keep at least this much memory free after any load (GB).
    pub min_free_gb: f64,
    /// Upper bound for everything Symferium loads (GB). 0 = min(total − min_free, Metal working set).
    pub max_total_gb: f64,
    /// Fixed per-engine overhead added to every estimate (MB).
    pub engine_overhead_mb: f64,
    /// Refuse new loads at or above this macOS memory pressure: "warn", "critical" or "never".
    pub refuse_on_pressure: String,
    /// What the watchdog does on critical pressure: "refuse" (block new loads) or "unload_lowest".
    pub pressure_policy: String,
    pub watchdog_interval_s: u64,
    /// A large-class model loads only when no other chat model is loaded, and vice versa.
    pub exclusive_large: bool,
    /// Learn estimate error per engine from measured footprints.
    pub learn_corrections: bool,
    /// Seconds after "ready" before the footprint is measured for learning.
    pub measure_after_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SupervisorConfig {
    pub health_interval_ms: u64,
    pub health_failures_before_restart: u32,
    pub startup_timeout_s: u64,
    pub max_restarts: u32,
    pub restart_backoff_ms: u64,
    pub stop_timeout_s: u64,
    pub log_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AutostartConfig {
    /// Restore the saved model set when the daemon starts.
    pub restore_model_set: bool,
    /// Seconds to wait after daemon start before restoring, so login is not slowed.
    pub delay_s: u64,
    /// Seconds between restored models (they load one at a time).
    pub stagger_s: u64,
    pub launchd_label: String,
    /// launchd ProcessType for the daemon: Background, Adaptive, Standard or Interactive.
    pub process_type: String,
    pub nice: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CatalogConfig {
    pub hf_endpoint: String,
    /// Hugging Face token for gated repos (Gemma). Stored in this file with 0600 permissions.
    pub hf_token: String,
    /// Community quant publishers offered beside the originals.
    pub authors: Vec<String>,
    pub search_limit: u32,
    pub cache_ttl_s: u64,
    /// Parameter bounds (billions) for the classes.
    pub small_max_b: f64,
    pub standard_min_b: f64,
    pub standard_max_b: f64,
    pub large_max_b: f64,
    /// Preferred GGUF quants, best first.
    pub quant_preference: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DownloadConfig {
    pub concurrent: usize,
    pub retries: u32,
    pub verify_sha256: bool,
    pub timeout_s: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BenchConfig {
    pub prompt: String,
    pub max_tokens: u32,
    pub runs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ClassDefaults {
    pub engine: String,
    /// Engine used when the model's files are MLX weights.
    pub engine_mlx: String,
    pub ctx: u32,
    pub gpu_layers: i32,
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub parallel: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub threads: u32,
    pub pooling: String,
    pub max_tokens: u32,
    /// llama.cpp context checkpoints for SWA/hybrid models (each holds a copy of the SWA state).
    pub ctx_checkpoints: u32,
    /// llama.cpp host prompt cache (MB). Counts against the memory budget.
    pub cache_ram_mb: u32,
    /// Lower number = unloaded first under memory pressure.
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EngineProfile {
    /// Adapter family: llamacpp, mlx, whispercpp, tts, command.
    pub adapter: String,
    pub enabled: bool,
    pub binary: String,
    pub args: Vec<String>,
    pub args_vision: Vec<String>,
    pub args_embed: Vec<String>,
    pub args_rerank: Vec<String>,
    pub extra_args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub health_path: String,
    /// Replace the "model" field of JSON requests with the served model (mlx servers need it).
    pub rewrite_model_field: bool,
    /// Model kinds this profile can serve: chat, vision, embed, rerank, stt, tts.
    pub kinds: Vec<String>,
    /// Model formats this profile can load: gguf, mlx, whisper, onnx.
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SymbiosisConfig {
    pub enabled: bool,
    pub qdrant_url: String,
    pub graph_url: String,
    pub memory_dir: String,
    pub context_threshold: f64,
}

// ---------------------------------------------------------------- defaults

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            control_port: 7700,
            model_port_start: 7701,
            model_port_end: 7799,
            internal_port_start: 17701,
            internal_port_end: 17899,
            admin_token: String::new(),
            agent_token: String::new(),
            proxy_wait_ready_s: 120,
            proxy_max_json_body: 64 * 1024 * 1024,
        }
    }
}

fn app_support() -> String {
    if cfg!(target_os = "macos") {
        "~/Library/Application Support/Symferium".into()
    } else {
        "~/.local/share/symferium".into()
    }
}

impl Default for PathsConfig {
    fn default() -> Self {
        let base = app_support();
        Self {
            models_dir: format!("{base}/models"),
            extra_model_dirs: vec!["~/.factorium/models".into()],
            state_dir: base.clone(),
            logs_dir: if cfg!(target_os = "macos") { "~/Library/Logs/Symferium".into() } else { format!("{base}/logs") },
            python_env: format!("{base}/pyenv"),
            sidecar_dir: format!("{base}/sidecars"),
        }
    }
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            min_free_gb: 4.0,
            max_total_gb: 0.0,
            engine_overhead_mb: 300.0,
            refuse_on_pressure: "warn".into(),
            pressure_policy: "refuse".into(),
            watchdog_interval_s: 5,
            exclusive_large: true,
            learn_corrections: true,
            measure_after_s: 20,
        }
    }
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            health_interval_ms: 2000,
            health_failures_before_restart: 5,
            startup_timeout_s: 600,
            max_restarts: 5,
            restart_backoff_ms: 1000,
            stop_timeout_s: 10,
            log_lines: 2000,
        }
    }
}

impl Default for AutostartConfig {
    fn default() -> Self {
        Self {
            restore_model_set: true,
            delay_s: 30,
            stagger_s: 5,
            launchd_label: "dev.symferium.daemon".into(),
            process_type: "Adaptive".into(),
            nice: 0,
        }
    }
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self {
            hf_endpoint: "https://huggingface.co".into(),
            hf_token: String::new(),
            authors: vec!["ggml-org".into(), "unsloth".into(), "bartowski".into(), "mlx-community".into(), "lmstudio-community".into()],
            search_limit: 40,
            cache_ttl_s: 3600,
            small_max_b: 5.0,
            standard_min_b: 5.0,
            standard_max_b: 22.0,
            large_max_b: 40.0,
            quant_preference: ["Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "Q4_K_S", "IQ4_XS", "Q6_K", "Q8_0", "MXFP4"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self { concurrent: 2, retries: 6, verify_sha256: true, timeout_s: 60 }
    }
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            prompt: "Explain in five sentences how unified memory on Apple Silicon changes the way local language models are served."
                .into(),
            max_tokens: 256,
            runs: 3,
        }
    }
}

impl Default for ClassDefaults {
    fn default() -> Self {
        Self {
            engine: "llamacpp".into(),
            engine_mlx: "mlx".into(),
            ctx: 32768,
            gpu_layers: 999,
            flash_attn: "on".into(),
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            parallel: 1,
            batch: 2048,
            ubatch: 512,
            threads: 4,
            pooling: "none".into(),
            max_tokens: 4096,
            ctx_checkpoints: 2,
            cache_ram_mb: 1024,
            priority: 50,
        }
    }
}

impl Default for EngineProfile {
    fn default() -> Self {
        Self {
            adapter: "command".into(),
            enabled: true,
            binary: String::new(),
            args: vec![],
            args_vision: vec![],
            args_embed: vec![],
            args_rerank: vec![],
            extra_args: vec![],
            env: BTreeMap::new(),
            health_path: "/health".into(),
            rewrite_model_field: false,
            kinds: vec![],
            formats: vec![],
        }
    }
}

impl Default for SymbiosisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            qdrant_url: "http://127.0.0.1:6333".into(),
            graph_url: String::new(),
            memory_dir: String::new(),
            context_threshold: 0.7,
        }
    }
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

pub fn default_engines() -> BTreeMap<String, EngineProfile> {
    let mut m = BTreeMap::new();
    let llama_args = s(&[
        "-m",
        "{model}",
        "--host",
        "{host}",
        "--port",
        "{port}",
        "-c",
        "{ctx}",
        "-ngl",
        "{gpu_layers}",
        "-fa",
        "{flash_attn}",
        "-ctk",
        "{cache_type_k}",
        "-ctv",
        "{cache_type_v}",
        "-np",
        "{parallel}",
        "-b",
        "{batch}",
        "-ub",
        "{ubatch}",
        "-t",
        "{threads}",
        "--ctx-checkpoints",
        "{ctx_checkpoints}",
        "--cache-ram",
        "{cache_ram_mb}",
        "--jinja",
        "--metrics",
        "--no-webui",
        "-a",
        "{alias}",
    ]);
    m.insert(
        "llamacpp".into(),
        EngineProfile {
            adapter: "llamacpp".into(),
            binary: "llama-server".into(),
            args: llama_args.clone(),
            args_vision: s(&["--mmproj", "{mmproj}"]),
            args_embed: s(&["--embeddings", "--pooling", "{pooling}"]),
            args_rerank: s(&["--reranking"]),
            kinds: s(&["chat", "vision", "embed", "rerank"]),
            formats: s(&["gguf"]),
            ..Default::default()
        },
    );
    // Same adapter, pointed at a TurboQuant llama.cpp fork (turbo3/turbo4 KV). Off until measured.
    m.insert(
        "llamacpp-turboquant".into(),
        EngineProfile {
            adapter: "llamacpp".into(),
            enabled: false,
            binary: "~/.local/opt/llama-cpp-turboquant/bin/llama-server".into(),
            args: llama_args,
            args_vision: s(&["--mmproj", "{mmproj}"]),
            args_embed: s(&["--embeddings", "--pooling", "{pooling}"]),
            args_rerank: s(&["--reranking"]),
            kinds: s(&["chat", "vision"]),
            formats: s(&["gguf"]),
            ..Default::default()
        },
    );
    m.insert(
        "mlx".into(),
        EngineProfile {
            adapter: "mlx".into(),
            binary: "{python_env}/bin/mlx_lm.server".into(),
            args: s(&["--model", "{model}", "--host", "{host}", "--port", "{port}", "--max-tokens", "{max_tokens}"]),
            health_path: "/v1/models".into(),
            rewrite_model_field: true,
            kinds: s(&["chat"]),
            formats: s(&["mlx"]),
            ..Default::default()
        },
    );
    m.insert(
        "mlx-vlm".into(),
        EngineProfile {
            adapter: "mlx".into(),
            binary: "{python_env}/bin/python".into(),
            args: s(&["-m", "mlx_vlm.server", "--model", "{model}", "--host", "{host}", "--port", "{port}"]),
            health_path: "/health".into(),
            rewrite_model_field: true,
            kinds: s(&["chat", "vision"]),
            formats: s(&["mlx"]),
            ..Default::default()
        },
    );
    m.insert(
        "rapid-mlx".into(),
        EngineProfile {
            adapter: "mlx".into(),
            enabled: false,
            binary: "{python_env}/bin/rapid-mlx".into(),
            args: s(&["serve", "{model}", "--host", "{host}", "--port", "{port}"]),
            health_path: "/v1/models".into(),
            rewrite_model_field: true,
            kinds: s(&["chat"]),
            formats: s(&["mlx"]),
            ..Default::default()
        },
    );
    m.insert(
        "whispercpp".into(),
        EngineProfile {
            adapter: "whispercpp".into(),
            binary: "whisper-server".into(),
            args: s(&[
                "-m",
                "{model}",
                "--host",
                "{host}",
                "--port",
                "{port}",
                "-t",
                "{threads}",
                "--inference-path",
                "/v1/audio/transcriptions",
            ]),
            health_path: "/".into(),
            kinds: s(&["stt"]),
            formats: s(&["whisper"]),
            ..Default::default()
        },
    );
    m.insert(
        "tts".into(),
        EngineProfile {
            adapter: "tts".into(),
            binary: "{python_env}/bin/python".into(),
            args: s(&["{sidecar_dir}/tts_server.py", "--model", "{model}", "--host", "{host}", "--port", "{port}"]),
            health_path: "/health".into(),
            kinds: s(&["tts"]),
            formats: s(&["onnx"]),
            ..Default::default()
        },
    );
    m
}

pub fn default_classes(threads: u32) -> BTreeMap<String, ClassDefaults> {
    let base = ClassDefaults { threads, ..Default::default() };
    let mut m = BTreeMap::new();
    m.insert("standard".into(), ClassDefaults { priority: 60, ..base.clone() });
    m.insert("large".into(), ClassDefaults { ctx: 16384, priority: 40, ..base.clone() });
    m.insert("small".into(), ClassDefaults { cache_type_k: "f16".into(), cache_type_v: "f16".into(), priority: 50, ..base.clone() });
    let helper = ClassDefaults {
        ctx: 8192,
        cache_type_k: "f16".into(),
        cache_type_v: "f16".into(),
        batch: 8192,
        ubatch: 8192,
        ctx_checkpoints: 0,
        cache_ram_mb: 0,
        priority: 70,
        ..base.clone()
    };
    m.insert("embed".into(), ClassDefaults { pooling: "cls".into(), ..helper.clone() });
    m.insert("rerank".into(), ClassDefaults { pooling: "rank".into(), ..helper.clone() });
    m.insert("decision".into(), ClassDefaults { ctx: 4096, ..helper });
    m.insert("stt".into(), ClassDefaults { engine: "whispercpp".into(), priority: 30, ..base.clone() });
    m.insert("tts".into(), ClassDefaults { engine: "tts".into(), priority: 30, ..base });
    m
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: Default::default(),
            paths: Default::default(),
            governor: Default::default(),
            supervisor: Default::default(),
            autostart: Default::default(),
            catalog: Default::default(),
            download: Default::default(),
            bench: Default::default(),
            classes: default_classes(4),
            engines: default_engines(),
            symbiosis: Default::default(),
        }
    }
}

impl Config {
    /// Defaults tuned to the probed machine.
    pub fn defaults_for(machine: &MachineInfo) -> Self {
        let mut c = Config::default();
        let threads = machine.perf_cores.max(1);
        c.classes = default_classes(threads);
        // Keep at least max(4 GB, 15 % of RAM) free for macOS and the user's apps.
        let total_gb = machine.total_bytes as f64 / GB;
        c.governor.min_free_gb = round1((total_gb * 0.15).max(4.0));
        c
    }

    pub fn class(&self, name: &str) -> ClassDefaults {
        self.classes.get(name).cloned().unwrap_or_default()
    }

    pub fn models_dir(&self) -> PathBuf {
        expand(&self.paths.models_dir)
    }
    pub fn state_dir(&self) -> PathBuf {
        expand(&self.paths.state_dir)
    }
    pub fn logs_dir(&self) -> PathBuf {
        expand(&self.paths.logs_dir)
    }

    pub fn validate(&self) -> Result<()> {
        let s = &self.server;
        if s.model_port_start > s.model_port_end {
            bail!("server.model_port_start must be <= server.model_port_end");
        }
        if s.internal_port_start > s.internal_port_end {
            bail!("server.internal_port_start must be <= server.internal_port_end");
        }
        let overlaps = |a: (u16, u16), b: (u16, u16)| a.0 <= b.1 && b.0 <= a.1;
        if overlaps((s.model_port_start, s.model_port_end), (s.internal_port_start, s.internal_port_end)) {
            bail!("model and internal port ranges overlap");
        }
        if (s.model_port_start..=s.model_port_end).contains(&s.control_port) {
            bail!("server.control_port is inside the model port range");
        }
        if !["warn", "critical", "never"].contains(&self.governor.refuse_on_pressure.as_str()) {
            bail!("governor.refuse_on_pressure must be warn, critical or never");
        }
        if !["refuse", "unload_lowest"].contains(&self.governor.pressure_policy.as_str()) {
            bail!("governor.pressure_policy must be refuse or unload_lowest");
        }
        for spec in schema(self) {
            if let (Some(v), true) = (get_value(self, &spec.key).ok(), spec.min.is_some() || spec.max.is_some()) {
                if let Some(n) = value_as_f64(&v) {
                    if spec.min.is_some_and(|m| n < m) || spec.max.is_some_and(|m| n > m) {
                        bail!("{} = {} is outside {:?}..{:?}", spec.key, n, spec.min, spec.max);
                    }
                }
            }
        }
        Ok(())
    }
}

pub const GB: f64 = 1024.0 * 1024.0 * 1024.0;
pub const MB: f64 = 1024.0 * 1024.0;

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Expand a leading `~` to the home directory.
pub fn expand(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest);
        }
    }
    if p == "~" {
        if let Some(h) = dirs::home_dir() {
            return h;
        }
    }
    PathBuf::from(p)
}

// ---------------------------------------------------------------- load / save

pub fn default_config_path() -> PathBuf {
    if let Ok(p) = std::env::var("SYMFERIUM_CONFIG") {
        return PathBuf::from(p);
    }
    expand(&app_support()).join("symferium.toml")
}

/// Load the config, or create it from probe-based defaults when it does not exist yet.
pub fn load_or_create(path: &Path, machine: &MachineInfo) -> Result<Config> {
    if path.exists() {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        // New engine profiles and classes added in later versions appear without overwriting user edits.
        for (k, v) in default_engines() {
            cfg.engines.entry(k).or_insert(v);
        }
        for (k, v) in default_classes(machine.perf_cores.max(1)) {
            cfg.classes.entry(k).or_insert(v);
        }
        cfg.validate()?;
        Ok(cfg)
    } else {
        let cfg = Config::defaults_for(machine);
        save(path, &cfg)?;
        Ok(cfg)
    }
}

/// Atomic save with 0600 permissions (the file can hold tokens).
pub fn save(path: &Path, cfg: &Config) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let text = format!(
        "# Symferium configuration. Every value here is also editable in the UI (http://127.0.0.1:{}/).\n\n{}",
        cfg.server.control_port,
        toml::to_string_pretty(cfg)?
    );
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, text)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------- schema

#[derive(Debug, Clone, Serialize)]
pub struct SettingSpec {
    pub key: String,
    pub section: String,
    /// bool, int, float, string, string_list, map
    pub kind: String,
    pub description: String,
    pub unit: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub choices: Option<Vec<String>>,
    pub agent_writable: bool,
    pub restart_required: bool,
    pub secret: bool,
    pub default: serde_json::Value,
}

struct Meta {
    pattern: &'static str,
    desc: &'static str,
    unit: Option<&'static str>,
    min: Option<f64>,
    max: Option<f64>,
    choices: Option<&'static [&'static str]>,
    agent: bool,
    restart: bool,
    secret: bool,
}

const fn m(pattern: &'static str, desc: &'static str) -> Meta {
    Meta { pattern, desc, unit: None, min: None, max: None, choices: None, agent: false, restart: false, secret: false }
}
impl Meta {
    const fn unit(mut self, u: &'static str) -> Self {
        self.unit = Some(u);
        self
    }
    const fn range(mut self, lo: f64, hi: f64) -> Self {
        self.min = Some(lo);
        self.max = Some(hi);
        self
    }
    const fn choices(mut self, c: &'static [&'static str]) -> Self {
        self.choices = Some(c);
        self
    }
    const fn agent(mut self) -> Self {
        self.agent = true;
        self
    }
    const fn restart(mut self) -> Self {
        self.restart = true;
        self
    }
    const fn secret(mut self) -> Self {
        self.secret = true;
        self
    }
}

const PRESSURE: &[&str] = &["warn", "critical", "never"];
const POLICY: &[&str] = &["refuse", "unload_lowest"];
const FA: &[&str] = &["on", "off", "auto"];
const CACHE: &[&str] = &["f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "turbo4", "turbo3", "turbo2"];
const POOL: &[&str] = &["none", "mean", "cls", "last", "rank"];
const PTYPE: &[&str] = &["Background", "Adaptive", "Standard", "Interactive"];
const ADAPTERS: &[&str] = &["llamacpp", "mlx", "whispercpp", "tts", "command"];

static META: &[Meta] = &[
    m("server.host", "Address every port binds to. Keep 127.0.0.1 unless you know why.").restart(),
    m("server.control_port", "Port of the control API and the UI.").range(1024.0, 65535.0).restart(),
    m("server.model_port_start", "First public port handed to a loaded model.").range(1024.0, 65535.0),
    m("server.model_port_end", "Last public port handed to a loaded model.").range(1024.0, 65535.0),
    m("server.internal_port_start", "First private port an engine binds behind the proxy.").range(1024.0, 65535.0),
    m("server.internal_port_end", "Last private port an engine binds behind the proxy.").range(1024.0, 65535.0),
    m("server.admin_token", "Bearer token required for the whole control API. Empty = no auth.").secret(),
    m("server.agent_token", "Bearer token that may call /api/tools only. Empty = no auth.").secret(),
    m("server.proxy_wait_ready_s", "How long a request waits on a model port while its engine restarts.")
        .unit("s")
        .range(0.0, 3600.0)
        .agent(),
    m("server.proxy_max_json_body", "Largest JSON body the proxy inspects.").unit("bytes").range(1024.0, 1e9),
    m("paths.models_dir", "Where downloads go. Scanned for models."),
    m("paths.extra_model_dirs", "Extra folders scanned for models, read-only."),
    m("paths.state_dir", "Saved model set, learned estimates, bench results.").restart(),
    m("paths.logs_dir", "Engine logs.").restart(),
    m("paths.python_env", "Python env (uv) holding mlx-lm, mlx-vlm and TTS packages."),
    m("paths.sidecar_dir", "Folder with Symferium's sidecar scripts."),
    m("governor.min_free_gb", "Keep at least this much memory free after any load.").unit("GB").range(1.0, 512.0),
    m("governor.max_total_gb", "Upper bound for all loaded models. 0 = min(total − min free, Metal working set).")
        .unit("GB")
        .range(0.0, 512.0),
    m("governor.engine_overhead_mb", "Fixed overhead added to every estimate.").unit("MB").range(0.0, 8192.0),
    m("governor.refuse_on_pressure", "Refuse loads at or above this macOS memory pressure.").choices(PRESSURE),
    m("governor.pressure_policy", "On critical pressure: only refuse new loads, or also unload the lowest-priority model.").choices(POLICY),
    m("governor.watchdog_interval_s", "How often memory pressure is checked.").unit("s").range(1.0, 3600.0),
    m("governor.exclusive_large", "A large model loads only when no other chat model is loaded."),
    m("governor.learn_corrections", "Learn estimate error per engine from measured memory."),
    m("governor.measure_after_s", "Seconds after ready before memory is measured for learning.").unit("s").range(1.0, 3600.0),
    m("supervisor.health_interval_ms", "Health-check interval.").unit("ms").range(100.0, 600000.0),
    m("supervisor.health_failures_before_restart", "Failed health checks in a row before a restart.").range(1.0, 1000.0),
    m("supervisor.startup_timeout_s", "Longest an engine may take to become healthy.").unit("s").range(5.0, 7200.0),
    m("supervisor.max_restarts", "Restarts before a model is marked failed.").range(0.0, 1000.0),
    m("supervisor.restart_backoff_ms", "First restart delay; doubles each time.").unit("ms").range(0.0, 600000.0),
    m("supervisor.stop_timeout_s", "Grace period between SIGTERM and SIGKILL.").unit("s").range(0.0, 600.0),
    m("supervisor.log_lines", "Log lines kept in memory per model.").range(10.0, 1_000_000.0),
    m("autostart.restore_model_set", "Load the saved model set when the daemon starts."),
    m("autostart.delay_s", "Wait after daemon start before restoring models.").unit("s").range(0.0, 3600.0),
    m("autostart.stagger_s", "Pause between restored models.").unit("s").range(0.0, 600.0),
    m("autostart.launchd_label", "launchd label of the LaunchAgent."),
    m("autostart.process_type", "launchd ProcessType for the daemon.").choices(PTYPE),
    m("autostart.nice", "Nice value of the daemon (inherited by engines).").range(0.0, 20.0),
    m("catalog.hf_endpoint", "Hugging Face endpoint."),
    m("catalog.hf_token", "Hugging Face token for gated models.").secret(),
    m("catalog.authors", "Community quant publishers searched beside the originals."),
    m("catalog.search_limit", "Results per catalog search.").range(1.0, 500.0).agent(),
    m("catalog.cache_ttl_s", "How long catalog answers are cached.").unit("s").range(0.0, 604800.0),
    m("catalog.small_max_b", "Small class: below this many billion parameters.").unit("B").range(0.1, 1000.0),
    m("catalog.standard_min_b", "Standard class lower bound.").unit("B").range(0.1, 1000.0),
    m("catalog.standard_max_b", "Standard class upper bound (inclusive).").unit("B").range(0.1, 1000.0),
    m("catalog.large_max_b", "Largest model offered.").unit("B").range(0.1, 2000.0),
    m("catalog.quant_preference", "GGUF quants, best first."),
    m("download.concurrent", "Downloads running at once.").range(1.0, 16.0).agent(),
    m("download.retries", "Retries per file before a download fails.").range(0.0, 100.0),
    m("download.verify_sha256", "Check every file against the Hugging Face sha256."),
    m("download.timeout_s", "Network timeout per request.").unit("s").range(1.0, 3600.0),
    m("bench.prompt", "Prompt used by the benchmark.").agent(),
    m("bench.max_tokens", "Tokens generated per benchmark run.").range(1.0, 32768.0).agent(),
    m("bench.runs", "Benchmark runs per model.").range(1.0, 100.0).agent(),
    m("classes.*.engine", "Default engine profile for GGUF/whisper/onnx files of this class."),
    m("classes.*.engine_mlx", "Default engine profile for MLX weights of this class."),
    m("classes.*.ctx", "Default context length.").unit("tokens").range(256.0, 1_048_576.0).agent(),
    m("classes.*.gpu_layers", "Layers offloaded to the GPU (999 = all).").range(-1.0, 999.0),
    m("classes.*.flash_attn", "Flash attention.").choices(FA).agent(),
    m("classes.*.cache_type_k", "KV cache type for keys.").choices(CACHE).agent(),
    m("classes.*.cache_type_v", "KV cache type for values.").choices(CACHE).agent(),
    m("classes.*.parallel", "Parallel sequences (slots).").range(1.0, 64.0).agent(),
    m("classes.*.batch", "Logical batch size.").range(1.0, 65536.0),
    m("classes.*.ubatch", "Physical batch size.").range(1.0, 65536.0),
    m("classes.*.threads", "CPU threads.").range(1.0, 64.0).agent(),
    m("classes.*.pooling", "Embedding pooling.").choices(POOL),
    m("classes.*.max_tokens", "Default max tokens for MLX servers.").range(1.0, 1_048_576.0).agent(),
    m("classes.*.ctx_checkpoints", "llama.cpp context checkpoints for SWA/hybrid models; each costs one SWA state.")
        .range(0.0, 64.0)
        .agent(),
    m("classes.*.cache_ram_mb", "llama.cpp host prompt cache.").unit("MB").range(0.0, 65536.0).agent(),
    m("classes.*.priority", "Lower = unloaded first under memory pressure.").range(0.0, 100.0).agent(),
    m("engines.*.adapter", "Adapter family.").choices(ADAPTERS),
    m("engines.*.enabled", "Profile can be used."),
    m("engines.*.binary", "Executable. {python_env} and ~ are expanded; bare names are looked up in PATH."),
    m("engines.*.args", "Arguments. Placeholders: {model} {mmproj} {host} {port} {ctx} {alias} and every class setting."),
    m("engines.*.args_vision", "Added when a projector (mmproj) is loaded."),
    m("engines.*.args_embed", "Added for embedding models."),
    m("engines.*.args_rerank", "Added for reranker models."),
    m("engines.*.extra_args", "Appended to every launch of this profile."),
    m("engines.*.env", "Environment variables for the engine process."),
    m("engines.*.health_path", "Path polled on the engine's private port."),
    m("engines.*.rewrite_model_field", "Replace the JSON \"model\" field with the served model."),
    m("engines.*.kinds", "Model kinds this profile serves."),
    m("engines.*.formats", "Model formats this profile loads."),
    m("symbiosis.enabled", "Symbiosis layer (v2). Off: requests pass through untouched."),
    m("symbiosis.qdrant_url", "Qdrant endpoint for Ghost and Whisper."),
    m("symbiosis.graph_url", "Graph store endpoint for Heron and Nexus."),
    m("symbiosis.memory_dir", "Raw memories folder."),
    m("symbiosis.context_threshold", "Fraction of the context window at which Whisper compacts.").range(0.1, 0.99),
];

fn pattern_matches(pattern: &str, key: &str) -> bool {
    let p: Vec<&str> = pattern.split('.').collect();
    let k: Vec<&str> = key.split('.').collect();
    p.len() == k.len() && p.iter().zip(&k).all(|(a, b)| *a == "*" || a == b)
}

fn kind_of(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::Boolean(_) => "bool",
        toml::Value::Integer(_) => "int",
        toml::Value::Float(_) => "float",
        toml::Value::String(_) => "string",
        toml::Value::Array(_) => "string_list",
        toml::Value::Table(_) => "map",
        toml::Value::Datetime(_) => "string",
    }
}

/// Every leaf of the config, described. Map-typed fields (engines.*.env) are one leaf.
pub fn schema(cfg: &Config) -> Vec<SettingSpec> {
    let root = toml::Value::try_from(cfg).expect("config serialises");
    let defaults = toml::Value::try_from(Config::default()).expect("default serialises");
    let mut out = Vec::new();
    walk(&root, "", &mut |key, v| {
        let section = key.split('.').next().unwrap_or("").to_string();
        let meta = META.iter().find(|m| pattern_matches(m.pattern, key));
        let default = lookup(&defaults, key).map(toml_to_json).unwrap_or(serde_json::Value::Null);
        out.push(SettingSpec {
            key: key.to_string(),
            section,
            kind: kind_of(v).into(),
            description: meta.map(|m| m.desc.to_string()).unwrap_or_default(),
            unit: meta.and_then(|m| m.unit.map(String::from)),
            min: meta.and_then(|m| m.min),
            max: meta.and_then(|m| m.max),
            choices: meta.and_then(|m| m.choices.map(|c| c.iter().map(|s| s.to_string()).collect())),
            agent_writable: meta.is_some_and(|m| m.agent),
            restart_required: meta.is_some_and(|m| m.restart),
            secret: meta.is_some_and(|m| m.secret),
            default,
        });
    });
    out
}

fn walk(v: &toml::Value, prefix: &str, f: &mut dyn FnMut(&str, &toml::Value)) {
    match v {
        toml::Value::Table(t) => {
            // engines.*.env is a free-form map: treat as a leaf.
            let is_leaf_map = prefix.ends_with(".env");
            if is_leaf_map {
                f(prefix, v);
                return;
            }
            for (k, child) in t {
                let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                walk(child, &key, f);
            }
        }
        _ => f(prefix, v),
    }
}

fn lookup<'a>(root: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    let mut cur = root;
    for part in key.split('.') {
        cur = cur.as_table()?.get(part)?;
    }
    Some(cur)
}

pub fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
}

fn value_as_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
}

/// Read one setting as JSON.
pub fn get_value(cfg: &Config, key: &str) -> Result<serde_json::Value> {
    let root = toml::Value::try_from(cfg)?;
    lookup(&root, key).map(toml_to_json).ok_or_else(|| anyhow!("unknown setting {key}"))
}

/// Config as JSON with secrets masked.
pub fn masked_json(cfg: &Config) -> serde_json::Value {
    let mut v = serde_json::to_value(cfg).unwrap_or_default();
    for spec in schema(cfg).iter().filter(|s| s.secret) {
        let mut cur = &mut v;
        let parts: Vec<&str> = spec.key.split('.').collect();
        for p in &parts[..parts.len() - 1] {
            cur = &mut cur[*p];
        }
        let last = parts[parts.len() - 1];
        if cur[last].as_str().is_some_and(|s| !s.is_empty()) {
            cur[last] = serde_json::Value::String("••••••".into());
        }
    }
    v
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    User,
    Agent,
}

/// Apply one change. Validates the key, the type, the range and the agent policy, and returns the
/// new config. The caller saves and broadcasts it.
pub fn apply_change(cfg: &Config, key: &str, value: serde_json::Value, actor: Actor) -> Result<Config> {
    let specs = schema(cfg);
    // Allow adding entries under engines.* / classes.* by matching the pattern of a sibling.
    let spec =
        specs.iter().find(|s| s.key == key).cloned().or_else(|| new_entry_spec(key)).ok_or_else(|| anyhow!("unknown setting {key}"))?;
    if actor == Actor::Agent && !spec.agent_writable {
        bail!("{key} is not agent_writable");
    }
    if spec.secret && value.as_str() == Some("••••••") {
        return Ok(cfg.clone()); // masked value echoed back by the UI: no change
    }
    if let Some(n) = value.as_f64() {
        if spec.min.is_some_and(|m| n < m) || spec.max.is_some_and(|m| n > m) {
            bail!("{key} = {n} is outside the allowed range {:?}..{:?}", spec.min, spec.max);
        }
    }
    if let (Some(choices), Some(sv)) = (&spec.choices, value.as_str()) {
        if !choices.iter().any(|c| c == sv) {
            bail!("{key} must be one of {}", choices.join(", "));
        }
    }
    let mut root = toml::Value::try_from(cfg)?;
    let new_val = json_to_toml(&value, &spec.kind)?;
    let parts: Vec<&str> = key.split('.').collect();
    let mut cur = &mut root;
    for p in &parts[..parts.len() - 1] {
        let t = cur.as_table_mut().ok_or_else(|| anyhow!("{key}: parent is not a table"))?;
        cur = t.entry(p.to_string()).or_insert_with(|| toml::Value::Table(Default::default()));
    }
    cur.as_table_mut().ok_or_else(|| anyhow!("{key}: parent is not a table"))?.insert(parts[parts.len() - 1].to_string(), new_val);
    let new_cfg: Config = root.try_into().map_err(|e| anyhow!("{key}: {e}"))?;
    new_cfg.validate()?;
    Ok(new_cfg)
}

fn new_entry_spec(key: &str) -> Option<SettingSpec> {
    let meta = META.iter().find(|m| m.pattern.contains('*') && pattern_matches(m.pattern, key))?;
    let probe = Config::default();
    let sample_key = key
        .split('.')
        .enumerate()
        .map(|(i, p)| {
            if i == 1 {
                if key.starts_with("engines.") {
                    "llamacpp"
                } else {
                    "standard"
                }
            } else {
                p
            }
        })
        .collect::<Vec<_>>()
        .join(".");
    let root = toml::Value::try_from(&probe).ok()?;
    let kind = lookup(&root, &sample_key).map(kind_of).unwrap_or("string");
    Some(SettingSpec {
        key: key.into(),
        section: key.split('.').next()?.into(),
        kind: kind.into(),
        description: meta.desc.into(),
        unit: meta.unit.map(String::from),
        min: meta.min,
        max: meta.max,
        choices: meta.choices.map(|c| c.iter().map(|s| s.to_string()).collect()),
        agent_writable: meta.agent,
        restart_required: meta.restart,
        secret: meta.secret,
        default: serde_json::Value::Null,
    })
}

fn json_to_toml(v: &serde_json::Value, kind: &str) -> Result<toml::Value> {
    use serde_json::Value as J;
    Ok(match (kind, v) {
        ("bool", J::Bool(b)) => toml::Value::Boolean(*b),
        ("bool", J::String(s)) => toml::Value::Boolean(s == "true"),
        ("int", J::Number(n)) => toml::Value::Integer(
            n.as_i64()
                .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                .ok_or_else(|| anyhow!("expected an integer"))?,
        ),
        ("int", J::String(s)) => toml::Value::Integer(s.trim().parse()?),
        ("float", J::Number(n)) => toml::Value::Float(n.as_f64().unwrap_or(0.0)),
        ("float", J::String(s)) => toml::Value::Float(s.trim().parse()?),
        ("string", J::String(s)) => toml::Value::String(s.clone()),
        ("string", J::Number(n)) => toml::Value::String(n.to_string()),
        ("string_list", J::Array(a)) => toml::Value::Array(
            a.iter().map(|x| toml::Value::String(x.as_str().map(String::from).unwrap_or_else(|| x.to_string()))).collect(),
        ),
        ("string_list", J::String(s)) => toml::Value::Array(
            s.lines().flat_map(|l| l.split(',')).map(str::trim).filter(|x| !x.is_empty()).map(|x| toml::Value::String(x.into())).collect(),
        ),
        ("map", J::Object(o)) => {
            let mut t = toml::map::Map::new();
            for (k, x) in o {
                t.insert(k.clone(), toml::Value::String(x.as_str().map(String::from).unwrap_or_else(|| x.to_string())));
            }
            toml::Value::Table(t)
        }
        (k, v) => bail!("expected a {k}, got {v}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_roundtrip_through_toml() {
        let c = Config::default();
        let text = toml::to_string_pretty(&c).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(c, back);
        c.validate().unwrap();
    }

    #[test]
    fn every_leaf_has_a_description() {
        let missing: Vec<_> = schema(&Config::default()).into_iter().filter(|s| s.description.is_empty()).map(|s| s.key).collect();
        assert!(missing.is_empty(), "undocumented settings: {missing:?}");
    }

    #[test]
    fn change_is_validated() {
        let c = Config::default();
        let c2 = apply_change(&c, "classes.standard.ctx", serde_json::json!(65536), Actor::Agent).unwrap();
        assert_eq!(c2.classes["standard"].ctx, 65536);
        assert!(apply_change(&c, "classes.standard.ctx", serde_json::json!(10), Actor::User).is_err());
        assert!(apply_change(&c, "governor.min_free_gb", serde_json::json!(2.0), Actor::Agent).is_err(), "not agent writable");
        assert!(apply_change(&c, "governor.pressure_policy", serde_json::json!("explode"), Actor::User).is_err());
        assert!(apply_change(&c, "nope.key", serde_json::json!(1), Actor::User).is_err());
        let c3 = apply_change(&c, "governor.min_free_gb", serde_json::json!("6.5"), Actor::User).unwrap();
        assert_eq!(c3.governor.min_free_gb, 6.5);
        // overlapping port ranges are rejected
        assert!(apply_change(&c, "server.internal_port_start", serde_json::json!(7750), Actor::User).is_err());
    }

    #[test]
    fn new_engine_profile_can_be_added() {
        let c = Config::default();
        let c2 = apply_change(&c, "engines.myengine.binary", serde_json::json!("/opt/x/server"), Actor::User).unwrap();
        assert_eq!(c2.engines["myengine"].binary, "/opt/x/server");
        assert_eq!(c2.engines["myengine"].health_path, "/health");
    }

    #[test]
    fn secrets_are_masked() {
        let mut c = Config::default();
        c.catalog.hf_token = "hf_secret".into();
        let j = masked_json(&c);
        assert_eq!(j["catalog"]["hf_token"], "••••••");
        let c2 = apply_change(&c, "catalog.hf_token", serde_json::json!("••••••"), Actor::User).unwrap();
        assert_eq!(c2.catalog.hf_token, "hf_secret");
    }
}
