//! Control API on :7700 and the embedded UI.

use crate::catalog::{self, FileRef, SearchQuery};
use crate::config::{self, Actor};
use crate::runtime::{LoadError, Runtime};
use crate::state::LoadRequest;
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use futures::Stream;
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;

type Rt = Arc<Runtime>;

pub struct ApiError(StatusCode, Value);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::BAD_REQUEST, json!({"error": format!("{e:#}")}))
    }
}

impl From<LoadError> for ApiError {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Refused(d) => ApiError(
                StatusCode::CONFLICT,
                json!({"error": "refused by the governor", "reasons": d.reasons, "estimate": d.estimate, "budget": d.budget}),
            ),
            LoadError::Invalid(e) => e.into(),
        }
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

pub fn router(rt: Rt) -> Router {
    let api = Router::new()
        .route("/api/machine", get(machine))
        .route("/api/metrics", get(metrics))
        .route("/api/models", get(models))
        .route("/api/models/rescan", post(rescan))
        .route("/api/estimate", post(estimate))
        .route("/api/load", post(load))
        .route("/api/unload", post(unload))
        .route("/api/instances/{id}/logs", get(logs))
        .route("/api/model-set", get(model_set))
        .route("/api/settings", get(settings).put(put_settings))
        .route("/api/settings/reset", post(reset_setting))
        .route("/api/catalog/search", get(catalog_search))
        .route("/api/catalog/model/{*repo}", get(catalog_detail))
        .route("/api/downloads", get(downloads).post(start_download))
        .route("/api/downloads/{id}", delete(cancel_download))
        .route("/api/bench", get(bench_results).post(bench_run))
        .route("/api/bench/compare", post(bench_compare))
        .route("/api/events", get(events))
        .route("/api/events/history", get(events_history))
        .route("/api/tools", get(tools_manifest))
        .route("/api/tools/{name}", post(tools_call))
        .route("/api/version", get(version))
        .route("/api/shutdown", put(shutdown))
        .layer(middleware::from_fn_with_state(rt.clone(), auth));
    Router::new()
        .route("/", get(ui_index))
        .route("/app.js", get(ui_js))
        .route("/style.css", get(ui_css))
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .merge(api)
        .with_state(rt)
}

async fn auth(State(rt): State<Rt>, req: Request, next: Next) -> Response {
    let cfg = rt.cfg();
    let path = req.uri().path().to_string();
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(String::from)
        .or_else(|| req.uri().query().and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("token=").map(String::from))));
    let admin_ok = cfg.server.admin_token.is_empty() || token.as_deref() == Some(cfg.server.admin_token.as_str());
    let is_tools = path.starts_with("/api/tools");
    let agent_ok = is_tools && (cfg.server.agent_token.is_empty() && cfg.server.admin_token.is_empty() || token.as_deref() == Some(cfg.server.agent_token.as_str()) && !cfg.server.agent_token.is_empty());
    if admin_ok || agent_ok {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, Json(json!({"error": "missing or wrong bearer token"}))).into_response()
    }
}

async fn version() -> Json<Value> {
    Json(json!({"name": "symferium", "version": env!("CARGO_PKG_VERSION")}))
}

async fn machine(State(rt): State<Rt>) -> Json<Value> {
    Json(json!({"machine": rt.probe.snapshot(), "budget": rt.budget(), "corrections": rt.state.data.lock().corrections}))
}

async fn metrics(State(rt): State<Rt>) -> Json<Value> {
    Json(json!({"machine": rt.probe.snapshot(), "budget": rt.budget(), "instances": rt.list()}))
}

async fn models(State(rt): State<Rt>) -> Json<Value> {
    let lib = rt.library.list();
    let prefs: Vec<Value> = lib
        .iter()
        .map(|e| e.family.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter_map(|f| rt.state.preferred_engine(&f).map(|(engine, model)| json!({"family": f, "engine": engine, "model": model})))
        .collect();
    Json(json!({"local": lib, "loaded": rt.list(), "engine_preferences": prefs}))
}

async fn rescan(State(rt): State<Rt>) -> Json<Value> {
    let list = rt.library.scan(&rt.cfg());
    Json(json!({"count": list.len()}))
}

async fn estimate(State(rt): State<Rt>, Json(req): Json<LoadRequest>) -> ApiResult {
    let p = rt.plan(&req)?;
    Ok(Json(json!({"engine": p.engine, "class": p.class, "ctx": p.ctx, "mmproj": p.mmproj, "allowed": p.decision.allowed,
        "reasons": p.decision.reasons, "estimate": p.estimate, "budget": p.decision.budget, "params": p.params})))
}

async fn load(State(rt): State<Rt>, Json(req): Json<LoadRequest>) -> ApiResult {
    Ok(Json(json!(rt.load(req).await?)))
}

#[derive(Deserialize)]
struct UnloadReq {
    id: String,
    #[serde(default = "yes")]
    forget: bool,
}
fn yes() -> bool {
    true
}

async fn unload(State(rt): State<Rt>, Json(r): Json<UnloadReq>) -> ApiResult {
    rt.unload(&r.id, r.forget).await?;
    Ok(Json(json!({"unloaded": r.id})))
}

#[derive(Deserialize)]
struct LogsQ {
    n: Option<usize>,
}

async fn logs(State(rt): State<Rt>, Path(id): Path<String>, Query(q): Query<LogsQ>) -> ApiResult {
    let inst = rt.get(&id).ok_or_else(|| anyhow::anyhow!("{id} is not loaded"))?;
    let l = inst.logs.lock();
    let n = q.n.unwrap_or(200).min(l.len());
    Ok(Json(json!({"id": id, "lines": l.iter().skip(l.len() - n).collect::<Vec<_>>()})))
}

async fn model_set(State(rt): State<Rt>) -> Json<Value> {
    Json(json!(rt.state.data.lock().model_set))
}

async fn settings(State(rt): State<Rt>) -> Json<Value> {
    let cfg = rt.cfg();
    Json(json!({"values": config::masked_json(&cfg), "schema": config::schema(&cfg), "path": rt.config_path}))
}

#[derive(Deserialize)]
struct SettingsPut {
    key: Option<String>,
    value: Option<Value>,
    /// Several changes at once: {"a.b": 1, "c.d": "x"}; all or nothing.
    changes: Option<serde_json::Map<String, Value>>,
}

async fn put_settings(State(rt): State<Rt>, Json(p): Json<SettingsPut>) -> ApiResult {
    let mut cfg = rt.cfg();
    let mut changed = vec![];
    if let (Some(k), Some(v)) = (p.key, p.value) {
        cfg = config::apply_change(&cfg, &k, v, Actor::User)?;
        changed.push(k);
    }
    for (k, v) in p.changes.unwrap_or_default() {
        cfg = config::apply_change(&cfg, &k, v, Actor::User)?;
        changed.push(k);
    }
    let restart: Vec<String> = config::schema(&cfg).into_iter().filter(|s| s.restart_required && changed.contains(&s.key)).map(|s| s.key).collect();
    rt.set_config(cfg)?;
    Ok(Json(json!({"ok": true, "changed": changed, "restart_required": restart})))
}

#[derive(Deserialize)]
struct ResetReq {
    key: Option<String>,
}

async fn reset_setting(State(rt): State<Rt>, Json(r): Json<ResetReq>) -> ApiResult {
    let fresh = config::Config::defaults_for(&rt.probe.snapshot());
    let cfg = match r.key {
        Some(k) => {
            let v = config::get_value(&fresh, &k)?;
            config::apply_change(&rt.cfg(), &k, v, Actor::User)?
        }
        None => fresh,
    };
    rt.set_config(cfg)?;
    Ok(Json(json!({"ok": true})))
}

async fn catalog_search(State(rt): State<Rt>, Query(q): Query<SearchQuery>) -> ApiResult {
    Ok(Json(json!(catalog::search(&rt, &q).await?)))
}

async fn catalog_detail(State(rt): State<Rt>, Path(repo): Path<String>) -> ApiResult {
    Ok(Json(json!(catalog::detail(&rt, &repo).await?)))
}

async fn downloads(State(rt): State<Rt>) -> Json<Value> {
    Json(json!(rt.downloads.list()))
}

#[derive(Deserialize)]
struct DownloadReq {
    repo: String,
    /// Either explicit files…
    files: Option<Vec<FileRef>>,
    /// …or a quant name (and the projector, unless include_mmproj is false).
    quant: Option<String>,
    include_mmproj: Option<bool>,
}

async fn start_download(State(rt): State<Rt>, Json(r): Json<DownloadReq>) -> ApiResult {
    if let Some(files) = r.files {
        let id = crate::download::start(&rt, &r.repo, files)?;
        return Ok(Json(json!({"download_id": id})));
    }
    let args = json!({"repo": r.repo, "quant": r.quant, "include_mmproj": r.include_mmproj.unwrap_or(true)});
    Ok(Json(crate::tools::call(&rt, "download", args).await?))
}

async fn cancel_download(State(rt): State<Rt>, Path(id): Path<String>) -> ApiResult {
    Ok(Json(json!({"cancelled": rt.downloads.cancel(&id)})))
}

async fn bench_results(State(rt): State<Rt>) -> Json<Value> {
    Json(json!(rt.state.data.lock().bench))
}

#[derive(Deserialize)]
struct BenchReq {
    id: String,
    prompt: Option<String>,
    max_tokens: Option<u32>,
    runs: Option<u32>,
}

async fn bench_run(State(rt): State<Rt>, Json(b): Json<BenchReq>) -> ApiResult {
    Ok(Json(json!(crate::bench::run(&rt, &b.id, b.prompt, b.max_tokens, b.runs).await?)))
}

#[derive(Deserialize)]
struct CompareReq {
    models: Vec<String>,
    ctx: Option<u32>,
}

async fn bench_compare(State(rt): State<Rt>, Json(c): Json<CompareReq>) -> ApiResult {
    crate::bench::compare(&rt, c.models.clone(), c.ctx);
    Ok(Json(json!({"started": c.models})))
}

async fn events(State(rt): State<Rt>) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let mut rx = rt.events.subscribe();
    let rt2 = rt.clone();
    let stream = async_stream::stream! {
        // A metrics tick every 2 s keeps the UI live without polling.
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            tokio::select! {
                ev = rx.recv() => match ev {
                    Ok(e) => yield Ok(SseEvent::default().event("event").json_data(&e).unwrap_or_default()),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                },
                _ = tick.tick() => {
                    let snap = json!({"machine": rt2.probe.snapshot(), "budget": rt2.budget(), "instances": rt2.list(), "downloads": rt2.downloads.list()});
                    yield Ok(SseEvent::default().event("metrics").json_data(&snap).unwrap_or_default());
                }
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn events_history(State(rt): State<Rt>) -> Json<Value> {
    Json(json!(rt.events.history()))
}

async fn tools_manifest() -> Json<Value> {
    Json(crate::tools::manifest())
}

async fn tools_call(State(rt): State<Rt>, Path(name): Path<String>, body: Option<Json<Value>>) -> ApiResult {
    let args = body.map(|b| b.0).unwrap_or(Value::Null);
    match crate::tools::call(&rt, &name, args).await {
        Ok(v) => Ok(Json(json!({"ok": true, "result": v}))),
        Err(e) => {
            let msg = format!("{e:#}");
            let detail: Value = serde_json::from_str(&msg).unwrap_or(json!(msg));
            let code = if detail.get("refused").is_some() { StatusCode::CONFLICT } else { StatusCode::BAD_REQUEST };
            Err(ApiError(code, json!({"ok": false, "error": detail})))
        }
    }
}

async fn shutdown(State(rt): State<Rt>) -> Json<Value> {
    let rt2 = rt.clone();
    tokio::spawn(async move {
        rt2.unload_all().await;
        std::process::exit(0);
    });
    Json(json!({"ok": true}))
}

const INDEX: &str = include_str!("ui/index.html");
const APP_JS: &str = include_str!("ui/app.js");
const STYLE: &str = include_str!("ui/style.css");

async fn ui_index() -> Html<&'static str> {
    Html(INDEX)
}
async fn ui_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8")], APP_JS)
}
async fn ui_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], STYLE)
}
