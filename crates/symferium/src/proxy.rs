//! Public model ports. Symferium owns each port and forwards to the engine's private port, so the
//! port never changes while an engine restarts, every engine answers /health and /v1/models the
//! same way, tok/s is measured identically for every engine, and the symbiosis middleware has one
//! place to hook in. Streaming responses are passed through chunk by chunk.

use crate::events::now_ms;
use crate::runtime::{Instance, Runtime};
use crate::symbiosis::{RequestCtx, ResponseSummary};
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct PState {
    rt: Arc<Runtime>,
    inst: Arc<Instance>,
}

pub fn serve(rt: Arc<Runtime>, inst: Arc<Instance>, listener: tokio::net::TcpListener) {
    let stop = inst.stop.clone();
    let app = axum::Router::new().fallback(handle).with_state(PState { rt, inst });
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).with_graceful_shutdown(async move { stop.cancelled().await }).await;
    });
}

const HOP: &[&str] =
    &["host", "connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "content-length"];

fn err(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": "symferium"}}))).into_response()
}

async fn handle(State(s): State<PState>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    let inst = &s.inst;
    if path == "/health" {
        let st = inst.state();
        let code = if st == "ready" { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
        return (
            code,
            Json(json!({"status": if st == "ready" { "ok" } else { st.as_str() }, "model": inst.id, "engine": inst.spec.engine})),
        )
            .into_response();
    }
    if req.method() == Method::GET && (path == "/v1/models" || path == "/models") {
        return Json(json!({"object": "list", "data": [{
            "id": inst.id, "object": "model", "owned_by": "symferium", "created": inst.status.read().started_ms / 1000,
            "meta": {"engine": inst.spec.engine, "kind": inst.model.kind, "class": inst.class, "ctx": inst.ctx,
                     "vision": inst.mmproj.is_some() || inst.model.flags.vision, "format": inst.model.format,
                     "flags": inst.model.flags}
        }]}))
        .into_response();
    }
    let cfg = s.rt.cfg();
    // Wait while the engine starts or restarts.
    if inst.state() != "ready" {
        let mut rx = inst.ready_tx.subscribe();
        let wait = Duration::from_secs(cfg.server.proxy_wait_ready_s);
        let ok = tokio::time::timeout(wait, async {
            loop {
                if *rx.borrow_and_update() {
                    return true;
                }
                if inst.state() == "failed" || inst.stop.is_cancelled() {
                    return false;
                }
                tokio::select! {
                    r = rx.changed() => if r.is_err() { return false },
                    _ = tokio::time::sleep(Duration::from_millis(250)) => {}
                }
            }
        })
        .await
        .unwrap_or(false);
        if !ok {
            let why = inst.status.read().last_error.clone().unwrap_or_default();
            return err(StatusCode::SERVICE_UNAVAILABLE, format!("{} is {} {}", inst.id, inst.state(), why));
        }
    }
    let started = Instant::now();
    inst.metrics.lock().in_flight += 1;
    let resp = forward(&s, req, &path, &cfg, started).await;
    if resp.status().is_server_error() || resp.status().is_client_error() {
        let mut m = inst.metrics.lock();
        m.errors += 1;
    }
    resp
}

async fn forward(s: &PState, req: Request, path: &str, cfg: &crate::config::Config, started: Instant) -> Response {
    let inst = s.inst.clone();
    let (parts, body) = req.into_parts();
    let pq = parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or(path);
    let url = format!("http://127.0.0.1:{}{}", inst.internal_port, pq);
    let run_id = parts.headers.get("x-symferium-run").and_then(|v| v.to_str().ok()).map(String::from);
    let is_json = parts.headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.starts_with("application/json"));
    let ctx = RequestCtx {
        instance: inst.id.clone(),
        model: inst.model.id.clone(),
        path: path.to_string(),
        run_id: run_id.clone(),
        ctx: inst.ctx,
    };
    let mut headers = reqwest::header::HeaderMap::new();
    for (k, v) in parts.headers.iter() {
        if !HOP.contains(&k.as_str()) {
            headers.insert(k.clone(), v.clone());
        }
    }
    let mut streaming_req = false;
    let rb = s.rt.http.request(parts.method.clone(), &url).headers(headers);
    let rb = if is_json && parts.method != Method::GET {
        let bytes = match axum::body::to_bytes(body, cfg.server.proxy_max_json_body).await {
            Ok(b) => b,
            Err(e) => return finish_err(&inst, StatusCode::PAYLOAD_TOO_LARGE, format!("request body: {e}")),
        };
        match serde_json::from_slice::<serde_json::Value>(&bytes) {
            Ok(mut v) if v.is_object() => {
                if inst.spec.rewrite_model_field {
                    v["model"] = json!(inst.spec.served_model);
                }
                streaming_req = v.get("stream").and_then(|x| x.as_bool()).unwrap_or(false);
                if streaming_req && path.ends_with("/completions") && v.get("stream_options").is_none() {
                    v["stream_options"] = json!({"include_usage": true});
                }
                let symbiosis = cfg.symbiosis.enabled && run_id.is_some();
                if symbiosis {
                    let chain = s.rt.middleware.read().clone();
                    chain.before(&ctx, &mut v).await;
                }
                rb.body(serde_json::to_vec(&v).unwrap_or_default())
            }
            _ => rb.body(bytes),
        }
    } else {
        rb.body(reqwest::Body::wrap_stream(body.into_data_stream()))
    };
    let upstream = match rb.send().await {
        Ok(r) => r,
        Err(e) => return finish_err(&inst, StatusCode::BAD_GATEWAY, format!("engine did not answer: {e}")),
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out_headers = HeaderMap::new();
    for (k, v) in upstream.headers() {
        if !HOP.contains(&k.as_str()) || k.as_str() == "content-length" {
            if let (Ok(n), Ok(val)) = (HeaderName::from_bytes(k.as_str().as_bytes()), HeaderValue::from_bytes(v.as_bytes())) {
                out_headers.insert(n, val);
            }
        }
    }
    let is_sse = upstream.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("event-stream"));
    let measure = path.contains("completions");
    let mut stream = upstream.bytes_stream();
    // Recorded when the body is dropped: hyper stops polling once Content-Length is satisfied, so
    // the end of the stream is not a reliable place to finish.
    let mut fin = Finisher {
        rt: s.rt.clone(),
        inst: inst.clone(),
        ctx: Some(ctx),
        tally: Some(Tally::new(is_sse || streaming_req)),
        started,
        status: status.as_u16(),
        measure,
    };
    let out = async_stream::stream! {
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(b) => {
                    if measure { if let Some(t) = fin.tally.as_mut() { t.feed(&b); } }
                    yield Ok::<bytes::Bytes, std::io::Error>(b);
                }
                Err(e) => {
                    yield Err(std::io::Error::other(e));
                    break;
                }
            }
        }
        drop(fin);
    };
    let mut resp = Response::new(Body::from_stream(out));
    *resp.status_mut() = status;
    *resp.headers_mut() = out_headers;
    resp
}

struct Finisher {
    rt: Arc<Runtime>,
    inst: Arc<Instance>,
    ctx: Option<RequestCtx>,
    tally: Option<Tally>,
    started: Instant,
    status: u16,
    measure: bool,
}

impl Drop for Finisher {
    fn drop(&mut self) {
        let (Some(t), Some(ctx)) = (self.tally.take(), self.ctx.take()) else { return };
        let summary = t.finish(self.started, self.status);
        let (rt, inst, measure) = (self.rt.clone(), self.inst.clone(), self.measure);
        tokio::spawn(async move { record(&rt, &inst, &ctx, &summary, measure).await });
    }
}

fn finish_err(inst: &Arc<Instance>, code: StatusCode, msg: String) -> Response {
    let mut m = inst.metrics.lock();
    m.in_flight -= 1;
    m.requests += 1;
    m.errors += 1;
    drop(m);
    err(code, msg)
}

async fn record(rt: &Arc<Runtime>, inst: &Arc<Instance>, ctx: &RequestCtx, s: &ResponseSummary, measured: bool) {
    {
        let mut m = inst.metrics.lock();
        m.in_flight -= 1;
        m.requests += 1;
        m.last_latency_ms = s.latency_ms;
        m.last_request_ms = now_ms();
        if measured {
            m.prompt_tokens += s.prompt_tokens;
            m.completion_tokens += s.completion_tokens;
            if s.decode_tps > 0.0 {
                m.last_decode_tps = s.decode_tps;
                m.avg_decode_tps = if m.avg_decode_tps == 0.0 { s.decode_tps } else { m.avg_decode_tps * 0.8 + s.decode_tps * 0.2 };
            }
            if s.prefill_tps > 0.0 {
                m.last_prefill_tps = s.prefill_tps;
            }
        }
    }
    rt.events.emit("request.completed", json!({"id": inst.id, "path": ctx.path, "run": ctx.run_id, "summary": s}));
    if rt.cfg().symbiosis.enabled && ctx.run_id.is_some() {
        let chain = rt.middleware.read().clone();
        chain.after(ctx, s).await;
    }
}

/// Reads usage and timings out of a response as it streams past.
struct Tally {
    sse: bool,
    buf: Vec<u8>,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    chunks: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    engine_decode_tps: f64,
    engine_prefill_tps: f64,
}

impl Tally {
    fn new(sse: bool) -> Self {
        Self {
            sse,
            buf: Vec::new(),
            first_token: None,
            last_token: None,
            chunks: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            engine_decode_tps: 0.0,
            engine_prefill_tps: 0.0,
        }
    }

    fn feed(&mut self, b: &[u8]) {
        if self.sse {
            self.buf.extend_from_slice(b);
            while let Some(pos) = self.buf.iter().position(|c| *c == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                if let Some(data) = line.trim().strip_prefix("data:") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                        self.event(&v);
                    }
                }
            }
        } else if self.buf.len() < 32 * 1024 * 1024 {
            self.buf.extend_from_slice(b);
        }
    }

    fn event(&mut self, v: &serde_json::Value) {
        let delta = &v["choices"][0]["delta"];
        let has_text = ["content", "reasoning_content", "reasoning"].iter().any(|k| delta[*k].as_str().is_some_and(|s| !s.is_empty()))
            || v["choices"][0]["text"].as_str().is_some_and(|s| !s.is_empty())
            || delta["tool_calls"].is_array();
        if has_text {
            let now = Instant::now();
            self.first_token.get_or_insert(now);
            self.last_token = Some(now);
            self.chunks += 1;
        }
        self.usage(v);
    }

    fn usage(&mut self, v: &serde_json::Value) {
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(self.prompt_tokens);
            self.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(self.completion_tokens);
        }
        if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
            self.engine_decode_tps = t["predicted_per_second"].as_f64().unwrap_or(0.0);
            self.engine_prefill_tps = t["prompt_per_second"].as_f64().unwrap_or(0.0);
        }
    }

    fn finish(mut self, started: Instant, status: u16) -> ResponseSummary {
        let end = Instant::now();
        if !self.sse && !self.buf.is_empty() {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&self.buf) {
                self.usage(&v);
            }
        }
        let latency = end.duration_since(started).as_secs_f64();
        let completion = if self.completion_tokens > 0 { self.completion_tokens } else { self.chunks };
        let (decode, prefill) = if self.engine_decode_tps > 0.0 {
            (self.engine_decode_tps, self.engine_prefill_tps)
        } else if let (Some(f), Some(l)) = (self.first_token, self.last_token) {
            let gen = l.duration_since(f).as_secs_f64();
            let ttft = f.duration_since(started).as_secs_f64();
            (
                if gen > 0.0 && completion > 1 { (completion - 1) as f64 / gen } else { 0.0 },
                if ttft > 0.0 && self.prompt_tokens > 0 { self.prompt_tokens as f64 / ttft } else { 0.0 },
            )
        } else if latency > 0.0 && completion > 0 {
            (completion as f64 / latency, 0.0)
        } else {
            (0.0, 0.0)
        };
        ResponseSummary {
            status,
            latency_ms: latency * 1000.0,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: completion,
            decode_tps: decode,
            prefill_tps: prefill,
        }
    }
}
