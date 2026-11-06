//! A fake engine that speaks llama-server's HTTP surface. Used by tests and CI smoke runs so the
//! supervisor, proxy, restart and metrics paths are exercised without model weights or Metal.
//!
//! Understands `--port`/`--host` and ignores every other flag. Env knobs:
//!   MOCK_STARTUP_MS   delay before /health turns 200
//!   MOCK_EXIT_AFTER   exit(1) after this many completion requests
//!   MOCK_TOKENS       tokens per completion (default 8)

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct S {
    started: Instant,
    startup: Duration,
    exit_after: u64,
    tokens: u64,
    count: AtomicU64,
    last_body: parking_lot::Mutex<Value>,
    args: Vec<String>,
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get_arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let port: u16 = get_arg("--port").and_then(|p| p.parse().ok()).unwrap_or(18080);
    let host = get_arg("--host").unwrap_or("127.0.0.1".into());
    let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let s = Arc::new(S {
        started: Instant::now(),
        startup: Duration::from_millis(env("MOCK_STARTUP_MS", 0)),
        exit_after: env("MOCK_EXIT_AFTER", 0),
        tokens: env("MOCK_TOKENS", 8),
        count: AtomicU64::new(0),
        last_body: parking_lot::Mutex::new(Value::Null),
        args: args.clone(),
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/v1/audio/transcriptions", post(|| async { Json(json!({"text": "mock transcription"})) }))
        .route("/debug/last-body", get(|State(s): State<Arc<S>>| async move { Json(s.last_body.lock().clone()) }))
        .route("/debug/args", get(|State(s): State<Arc<S>>| async move { Json(json!(s.args)) }))
        .with_state(s);
    let l = tokio::net::TcpListener::bind((host.as_str(), port)).await.expect("bind");
    eprintln!("mock-engine listening on {host}:{port}");
    axum::serve(l, app).await.unwrap();
}

async fn health(State(s): State<Arc<S>>) -> impl IntoResponse {
    if s.started.elapsed() < s.startup {
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "loading model"})));
    }
    (axum::http::StatusCode::OK, Json(json!({"status": "ok"})))
}

async fn models() -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": "mock", "object": "model"}]}))
}

fn bump(s: &S) {
    let n = s.count.fetch_add(1, Ordering::SeqCst) + 1;
    if s.exit_after > 0 && n >= s.exit_after {
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            std::process::exit(1);
        });
    }
}

async fn chat(State(s): State<Arc<S>>, Json(body): Json<Value>) -> axum::response::Response {
    *s.last_body.lock() = body.clone();
    bump(&s);
    let model = body["model"].as_str().unwrap_or("mock").to_string();
    let n = body["max_tokens"].as_u64().unwrap_or(s.tokens).min(s.tokens);
    if body["stream"].as_bool().unwrap_or(false) {
        let stream = async_stream::stream! {
            for i in 0..n {
                tokio::time::sleep(Duration::from_millis(5)).await;
                let ev = json!({"id": "c", "object": "chat.completion.chunk", "model": model,
                    "choices": [{"index": 0, "delta": {"content": format!("t{i} ")}, "finish_reason": null}]});
                yield Ok::<_, std::convert::Infallible>(Event::default().data(ev.to_string()));
            }
            let fin = json!({"id": "c", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": n, "total_tokens": 12 + n},
                "timings": {"prompt_per_second": 100.0, "predicted_per_second": 42.0}});
            yield Ok(Event::default().data(fin.to_string()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Sse::new(stream).into_response();
    }
    let text: String = (0..n).map(|i| format!("t{i} ")).collect();
    Json(json!({"id": "c", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": n, "total_tokens": 12 + n}}))
    .into_response()
}

async fn embeddings(State(s): State<Arc<S>>, Json(body): Json<Value>) -> Json<Value> {
    *s.last_body.lock() = body.clone();
    let inputs = match &body["input"] {
        Value::Array(a) => a.len(),
        _ => 1,
    };
    Json(
        json!({"object": "list", "data": (0..inputs).map(|i| json!({"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]})).collect::<Vec<_>>(),
        "usage": {"prompt_tokens": 3, "total_tokens": 3}}),
    )
}

async fn rerank(Json(body): Json<Value>) -> Json<Value> {
    let docs = body["documents"].as_array().map(|a| a.len()).unwrap_or(0);
    Json(json!({"results": (0..docs).map(|i| json!({"index": i, "relevance_score": 1.0 / (i as f64 + 1.0)})).collect::<Vec<_>>()}))
}
