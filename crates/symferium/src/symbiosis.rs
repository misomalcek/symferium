//! Seams for the symbiosis layer (v2, see docs/SYMBIOSIS.md).
//!
//! v1 ships the middleware chain (empty unless something registers) and the event bus. Ghost,
//! Whisper, Nexus and Heron will plug in here without touching engines or clients.

use futures::future::BoxFuture;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub struct RequestCtx {
    pub instance: String,
    pub model: String,
    pub path: String,
    /// From the X-Symferium-Run header: ties requests of one agentic run together.
    pub run_id: Option<String>,
    /// The loaded context window.
    pub ctx: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ResponseSummary {
    pub status: u16,
    pub latency_ms: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub decode_tps: f64,
    pub prefill_tps: f64,
}

pub trait Middleware: Send + Sync {
    fn name(&self) -> &str;
    /// May rewrite the JSON request body (Whisper compacts and injects context here).
    fn before_request<'a>(&'a self, ctx: &'a RequestCtx, body: &'a mut serde_json::Value) -> BoxFuture<'a, anyhow::Result<()>>;
    /// Sees the outcome (Ghost records here). Must not block the response.
    fn after_response<'a>(&'a self, ctx: &'a RequestCtx, summary: &'a ResponseSummary) -> BoxFuture<'a, ()>;
}

#[derive(Default, Clone)]
pub struct Chain {
    items: Vec<Arc<dyn Middleware>>,
}

impl Chain {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, m: Arc<dyn Middleware>) {
        self.items.push(m);
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn names(&self) -> Vec<String> {
        self.items.iter().map(|m| m.name().to_string()).collect()
    }
    /// A failing middleware is logged and skipped: symbiosis degrades, it never blocks inference.
    pub async fn before(&self, ctx: &RequestCtx, body: &mut serde_json::Value) {
        for m in &self.items {
            if let Err(e) = m.before_request(ctx, body).await {
                tracing::warn!(middleware = m.name(), "before_request failed: {e:#}");
            }
        }
    }
    pub async fn after(&self, ctx: &RequestCtx, s: &ResponseSummary) {
        for m in &self.items {
            m.after_response(ctx, s).await;
        }
    }
}
