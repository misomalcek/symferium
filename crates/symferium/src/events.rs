//! Event bus: model lifecycle, requests, downloads, settings. Served as SSE on /api/events and kept
//! in a short history so a page that opens late still sees what happened. The symbiosis layer
//! (v2) subscribes here.

use parking_lot::Mutex;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize)]
pub struct Event {
    pub seq: u64,
    pub ts_ms: u64,
    pub kind: String,
    pub data: serde_json::Value,
}

pub struct EventBus {
    tx: broadcast::Sender<Event>,
    seq: AtomicU64,
    history: Mutex<VecDeque<Event>>,
    keep: usize,
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

impl EventBus {
    pub fn new(keep: usize) -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self { tx, seq: AtomicU64::new(0), history: Mutex::new(VecDeque::new()), keep }
    }

    pub fn emit(&self, kind: &str, data: serde_json::Value) {
        let ev = Event { seq: self.seq.fetch_add(1, Ordering::Relaxed) + 1, ts_ms: now_ms(), kind: kind.into(), data };
        tracing::debug!(kind, "event");
        {
            let mut h = self.history.lock();
            h.push_back(ev.clone());
            while h.len() > self.keep {
                h.pop_front();
            }
        }
        let _ = self.tx.send(ev);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }

    pub fn history(&self) -> Vec<Event> {
        self.history.lock().iter().cloned().collect()
    }
}
