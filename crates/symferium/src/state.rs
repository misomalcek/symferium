//! Persisted runtime state: the model set to restore, learned estimate corrections, bench results.

use anyhow::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct LoadRequest {
    /// Library id or absolute path.
    pub model: String,
    /// Name the model is served under (also the instance id). Default: the library id.
    pub alias: Option<String>,
    /// Engine profile. Default: the class default for the model's format, or the bench winner.
    pub engine: Option<String>,
    /// Public port. Default: first free port in the model range.
    pub port: Option<u16>,
    pub ctx: Option<u32>,
    /// "auto" (default: the projector found next to the model), "none", or a path.
    pub mmproj: Option<String>,
    /// Overrides for any class setting / template placeholder.
    pub params: BTreeMap<String, String>,
    /// Keep in the saved model set (restored at login). Default true.
    pub persist: Option<bool>,
    /// Wait until the model answers before returning.
    pub wait: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BenchResult {
    pub model: String,
    pub family: String,
    pub engine: String,
    pub format: String,
    pub quant: Option<String>,
    pub ts_ms: u64,
    pub runs: u32,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub ttft_ms: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct StateData {
    pub model_set: Vec<LoadRequest>,
    /// measured / estimated memory, per adapter (EMA).
    pub corrections: BTreeMap<String, f64>,
    pub bench: Vec<BenchResult>,
}

pub struct StateStore {
    path: Option<PathBuf>,
    pub data: Mutex<StateData>,
}

impl StateStore {
    pub fn load(path: PathBuf) -> Self {
        let data = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        Self { path: Some(path), data: Mutex::new(data) }
    }
    pub fn in_memory() -> Self {
        Self { path: None, data: Mutex::new(StateData::default()) }
    }
    pub fn save(&self) -> Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let text = serde_json::to_string_pretty(&*self.data.lock())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }
    pub fn update<R>(&self, f: impl FnOnce(&mut StateData) -> R) -> R {
        let r = f(&mut self.data.lock());
        if let Err(e) = self.save() {
            tracing::warn!("saving state: {e:#}");
        }
        r
    }
    pub fn correction(&self, adapter: &str) -> f64 {
        self.data.lock().corrections.get(adapter).copied().unwrap_or(1.0)
    }
    /// Fold a measured/estimated ratio into the adapter's correction (EMA, clamped).
    pub fn learn(&self, adapter: &str, ratio: f64) -> f64 {
        let ratio = ratio.clamp(0.5, 2.0);
        self.update(|d| {
            let c = d.corrections.entry(adapter.to_string()).or_insert(1.0);
            *c = (*c * 0.7 + ratio * 0.3).clamp(0.5, 2.0);
            *c
        })
    }
    /// Best engine for a model family by measured decode tok/s.
    pub fn preferred_engine(&self, family: &str) -> Option<(String, String)> {
        let d = self.data.lock();
        d.bench
            .iter()
            .filter(|b| b.family == family && b.decode_tps > 0.0)
            .max_by(|a, b| a.decode_tps.total_cmp(&b.decode_tps))
            .map(|b| (b.engine.clone(), b.model.clone()))
    }
}
