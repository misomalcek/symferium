//! Presets: named model sets loaded in one call (e.g. the M5 Factorium parity set).
//! Built-in presets ship in the binary; more can be dropped into <state_dir>/presets/*.json.

use crate::runtime::Runtime;
use crate::state::LoadRequest;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    #[serde(default)]
    pub description: String,
    pub models: Vec<LoadRequest>,
}

const BUILTIN: &[(&str, &str)] = &[("factorium-parity", include_str!("../../../presets/factorium-parity.json"))];

pub fn all(rt: &Runtime) -> BTreeMap<String, Preset> {
    let mut out = BTreeMap::new();
    for (name, text) in BUILTIN {
        if let Ok(p) = serde_json::from_str(text) {
            out.insert(name.to_string(), p);
        }
    }
    let dir = rt.cfg().state_dir().join("presets");
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "json") {
                match std::fs::read_to_string(&p).map_err(anyhow::Error::from).and_then(|t| Ok(serde_json::from_str::<Preset>(&t)?)) {
                    Ok(preset) => {
                        out.insert(p.file_stem().unwrap().to_string_lossy().into_owned(), preset);
                    }
                    Err(e) => tracing::warn!("preset {}: {e:#}", p.display()),
                }
            }
        }
    }
    out
}

/// Load every model of a preset, one at a time, each waiting until it answers.
pub async fn load(rt: &Arc<Runtime>, name: &str) -> Result<Vec<serde_json::Value>> {
    let preset = all(rt).remove(name).ok_or_else(|| anyhow!("no preset named {name}"))?;
    let mut results = vec![];
    for mut req in preset.models {
        req.wait = true;
        let label = req.alias.clone().unwrap_or(req.model.clone());
        results.push(match rt.load(req).await {
            Ok(v) => serde_json::json!({"id": label, "ok": true, "url": v.url, "estimate_bytes": v.estimate.total}),
            Err(e) => serde_json::json!({"id": label, "ok": false, "error": e.to_string()}),
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    #[test]
    fn builtin_presets_parse() {
        for (name, text) in super::BUILTIN {
            let p: super::Preset = serde_json::from_str(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(!p.models.is_empty());
        }
    }
}
