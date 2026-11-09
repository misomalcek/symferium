//! Agent tooling: a JSON tool manifest and one HTTP call per tool. Agents may change only settings
//! marked agent_writable, and every load goes through the governor.

use crate::catalog::{self, FileRef, SearchQuery};
use crate::config::{self, Actor};
use crate::runtime::{LoadError, Runtime};
use crate::state::LoadRequest;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;

pub fn manifest() -> Value {
    json!({
        "name": "symferium",
        "description": "Local model runtime on this Mac. Load and unload models (each gets its own OpenAI-compatible port), search and download models, read metrics, change agent-writable settings. Every load is checked against the memory budget.",
        "call": "POST /api/tools/{name} with the arguments as a JSON object",
        "tools": [
            {"name": "list_models", "description": "Models on disk and the ones loaded now, with their ports, engines, state and tok/s.",
             "parameters": {"type": "object", "properties": {}}},
            {"name": "machine", "description": "Memory, pressure and the current budget (headroom for the next load).",
             "parameters": {"type": "object", "properties": {}}},
            {"name": "metrics", "description": "Per-model request counts, tokens and tok/s.",
             "parameters": {"type": "object", "properties": {}}},
            {"name": "estimate", "description": "Dry run: memory estimate for loading a model and whether the governor would allow it.",
             "parameters": {"$ref": "#/definitions/load"}},
            {"name": "load", "description": "Load a model on its own port. Refused with numbers if it does not fit.",
             "parameters": {"$ref": "#/definitions/load"}},
            {"name": "unload", "description": "Unload a loaded model.",
             "parameters": {"type": "object", "required": ["id"], "properties": {
                "id": {"type": "string", "description": "Instance id (alias) from list_models"},
                "forget": {"type": "boolean", "description": "Also drop it from the set restored at login (default true)"}}}},
            {"name": "search_catalog", "description": "Search Hugging Face for models, with class, format and capability filters.",
             "parameters": {"type": "object", "properties": {
                "q": {"type": "string"}, "author": {"type": "string"},
                "format": {"type": "string", "enum": ["gguf", "mlx"]},
                "class": {"type": "string", "enum": ["small", "standard", "large", "embed", "rerank", "decision", "stt", "tts"]},
                "vision": {"type": "boolean"}, "tools": {"type": "boolean"}, "thinking": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200}}}},
            {"name": "catalog_detail", "description": "Files, quants (with fit against the budget), projector files and flags of one repo.",
             "parameters": {"type": "object", "required": ["repo"], "properties": {"repo": {"type": "string"}}}},
            {"name": "download", "description": "Download a quant (and its projector) from a repo into the models folder.",
             "parameters": {"type": "object", "required": ["repo"], "properties": {
                "repo": {"type": "string"}, "quant": {"type": "string", "description": "e.g. Q4_K_M; default: the recommended quant"},
                "include_mmproj": {"type": "boolean", "default": true}}}},
            {"name": "get_settings", "description": "All settings with their schema (agent_writable tells what you may change).",
             "parameters": {"type": "object", "properties": {"agent_writable_only": {"type": "boolean"}}}},
            {"name": "set_setting", "description": "Change one agent_writable setting.",
             "parameters": {"type": "object", "required": ["key", "value"], "properties": {"key": {"type": "string"}, "value": {}}}},
            {"name": "load_preset", "description": "Load a named model set (GET /api/presets lists them), one model at a time.",
             "parameters": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
            {"name": "bench", "description": "Measure prefill and decode tok/s of a loaded chat model.",
             "parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}, "max_tokens": {"type": "integer"}, "runs": {"type": "integer"}}}}
        ],
        "definitions": {
            "load": {"type": "object", "required": ["model"], "properties": {
                "model": {"type": "string", "description": "Library id from list_models"},
                "alias": {"type": "string", "description": "Name to serve it under"},
                "engine": {"type": "string", "description": "Engine profile (default: chosen for the model)"},
                "port": {"type": "integer"}, "ctx": {"type": "integer", "description": "Context length"},
                "mmproj": {"type": "string", "description": "auto, none, or a path"},
                "params": {"type": "object", "additionalProperties": {"type": "string"}},
                "persist": {"type": "boolean", "description": "Restore at login (default true)"},
                "wait": {"type": "boolean", "description": "Return only when the model answers"}}}
        }
    })
}

fn load_err(e: LoadError) -> anyhow::Error {
    match e {
        LoadError::Refused(d) => anyhow!("{}", json!({"refused": true, "reasons": d.reasons, "estimate": d.estimate, "budget": d.budget})),
        LoadError::Invalid(e) => e,
    }
}

pub async fn call(rt: &Arc<Runtime>, name: &str, args: Value) -> Result<Value> {
    let args = if args.is_null() { json!({}) } else { args };
    Ok(match name {
        "list_models" => json!({"loaded": rt.list(), "local": rt.library.list().into_iter().map(|e| json!({
            "id": e.id, "name": e.name, "kind": e.kind, "class": e.class, "format": e.format, "quant": e.quant,
            "size_bytes": e.size_bytes, "flags": e.flags, "ctx_train": e.ctx_train, "vision_projector": e.mmproj.is_some()})).collect::<Vec<_>>()}),
        "machine" => json!({"machine": rt.probe.snapshot(), "budget": rt.budget()}),
        "metrics" => json!(rt
            .list()
            .into_iter()
            .map(|i| json!({"id": i.id, "state": i.status.state, "port": i.port, "metrics": i.metrics}))
            .collect::<Vec<_>>()),
        "estimate" => {
            let req: LoadRequest = serde_json::from_value(args)?;
            let p = rt.plan(&req)?;
            json!({"engine": p.engine, "class": p.class, "ctx": p.ctx, "allowed": p.decision.allowed, "reasons": p.decision.reasons, "estimate": p.estimate, "budget": p.decision.budget})
        }
        "load" => {
            let req: LoadRequest = serde_json::from_value(args)?;
            json!(rt.load(req).await.map_err(load_err)?)
        }
        "unload" => {
            let id = args["id"].as_str().ok_or_else(|| anyhow!("id is required"))?;
            rt.unload(id, args["forget"].as_bool().unwrap_or(true)).await?;
            json!({"unloaded": id})
        }
        "search_catalog" => {
            let q: SearchQuery = serde_json::from_value(args)?;
            json!(catalog::search(rt, &q).await?)
        }
        "catalog_detail" => {
            let repo = args["repo"].as_str().ok_or_else(|| anyhow!("repo is required"))?;
            json!(catalog::detail(rt, repo).await?)
        }
        "download" => {
            let repo = args["repo"].as_str().ok_or_else(|| anyhow!("repo is required"))?;
            let m = catalog::detail(rt, repo).await?;
            let q = match args["quant"].as_str() {
                Some(q) => m.quants.iter().find(|x| x.quant.eq_ignore_ascii_case(q)),
                None => m.quants.iter().find(|x| x.recommended).or(m.quants.first()),
            }
            .ok_or_else(|| anyhow!("no such quant in {repo}"))?;
            if q.fit == "too_big" {
                return Err(anyhow!("{} {} does not fit this machine's ceiling", repo, q.quant));
            }
            let mut files: Vec<FileRef> = q.files.clone();
            if args["include_mmproj"].as_bool().unwrap_or(true) {
                if let Some(p) = m.mmproj.first() {
                    files.push(p.clone());
                }
            }
            let id = crate::download::start(rt, repo, files)?;
            json!({"download_id": id, "quant": q.quant})
        }
        "get_settings" => {
            let cfg = rt.cfg();
            let mut specs = config::schema(&cfg);
            if args["agent_writable_only"].as_bool().unwrap_or(false) {
                specs.retain(|s| s.agent_writable);
            }
            let specs: Vec<Value> = specs
                .into_iter()
                .filter(|s| !s.secret)
                .map(|s| {
                    let mut v = serde_json::to_value(&s).unwrap_or_default();
                    v["value"] = config::get_value(&cfg, &s.key).unwrap_or(Value::Null);
                    v
                })
                .collect();
            json!(specs)
        }
        "set_setting" => {
            let key = args["key"].as_str().ok_or_else(|| anyhow!("key is required"))?;
            let new = config::apply_change(&rt.cfg(), key, args["value"].clone(), Actor::Agent)?;
            rt.set_config(new)?;
            json!({"ok": true, "key": key, "value": config::get_value(&rt.cfg(), key)?})
        }
        "load_preset" => {
            let name = args["name"].as_str().ok_or_else(|| anyhow!("name is required"))?;
            json!(crate::presets::load(rt, name).await?)
        }
        "bench" => {
            let id = args["id"].as_str().ok_or_else(|| anyhow!("id is required"))?;
            json!(
                crate::bench::run(rt, id, None, args["max_tokens"].as_u64().map(|v| v as u32), args["runs"].as_u64().map(|v| v as u32))
                    .await?
            )
        }
        other => return Err(anyhow!("unknown tool {other}")),
    })
}
