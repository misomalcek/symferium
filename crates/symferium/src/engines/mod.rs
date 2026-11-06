//! Engine profiles → concrete launch commands.
//!
//! A profile is a binary plus argument templates. `{placeholders}` are filled from the model, the
//! port and the merged class settings, so any engine flag is visible and editable in the config.

use crate::config::{expand, ClassDefaults, Config, EngineProfile};
use crate::library::ModelEntry;
use crate::state::LoadRequest;
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct LaunchSpec {
    pub engine: String,
    pub adapter: String,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub health_path: String,
    pub rewrite_model_field: bool,
    /// What the engine itself calls the model (used for the "model" rewrite).
    pub served_model: String,
}

impl LaunchSpec {
    pub fn command_line(&self) -> String {
        let mut s = shell_quote(&self.program.to_string_lossy());
        for a in &self.args {
            s.push(' ');
            s.push_str(&shell_quote(a));
        }
        s
    }
}

fn shell_quote(s: &str) -> String {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:=,{}".contains(c)) && !s.is_empty() {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Merged settings for one load: class defaults, then request params.
pub fn params_for(class: &ClassDefaults, req: &LoadRequest) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    if let Ok(serde_json::Value::Object(o)) = serde_json::to_value(class) {
        for (k, v) in o {
            p.insert(k, v.as_str().map(String::from).unwrap_or_else(|| v.to_string()));
        }
    }
    if let Some(c) = req.ctx {
        p.insert("ctx".into(), c.to_string());
    }
    for (k, v) in &req.params {
        p.insert(k.clone(), v.clone());
    }
    p
}

/// Which profile serves this model.
pub fn choose_engine(cfg: &Config, entry: &ModelEntry, req: &LoadRequest, class: &ClassDefaults) -> Result<String> {
    let usable = |name: &str| -> Option<&EngineProfile> {
        cfg.engines.get(name).filter(|p| p.enabled && (p.formats.is_empty() || p.formats.contains(&entry.format)))
    };
    if let Some(e) = &req.engine {
        let p = cfg.engines.get(e).ok_or_else(|| anyhow!("unknown engine profile {e}"))?;
        if !p.enabled {
            bail!("engine profile {e} is disabled (engines.{e}.enabled)");
        }
        if !p.formats.is_empty() && !p.formats.contains(&entry.format) {
            bail!("engine profile {e} cannot load {} files", entry.format);
        }
        return Ok(e.clone());
    }
    let preferred = if entry.format == "mlx" {
        if entry.flags.vision { "mlx-vlm".to_string() } else { class.engine_mlx.clone() }
    } else {
        class.engine.clone()
    };
    if usable(&preferred).is_some() {
        return Ok(preferred);
    }
    cfg.engines
        .iter()
        .find(|(_, p)| p.enabled && p.formats.contains(&entry.format) && (p.kinds.is_empty() || p.kinds.contains(&entry.kind)))
        .map(|(k, _)| k.clone())
        .ok_or_else(|| anyhow!("no enabled engine profile loads {} {} models", entry.format, entry.kind))
}

/// Find an executable: absolute/relative paths as given, bare names in PATH and the Homebrew
/// prefixes (launchd starts agents with a minimal PATH).
pub fn resolve_binary(bin: &str) -> Result<PathBuf> {
    let p = expand(bin);
    if bin.contains('/') {
        if p.exists() {
            return Ok(p);
        }
        bail!("engine binary {} not found", p.display());
    }
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH").map(|v| std::env::split_paths(&v).collect()).unwrap_or_default();
    for extra in ["/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "~/.cargo/bin"] {
        dirs.push(expand(extra));
    }
    for d in dirs {
        let c = d.join(bin);
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!("engine binary `{bin}` not found in PATH, /opt/homebrew/bin or /usr/local/bin (install it or set the profile's binary path)")
}

fn fill(t: &str, vars: &BTreeMap<String, String>) -> Result<String> {
    let mut out = String::new();
    let mut rest = t;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            out.push_str(&rest[start..]);
            return Ok(out);
        };
        let key = &after[..end];
        if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !key.is_empty() {
            let v = vars.get(key).ok_or_else(|| anyhow!("unknown placeholder {{{key}}} in engine arguments"))?;
            out.push_str(v);
        } else {
            out.push('{');
            out.push_str(key);
            out.push('}');
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    cfg: &Config,
    engine: &str,
    entry: &ModelEntry,
    params: &BTreeMap<String, String>,
    mmproj: Option<&Path>,
    host: &str,
    port: u16,
    alias: &str,
) -> Result<LaunchSpec> {
    let p = cfg.engines.get(engine).ok_or_else(|| anyhow!("unknown engine profile {engine}"))?;
    let mut vars = params.clone();
    vars.insert("model".into(), entry.path.to_string_lossy().into());
    vars.insert("mmproj".into(), mmproj.map(|m| m.to_string_lossy().into_owned()).unwrap_or_default());
    vars.insert("host".into(), host.into());
    vars.insert("port".into(), port.to_string());
    vars.insert("alias".into(), alias.into());
    vars.insert("python_env".into(), expand(&cfg.paths.python_env).to_string_lossy().into());
    vars.insert("sidecar_dir".into(), expand(&cfg.paths.sidecar_dir).to_string_lossy().into());
    vars.insert("models_dir".into(), cfg.models_dir().to_string_lossy().into());
    let mut templ: Vec<&String> = p.args.iter().collect();
    if mmproj.is_some() {
        templ.extend(&p.args_vision);
    }
    match entry.kind.as_str() {
        "embed" => templ.extend(&p.args_embed),
        "rerank" => templ.extend(&p.args_rerank),
        _ => {}
    }
    templ.extend(&p.extra_args);
    let args = templ.iter().map(|a| fill(a, &vars)).collect::<Result<Vec<_>>>()?;
    let program = resolve_binary(&fill(&p.binary, &vars)?)?;
    let env = p.env.iter().map(|(k, v)| Ok((k.clone(), fill(v, &vars)?))).collect::<Result<_>>()?;
    Ok(LaunchSpec {
        engine: engine.into(),
        adapter: p.adapter.clone(),
        program,
        args,
        env,
        health_path: p.health_path.clone(),
        rewrite_model_field: p.rewrite_model_field,
        served_model: if p.adapter == "llamacpp" { alias.to_string() } else { entry.path.to_string_lossy().into() },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kind: &str, format: &str) -> ModelEntry {
        ModelEntry {
            id: "m".into(),
            name: "m".into(),
            path: "/models/m.gguf".into(),
            format: format.into(),
            kind: kind.into(),
            class: "standard".into(),
            size_bytes: 0,
            params: None,
            arch: "llama".into(),
            quant: None,
            ctx_train: None,
            flags: Default::default(),
            flag_sources: Default::default(),
            mmproj: None,
            mmproj_bytes: 0,
            read_only: false,
            geometry: None,
            family: "m".into(),
        }
    }

    #[test]
    fn llamacpp_command_for_vision_chat() {
        let mut cfg = Config::default();
        cfg.engines.get_mut("llamacpp").unwrap().binary = "/bin/sh".into();
        let req = LoadRequest { ctx: Some(65536), ..Default::default() };
        let params = params_for(&cfg.class("standard"), &req);
        let spec = build(&cfg, "llamacpp", &entry("chat", "gguf"), &params, Some(Path::new("/models/mmproj.gguf")), "127.0.0.1", 17701, "gemma").unwrap();
        let line = spec.command_line();
        assert!(line.contains("-c 65536"), "{line}");
        assert!(line.contains("--mmproj /models/mmproj.gguf"));
        assert!(line.contains("-ctk q8_0"));
        assert!(line.contains("--port 17701"));
        assert!(!line.contains("--embeddings"));
    }

    #[test]
    fn embed_and_unknown_placeholder() {
        let mut cfg = Config::default();
        cfg.engines.get_mut("llamacpp").unwrap().binary = "/bin/sh".into();
        let params = params_for(&cfg.class("embed"), &LoadRequest::default());
        let spec = build(&cfg, "llamacpp", &entry("embed", "gguf"), &params, None, "127.0.0.1", 1, "e").unwrap();
        assert!(spec.command_line().contains("--embeddings --pooling cls"));
        cfg.engines.get_mut("llamacpp").unwrap().extra_args = vec!["{nope}".into()];
        assert!(build(&cfg, "llamacpp", &entry("embed", "gguf"), &params, None, "h", 1, "e").is_err());
    }

    #[test]
    fn engine_choice() {
        let cfg = Config::default();
        let req = LoadRequest::default();
        assert_eq!(choose_engine(&cfg, &entry("chat", "gguf"), &req, &cfg.class("standard")).unwrap(), "llamacpp");
        assert_eq!(choose_engine(&cfg, &entry("chat", "mlx"), &req, &cfg.class("standard")).unwrap(), "mlx");
        assert_eq!(choose_engine(&cfg, &entry("stt", "whisper"), &req, &cfg.class("stt")).unwrap(), "whispercpp");
        let bad = LoadRequest { engine: Some("mlx".into()), ..Default::default() };
        assert!(choose_engine(&cfg, &entry("chat", "gguf"), &bad, &cfg.class("standard")).is_err());
    }
}
