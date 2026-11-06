//! Local model library: scans the models folders and describes every model it finds.
//! GGUF files (with their mmproj projectors), MLX / safetensors folders, whisper.cpp ggml files,
//! and ONNX TTS voices.

use crate::config::{expand, Config};
use crate::estimate::Geometry;
use crate::gguf;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Flags {
    pub vision: bool,
    pub thinking: bool,
    pub tools: bool,
    pub agentic: bool,
    pub coding: bool,
    pub audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    /// gguf, mlx, whisper, onnx
    pub format: String,
    /// chat, embed, rerank, stt, tts
    pub kind: String,
    /// small, standard, large, embed, rerank, decision, stt, tts, oversize
    pub class: String,
    pub size_bytes: u64,
    pub params: Option<u64>,
    pub arch: String,
    pub quant: Option<String>,
    pub ctx_train: Option<u64>,
    pub flags: Flags,
    /// Where each flag came from (template, config, projector file…), so nothing is guessed silently.
    pub flag_sources: BTreeMap<String, String>,
    pub mmproj: Option<PathBuf>,
    pub mmproj_bytes: u64,
    pub read_only: bool,
    pub geometry: Option<Geometry>,
    /// Name with format/quant suffixes removed; groups GGUF and MLX builds of the same model.
    pub family: String,
}

#[derive(Default)]
pub struct Library {
    entries: RwLock<Vec<ModelEntry>>,
    cache: RwLock<HashMap<PathBuf, (u64, SystemTime, gguf::GgufInfo)>>,
}

pub fn quant_from_name(name: &str) -> Option<String> {
    let re = regex::Regex::new(r"(?i)(UD-)?(IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_K(?:_[SMLX]{1,2})?|Q[4-8]_[01]|Q8_K_XL|TQ[12]_0|MXFP4(?:_MOE)?|BF16|F16|F32|[2-8]bit|DWQ)").ok()?;
    let caps: Vec<_> = re.find_iter(name).collect();
    caps.last().map(|m| m.as_str().to_uppercase().replace("BIT", "bit"))
}

pub fn family_of(name: &str) -> String {
    let mut s = name.to_lowercase();
    for cut in ["-gguf", "-mlx", ".gguf"] {
        s = s.replace(cut, "");
    }
    if let Some(q) = quant_from_name(&s) {
        if let Some(pos) = s.to_lowercase().rfind(&q.to_lowercase()) {
            s.truncate(pos);
        }
    }
    s.trim_end_matches(['-', '_', '.']).to_string()
}

pub fn classify(kind: &str, params: Option<u64>, name: &str, cfg: &Config) -> String {
    match kind {
        "embed" | "rerank" | "stt" | "tts" => return kind.to_string(),
        _ => {}
    }
    let lname = name.to_lowercase();
    if lname.contains("decision") || lname.contains("function-call") || lname.contains("-fc-") {
        return "decision".into();
    }
    let Some(p) = params else { return "standard".into() };
    let b = p as f64 / 1e9;
    let c = &cfg.catalog;
    if b < c.small_max_b {
        "small".into()
    } else if b >= c.standard_min_b && b <= c.standard_max_b {
        "standard".into()
    } else if b <= c.large_max_b {
        "large".into()
    } else {
        "oversize".into()
    }
}

pub fn flags_from_template(tpl: &str, f: &mut Flags, src: &mut BTreeMap<String, String>) {
    if tpl.contains("tools") || tpl.contains("tool_call") {
        f.tools = true;
        src.insert("tools".into(), "chat template".into());
    }
    if tpl.contains("enable_thinking") || tpl.contains("<think>") || tpl.contains("reasoning_content") || tpl.contains("thinking") {
        f.thinking = true;
        src.insert("thinking".into(), "chat template".into());
    }
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn dir_size(dir: &Path) -> u64 {
    walk_files(dir, 3).iter().filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum()
}

fn walk_files(dir: &Path, depth: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            if depth > 0 {
                out.extend(walk_files(&p, depth - 1));
            }
        } else {
            out.push(p);
        }
    }
    out
}

impl Library {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        self.entries.read().clone()
    }

    pub fn get(&self, id: &str) -> Option<ModelEntry> {
        self.entries.read().iter().find(|e| e.id == id || e.path.to_string_lossy() == id).cloned()
    }

    fn gguf_info(&self, p: &Path) -> Option<gguf::GgufInfo> {
        let md = p.metadata().ok()?;
        let (len, mtime) = (md.len(), md.modified().ok()?);
        if let Some((l, m, info)) = self.cache.read().get(p) {
            if *l == len && *m == mtime {
                return Some(info.clone());
            }
        }
        let info = gguf::read(p).map_err(|e| tracing::warn!("{e:#}")).ok()?;
        self.cache.write().insert(p.to_path_buf(), (len, mtime, info.clone()));
        Some(info)
    }

    pub fn scan(&self, cfg: &Config) -> Vec<ModelEntry> {
        let mut dirs: Vec<(PathBuf, bool)> = vec![(cfg.models_dir(), false)];
        dirs.extend(cfg.paths.extra_model_dirs.iter().map(|d| (expand(d), true)));
        let mut found = Vec::new();
        for (dir, ro) in dirs {
            if dir.is_dir() {
                found.extend(self.scan_dir(&dir, ro, cfg));
            }
        }
        // Unique ids.
        let mut seen: HashMap<String, usize> = HashMap::new();
        for e in found.iter_mut() {
            let n = seen.entry(e.id.clone()).or_insert(0);
            *n += 1;
            if *n > 1 {
                e.id = format!("{}-{}", e.id, n);
            }
        }
        found.sort_by(|a, b| a.class.cmp(&b.class).then(a.name.cmp(&b.name)));
        *self.entries.write() = found.clone();
        found
    }

    fn scan_dir(&self, root: &Path, read_only: bool, cfg: &Config) -> Vec<ModelEntry> {
        let files = walk_files(root, 4);
        let mut out = Vec::new();
        // GGUF: collect projectors per folder first.
        let mut projectors: HashMap<PathBuf, Vec<(PathBuf, u64)>> = HashMap::new();
        let mut ggufs = Vec::new();
        let shard_re = regex::Regex::new(r"-(\d{5})-of-(\d{5})\.gguf$").unwrap();
        for f in files.iter().filter(|f| f.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))) {
            let name = f.file_name().unwrap().to_string_lossy().to_lowercase();
            // Only the first shard of a split model.
            if let Some(c) = shard_re.captures(&name) {
                if &c[1] != "00001" {
                    continue;
                }
            }
            let Some(info) = self.gguf_info(f) else { continue };
            if info.is_projector() || name.starts_with("mmproj") {
                let dir = f.parent().unwrap().to_path_buf();
                projectors.entry(dir).or_default().push((f.clone(), f.metadata().map(|m| m.len()).unwrap_or(0)));
            } else {
                ggufs.push((f.clone(), info));
            }
        }
        for (f, info) in ggufs {
            let dir = f.parent().unwrap().to_path_buf();
            let name = stem(&f);
            let size = if name.contains("-00001-of-") {
                let prefix = name.split("-00001-of-").next().unwrap_or("").to_string();
                walk_files(&dir, 0)
                    .iter()
                    .filter(|p| stem(p).starts_with(&prefix))
                    .filter_map(|p| p.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            } else {
                f.metadata().map(|m| m.len()).unwrap_or(0)
            };
            let arch = info.arch().to_string();
            let lname = name.to_lowercase();
            let has_pooling = info.arch_val("pooling_type").is_some();
            let geometry = Geometry::from_gguf(&info);
            let encoder = geometry.as_ref().is_some_and(|g| g.encoder_only);
            let kind = if lname.contains("rerank") || info.arch_u64("pooling_type") == Some(4) {
                "rerank"
            } else if encoder || has_pooling || lname.contains("embed") {
                "embed"
            } else {
                "chat"
            };
            let mut flags = Flags::default();
            let mut src = BTreeMap::new();
            if let Some(tpl) = info.str("tokenizer.chat_template") {
                flags_from_template(tpl, &mut flags, &mut src);
            }
            if lname.contains("coder") || lname.contains("code") {
                flags.coding = true;
                src.insert("coding".into(), "model name".into());
            }
            // Pick the best projector in the same folder (prefer f16/bf16, then q8).
            let mut mmproj = None;
            let mut mmproj_bytes = 0;
            if kind == "chat" {
                if let Some(ps) = projectors.get(&dir) {
                    let mut ps = ps.clone();
                    ps.sort_by_key(|(p, _)| {
                        let n = p.to_string_lossy().to_lowercase();
                        if n.contains("f16") && !n.contains("bf16") { 0 } else if n.contains("bf16") { 1 } else if n.contains("q8") { 2 } else { 3 }
                    });
                    if let Some((p, s)) = ps.first() {
                        mmproj = Some(p.clone());
                        mmproj_bytes = *s;
                        flags.vision = true;
                        src.insert("vision".into(), format!("projector {}", p.file_name().unwrap().to_string_lossy()));
                    }
                }
            }
            let params = Some(info.parameter_count).filter(|p| *p > 0).or_else(|| info.get("general.parameter_count").and_then(|v| v.as_u64()));
            let display = info.str("general.name").map(String::from).unwrap_or_else(|| name.clone());
            out.push(ModelEntry {
                id: slug(&name),
                class: classify(kind, params, &name, cfg),
                name: display,
                path: f.clone(),
                format: "gguf".into(),
                kind: kind.into(),
                size_bytes: size,
                params,
                arch,
                quant: quant_from_name(&name),
                ctx_train: geometry.as_ref().map(|g| g.ctx_train).filter(|c| *c > 0),
                flags,
                flag_sources: src,
                mmproj,
                mmproj_bytes,
                read_only,
                geometry,
                family: family_of(&name),
            });
        }
        // MLX / safetensors folders.
        let mut mlx_dirs: Vec<PathBuf> = files
            .iter()
            .filter(|f| f.file_name().is_some_and(|n| n == "config.json"))
            .filter_map(|f| f.parent().map(Path::to_path_buf))
            .filter(|d| walk_files(d, 0).iter().any(|p| p.extension().is_some_and(|x| x == "safetensors")))
            .collect();
        mlx_dirs.sort();
        mlx_dirs.dedup();
        for d in mlx_dirs {
            let Ok(text) = std::fs::read_to_string(d.join("config.json")) else { continue };
            let Ok(cfg_json) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
            let name = d.file_name().unwrap().to_string_lossy().into_owned();
            let lname = name.to_lowercase();
            let mut flags = Flags::default();
            let mut src = BTreeMap::new();
            if cfg_json.get("vision_config").is_some() {
                flags.vision = true;
                src.insert("vision".into(), "config.json vision_config".into());
            }
            for tf in ["tokenizer_config.json", "chat_template.jinja", "chat_template.json"] {
                if let Ok(t) = std::fs::read_to_string(d.join(tf)) {
                    flags_from_template(&t, &mut flags, &mut src);
                }
            }
            let kind = if lname.contains("rerank") {
                "rerank"
            } else if lname.contains("embed") || lname.contains("minilm") || lname.contains("bge") {
                "embed"
            } else {
                "chat"
            };
            let size = dir_size(&d);
            let geometry = Geometry::from_hf_config(&cfg_json);
            let bits = cfg_json.pointer("/quantization/bits").and_then(|v| v.as_u64());
            // Parameter estimate from size and bits when no count is given.
            let params = bits.map(|b| (size as f64 * 8.0 / (b as f64 + 0.5)) as u64).or(Some(size / 2));
            out.push(ModelEntry {
                id: slug(&name),
                class: classify(kind, params, &name, cfg),
                name: name.clone(),
                path: d.clone(),
                format: "mlx".into(),
                kind: kind.into(),
                size_bytes: size,
                params,
                arch: geometry.as_ref().map(|g| g.arch.clone()).unwrap_or_default(),
                quant: bits.map(|b| format!("{b}bit")).or_else(|| quant_from_name(&name)),
                ctx_train: geometry.as_ref().map(|g| g.ctx_train).filter(|c| *c > 0),
                flags,
                flag_sources: src,
                mmproj: None,
                mmproj_bytes: 0,
                read_only,
                geometry,
                family: family_of(&name),
            });
        }
        // whisper.cpp models.
        for f in files.iter().filter(|f| {
            let n = f.file_name().unwrap().to_string_lossy().to_lowercase();
            n.starts_with("ggml-") && n.ends_with(".bin") && !n.contains("encoder")
        }) {
            let name = stem(f);
            let mut e = simple_entry(f, &name, "whisper", "stt", read_only);
            e.flags.audio = true;
            out.push(e);
        }
        // ONNX TTS (Piper voices have a .onnx.json next to them; Kokoro has kokoro*.onnx).
        for f in files.iter().filter(|f| f.extension().is_some_and(|x| x == "onnx")) {
            let n = f.file_name().unwrap().to_string_lossy().to_lowercase();
            let piper = f.with_extension("onnx.json").exists();
            if piper || n.starts_with("kokoro") {
                let mut e = simple_entry(f, &stem(f), "onnx", "tts", read_only);
                e.flags.audio = true;
                e.arch = if piper { "piper".into() } else { "kokoro".into() };
                out.push(e);
            }
        }
        out
    }
}

fn simple_entry(f: &Path, name: &str, format: &str, kind: &str, read_only: bool) -> ModelEntry {
    ModelEntry {
        id: slug(name),
        name: name.to_string(),
        path: f.to_path_buf(),
        format: format.into(),
        kind: kind.into(),
        class: kind.into(),
        size_bytes: f.metadata().map(|m| m.len()).unwrap_or(0),
        params: None,
        arch: format.into(),
        quant: quant_from_name(name),
        ctx_train: None,
        flags: Flags::default(),
        flag_sources: BTreeMap::new(),
        mmproj: None,
        mmproj_bytes: 0,
        read_only,
        geometry: None,
        family: family_of(name),
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::gguf::write::*;

    pub fn write_chat_gguf(path: &Path, arch: &str, params_dims: Vec<u64>, template: &str) {
        let bytes = build(
            &[
                ("general.architecture", W::Str(arch)),
                ("general.name", W::Str("Test Model")),
                (&format!("{arch}.block_count") as &str, W::U32(4)),
                (&format!("{arch}.embedding_length") as &str, W::U32(256)),
                (&format!("{arch}.attention.head_count") as &str, W::U32(4)),
                (&format!("{arch}.attention.head_count_kv") as &str, W::U32(2)),
                (&format!("{arch}.context_length") as &str, W::U32(8192)),
                ("tokenizer.chat_template", W::Str(template)),
            ],
            &[("token_embd.weight", params_dims)],
        );
        to_file(path, &bytes, 4096);
    }

    #[test]
    fn scans_gguf_with_projector_and_mlx_and_whisper() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = Config::default();
        cfg.paths.models_dir = tmp.path().to_string_lossy().into();
        cfg.paths.extra_model_dirs = vec![];
        let g = tmp.path().join("gemma");
        std::fs::create_dir_all(&g).unwrap();
        write_chat_gguf(&g.join("gemma-4-12b-it-Q5_K_M.gguf"), "gemma4", vec![12_000_000, 1000], "{% if tools %}{{ enable_thinking }}{% endif %}");
        let proj = build(&[("general.architecture", W::Str("clip")), ("general.type", W::Str("mmproj"))], &[]);
        to_file(&g.join("mmproj-F16.gguf"), &proj, 1000);
        let emb = tmp.path().join("embed");
        std::fs::create_dir_all(&emb).unwrap();
        let eb = build(&[("general.architecture", W::Str("bert")), ("bert.block_count", W::U32(24)), ("bert.pooling_type", W::U32(2))], &[("w", vec![500_000_000])]);
        to_file(&emb.join("bge-m3-Q8_0.gguf"), &eb, 2000);
        let mlx = tmp.path().join("mlx-community/Qwen3-4B-4bit");
        std::fs::create_dir_all(&mlx).unwrap();
        std::fs::write(mlx.join("config.json"), r#"{"model_type":"qwen3","num_hidden_layers":36,"hidden_size":2560,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"quantization":{"bits":4}}"#).unwrap();
        std::fs::write(mlx.join("model.safetensors"), vec![0u8; 1000]).unwrap();
        let w = tmp.path().join("whisper");
        std::fs::create_dir_all(&w).unwrap();
        std::fs::write(w.join("ggml-large-v3-turbo.bin"), b"x").unwrap();

        let lib = Library::new();
        let list = lib.scan(&cfg);
        let chat = list.iter().find(|e| e.format == "gguf" && e.kind == "chat").unwrap();
        assert!(chat.flags.vision && chat.flags.tools && chat.flags.thinking);
        assert_eq!(chat.quant.as_deref(), Some("Q5_K_M"));
        assert!(chat.mmproj.as_ref().unwrap().ends_with("mmproj-F16.gguf"));
        assert_eq!(chat.family, "gemma-4-12b-it");
        let e = list.iter().find(|e| e.kind == "embed").unwrap();
        assert_eq!(e.class, "embed");
        assert!(e.geometry.as_ref().unwrap().encoder_only);
        assert!(list.iter().any(|e| e.format == "mlx" && e.quant.as_deref() == Some("4bit")));
        assert!(list.iter().any(|e| e.kind == "stt"));
        assert_eq!(list.iter().filter(|e| e.path.to_string_lossy().contains("mmproj")).count(), 0);
    }

    #[test]
    fn quant_and_class() {
        assert_eq!(quant_from_name("Qwen3-14B-UD-Q4_K_XL").as_deref(), Some("UD-Q4_K_XL"));
        assert_eq!(quant_from_name("gpt-oss-20b-MXFP4").as_deref(), Some("MXFP4"));
        assert_eq!(quant_from_name("gemma-3-4b-it-4bit").as_deref(), Some("4bit"));
        let c = Config::default();
        assert_eq!(classify("chat", Some(12_000_000_000), "x", &c), "standard");
        assert_eq!(classify("chat", Some(20_900_000_000), "gpt-oss-20b", &c), "standard");
        assert_eq!(classify("chat", Some(26_000_000_000), "x", &c), "large");
        assert_eq!(classify("chat", Some(4_000_000_000), "x", &c), "small");
        assert_eq!(classify("chat", Some(1_000_000_000), "LFM2-1.2B-decision", &c), "decision");
    }
}
