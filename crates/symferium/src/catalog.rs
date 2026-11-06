//! Hugging Face catalog: search, model details, quant options, projector files, capability flags.
//! Flags come from tags, config.json and the chat template, each with its source; nothing is guessed.

use crate::config::Config;
use crate::estimate::{self, EstimateInput, Geometry};
use crate::library::{classify, family_of, flags_from_template, quant_from_name, Flags};
use crate::runtime::Runtime;
use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Cache {
    map: Mutex<HashMap<String, (Instant, serde_json::Value)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FileRef {
    pub path: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantOption {
    pub quant: String,
    pub files: Vec<FileRef>,
    pub size: Option<u64>,
    /// fits (now), needs_unload (fits the ceiling but not current free memory), too_big, unknown
    pub fit: String,
    pub estimate_bytes: Option<u64>,
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CatalogModel {
    pub repo: String,
    pub author: String,
    pub name: String,
    pub family: String,
    /// gguf, mlx, safetensors, onnx, whisper
    pub format: String,
    pub kind: String,
    pub class: String,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub params: Option<u64>,
    pub ctx: Option<u64>,
    pub moe: bool,
    pub flags: Flags,
    pub flag_sources: BTreeMap<String, String>,
    pub quants: Vec<QuantOption>,
    pub mmproj: Vec<FileRef>,
    pub gated: bool,
    pub base_model: Option<String>,
    pub detailed: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub author: Option<String>,
    /// gguf or mlx
    pub format: Option<String>,
    /// small, standard, large, embed, rerank, decision, stt, tts
    pub class: Option<String>,
    pub vision: Option<bool>,
    pub tools: Option<bool>,
    pub thinking: Option<bool>,
    pub limit: Option<u32>,
    pub sort: Option<String>,
}

const EXPAND: &[&str] =
    &["gguf", "safetensors", "siblings", "tags", "pipeline_tag", "downloads", "likes", "lastModified", "config", "cardData", "gated"];

fn kind_from(pipeline: Option<&str>, tags: &[String], name: &str) -> &'static str {
    let n = name.to_lowercase();
    let has = |t: &str| tags.iter().any(|x| x.eq_ignore_ascii_case(t));
    match pipeline {
        Some("automatic-speech-recognition") => return "stt",
        Some("text-to-speech" | "text-to-audio") => return "tts",
        Some("sentence-similarity" | "feature-extraction") if !n.contains("rerank") => return "embed",
        Some("text-ranking") => return "rerank",
        _ => {}
    }
    if n.contains("rerank") || has("reranker") {
        "rerank"
    } else if n.contains("embed") || has("sentence-transformers") && !n.contains("instruct") {
        "embed"
    } else if n.contains("whisper") {
        "stt"
    } else if n.contains("kokoro") || n.contains("piper") || n.contains("tts") {
        "tts"
    } else {
        "chat"
    }
}

fn format_from(tags: &[String], siblings: &[String], repo: &str) -> &'static str {
    let r = repo.to_lowercase();
    if siblings.iter().any(|s| s.ends_with(".gguf")) || tags.iter().any(|t| t == "gguf") {
        "gguf"
    } else if r.starts_with("mlx-community/") || tags.iter().any(|t| t == "mlx") {
        "mlx"
    } else if siblings.iter().any(|s| s.starts_with("ggml-") && s.ends_with(".bin")) {
        "whisper"
    } else if siblings.iter().any(|s| s.ends_with(".onnx")) {
        "onnx"
    } else {
        "safetensors"
    }
}

/// Group GGUF files by quant (split shards together); projectors apart.
pub fn gguf_quants(files: &[FileRef]) -> (Vec<QuantOption>, Vec<FileRef>) {
    let mut by: BTreeMap<String, Vec<FileRef>> = BTreeMap::new();
    let mut mmproj = vec![];
    for f in files.iter().filter(|f| f.path.to_lowercase().ends_with(".gguf")) {
        let base = f.path.rsplit('/').next().unwrap_or(&f.path).to_lowercase();
        if base.starts_with("mmproj") || base.contains("-mmproj") {
            mmproj.push(f.clone());
            continue;
        }
        let q = quant_from_name(&f.path).unwrap_or_else(|| "unknown".into());
        by.entry(q).or_default().push(f.clone());
    }
    let quants = by
        .into_iter()
        .map(|(quant, files)| {
            let size = files.iter().map(|f| f.size).sum::<Option<u64>>();
            QuantOption { quant, files, size, fit: "unknown".into(), ..Default::default() }
        })
        .collect();
    (quants, mmproj)
}

/// Parse one item of /api/models (search) or the /api/models/{repo} answer.
pub fn parse_model(v: &serde_json::Value, cfg: &Config) -> CatalogModel {
    let repo = v["id"].as_str().or(v["modelId"].as_str()).unwrap_or("").to_string();
    let (author, name) = repo.split_once('/').map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or((String::new(), repo.clone()));
    let tags: Vec<String> =
        v["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
    let siblings: Vec<FileRef> = v["siblings"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| {
                    Some(FileRef {
                        path: s["rfilename"].as_str()?.to_string(),
                        size: s["size"].as_u64().or(s["lfs"]["size"].as_u64()),
                        sha256: s["lfs"]["sha256"].as_str().map(String::from),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let names: Vec<String> = siblings.iter().map(|s| s.path.clone()).collect();
    let pipeline = v["pipeline_tag"].as_str().map(String::from);
    let format = format_from(&tags, &names, &repo).to_string();
    let kind = kind_from(pipeline.as_deref(), &tags, &name).to_string();
    let config = &v["config"];
    let params = v["gguf"]["total"].as_u64().or(v["safetensors"]["total"].as_u64());
    let mut flags = Flags::default();
    let mut src = BTreeMap::new();
    let has_tag = |t: &str| tags.iter().any(|x| x.eq_ignore_ascii_case(t));
    if matches!(pipeline.as_deref(), Some("image-text-to-text") | Some("any-to-any")) || has_tag("vision") || has_tag("multimodal") {
        flags.vision = true;
        src.insert("vision".into(), format!("pipeline/tags ({})", pipeline.clone().unwrap_or_default()));
    }
    if config.get("vision_config").is_some() {
        flags.vision = true;
        src.insert("vision".into(), "config.json vision_config".into());
    }
    if let Some(t) = v["gguf"]["chat_template"].as_str() {
        flags_from_template(t, &mut flags, &mut src);
    }
    if let Some(t) = config["tokenizer_config"]["chat_template"].as_str() {
        flags_from_template(t, &mut flags, &mut src);
    }
    if has_tag("function-calling") || has_tag("tool-use") || has_tag("tools") {
        flags.tools = true;
        src.insert("tools".into(), "tags".into());
    }
    if has_tag("reasoning") || has_tag("thinking") {
        flags.thinking = true;
        src.insert("thinking".into(), "tags".into());
    }
    if has_tag("agent") || has_tag("agentic") || has_tag("agents") {
        flags.agentic = true;
        src.insert("agentic".into(), "tags".into());
    }
    let lname = name.to_lowercase();
    if has_tag("code") || has_tag("coding") || lname.contains("coder") || lname.contains("devstral") || lname.contains("codestral") {
        flags.coding = true;
        src.insert("coding".into(), if has_tag("code") || has_tag("coding") { "tags" } else { "model name" }.into());
    }
    if kind == "stt" || kind == "tts" {
        flags.audio = true;
    }
    let moe = has_tag("moe") || lname.contains("-a3b") || lname.contains("-a4b") || lname.contains("moe") || lname.contains("gpt-oss");
    let ctx = v["gguf"]["context_length"].as_u64();
    let (quants, mmproj) = if format == "gguf" {
        gguf_quants(&siblings)
    } else {
        let total = siblings.iter().map(|s| s.size).sum::<Option<u64>>();
        let q = quant_from_name(&name).unwrap_or_else(|| "full".into());
        (vec![QuantOption { quant: q, files: siblings.clone(), size: total, fit: "unknown".into(), ..Default::default() }], vec![])
    };
    if !mmproj.is_empty() && !flags.vision {
        flags.vision = true;
        src.insert("vision".into(), "projector file in repo".into());
    }
    let class = classify(&kind, params, &name, cfg);
    CatalogModel {
        family: family_of(&name),
        repo,
        author,
        name,
        format,
        kind,
        class,
        pipeline_tag: pipeline,
        downloads: v["downloads"].as_u64().unwrap_or(0),
        likes: v["likes"].as_u64().unwrap_or(0),
        last_modified: v["lastModified"].as_str().map(String::from),
        params,
        ctx,
        moe,
        flags,
        flag_sources: src,
        quants,
        mmproj,
        gated: v["gated"].as_bool().unwrap_or(false) || v["gated"].as_str().is_some(),
        base_model: v["cardData"]["base_model"]
            .as_str()
            .map(String::from)
            .or_else(|| v["cardData"]["base_model"][0].as_str().map(String::from)),
        tags,
        detailed: false,
    }
}

/// Apply sizes and sha256 from /api/models/{repo}/tree/main?recursive=true.
pub fn apply_tree(m: &mut CatalogModel, tree: &serde_json::Value) {
    let mut files: HashMap<String, FileRef> = HashMap::new();
    for e in tree.as_array().into_iter().flatten() {
        if e["type"].as_str() != Some("file") {
            continue;
        }
        let path = e["path"].as_str().unwrap_or("").to_string();
        files.insert(
            path.clone(),
            FileRef { path, size: e["lfs"]["size"].as_u64().or(e["size"].as_u64()), sha256: e["lfs"]["oid"].as_str().map(String::from) },
        );
    }
    let fix = |f: &mut FileRef| {
        if let Some(t) = files.get(&f.path) {
            *f = t.clone();
        }
    };
    if m.format == "gguf" {
        let all: Vec<FileRef> = files.values().cloned().collect();
        let (q, p) = gguf_quants(&all);
        m.quants = q;
        m.mmproj = p;
    } else {
        for q in m.quants.iter_mut() {
            if q.files.is_empty() {
                q.files = files.values().cloned().collect();
            }
            q.files.iter_mut().for_each(fix);
            q.files.retain(|f| !f.path.ends_with(".md") && !f.path.starts_with('.'));
            q.size = q.files.iter().map(|f| f.size).sum::<Option<u64>>();
        }
    }
    m.mmproj.sort_by_key(|f| {
        let n = f.path.to_lowercase();
        if n.contains("f16") && !n.contains("bf16") {
            0
        } else if n.contains("bf16") {
            1
        } else if n.contains("q8") {
            2
        } else {
            3
        }
    });
}

/// Mark each quant as fits / needs_unload / too_big against the current budget, and recommend one.
pub fn annotate_fit(m: &mut CatalogModel, cfg: &Config, geometry: Option<&Geometry>, headroom: u64, cap: u64, correction: f64) {
    let class = cfg.class(&m.class);
    let engine = if m.format == "mlx" { class.engine_mlx.clone() } else { class.engine.clone() };
    let adapter = cfg.engines.get(&engine).map(|p| p.adapter.clone()).unwrap_or_default();
    let proj = if m.flags.vision { m.mmproj.first().and_then(|f| f.size).unwrap_or(0) } else { 0 };
    let mut ctx = class.ctx as u64;
    if let Some(c) = m.ctx.or(geometry.map(|g| g.ctx_train)).filter(|c| *c > 0) {
        ctx = ctx.min(c);
    }
    for q in m.quants.iter_mut() {
        let Some(size) = q.size else { continue };
        let e = estimate::estimate(&EstimateInput {
            adapter: adapter.clone(),
            kind: m.kind.clone(),
            weights_bytes: size,
            projector_bytes: proj,
            geometry: geometry.cloned(),
            ctx,
            parallel: class.parallel as u64,
            ubatch: class.ubatch as u64,
            cache_type_k: class.cache_type_k.clone(),
            cache_type_v: class.cache_type_v.clone(),
            ctx_checkpoints: class.ctx_checkpoints as u64,
            cache_ram_mb: class.cache_ram_mb as u64,
            overhead_mb: cfg.governor.engine_overhead_mb,
            correction,
        });
        q.estimate_bytes = Some(e.total);
        q.fit = if e.total <= headroom {
            "fits"
        } else if e.total <= cap {
            "needs_unload"
        } else {
            "too_big"
        }
        .into();
    }
    let pref = &cfg.catalog.quant_preference;
    let best = pref
        .iter()
        .find_map(|p| m.quants.iter().position(|q| q.quant.eq_ignore_ascii_case(p) && q.fit != "too_big"))
        .or_else(|| m.quants.iter().position(|q| q.fit == "fits"));
    if let Some(i) = best {
        m.quants[i].recommended = true;
    }
}

fn wanted(m: &CatalogModel, q: &SearchQuery, cfg: &Config) -> bool {
    let c = &cfg.catalog;
    if !c.formats.iter().any(|f| f == &m.format) {
        return false;
    }
    if m.pipeline_tag.as_ref().is_some_and(|p| c.exclude_pipeline_tags.contains(p)) {
        return false;
    }
    if m.downloads < c.min_downloads {
        return false;
    }
    if let Some(c) = &q.class {
        if &m.class != c {
            return false;
        }
    }
    if q.vision == Some(true) && !m.flags.vision {
        return false;
    }
    if q.tools == Some(true) && !m.flags.tools {
        return false;
    }
    if q.thinking == Some(true) && !m.flags.thinking {
        return false;
    }
    if m.class == "oversize" && q.class.is_none() {
        return false;
    }
    true
}

async fn get_json(rt: &Arc<Runtime>, url: &str) -> Result<serde_json::Value> {
    let cfg = rt.cfg();
    if let Some((t, v)) = rt.catalog_cache.map.lock().get(url) {
        if t.elapsed() < Duration::from_secs(cfg.catalog.cache_ttl_s) {
            return Ok(v.clone());
        }
    }
    let mut rb = rt.http.get(url).timeout(Duration::from_secs(cfg.download.timeout_s));
    if !cfg.catalog.hf_token.is_empty() {
        rb = rb.bearer_auth(&cfg.catalog.hf_token);
    }
    let resp = rb.send().await.with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("GET {url}: HTTP {}", resp.status()));
    }
    let v: serde_json::Value = resp.json().await?;
    rt.catalog_cache.map.lock().insert(url.to_string(), (Instant::now(), v.clone()));
    Ok(v)
}

pub async fn search(rt: &Arc<Runtime>, q: &SearchQuery) -> Result<Vec<CatalogModel>> {
    let cfg = rt.cfg();
    let base = cfg.catalog.hf_endpoint.trim_end_matches('/');
    let mut url = format!(
        "{base}/api/models?sort={}&direction=-1&limit={}",
        q.sort.clone().unwrap_or("downloads".into()),
        q.limit.unwrap_or(cfg.catalog.search_limit)
    );
    for e in EXPAND {
        url.push_str(&format!("&expand[]={e}"));
    }
    if let Some(s) = q.q.as_deref().filter(|s| !s.is_empty()) {
        url.push_str(&format!("&search={}", urlencode(s)));
    }
    if let Some(a) = q.author.as_deref().filter(|s| !s.is_empty()) {
        url.push_str(&format!("&author={}", urlencode(a)));
    }
    match q.format.as_deref() {
        Some("gguf") => url.push_str("&filter=gguf"),
        Some("mlx") => url.push_str("&filter=mlx"),
        _ => {}
    }
    match q.class.as_deref() {
        Some("embed") => url.push_str("&pipeline_tag=sentence-similarity"),
        Some("stt") => url.push_str("&pipeline_tag=automatic-speech-recognition"),
        Some("tts") => url.push_str("&pipeline_tag=text-to-speech"),
        _ => {}
    }
    let v = get_json(rt, &url).await?;
    let b = rt.budget();
    let corr = 1.0;
    let mut out: Vec<CatalogModel> = v
        .as_array()
        .ok_or_else(|| anyhow!("unexpected search answer"))?
        .iter()
        .map(|item| parse_model(item, &cfg))
        .filter(|m| wanted(m, q, &cfg))
        .collect();
    for m in out.iter_mut() {
        annotate_fit(m, &cfg, None, b.headroom_bytes, b.cap_bytes, corr);
        if cfg.catalog.hide_too_big {
            m.quants.retain(|q| q.fit != "too_big");
        }
    }
    if cfg.catalog.hide_too_big {
        out.retain(|m| !m.quants.is_empty());
    }
    Ok(out)
}

pub async fn detail(rt: &Arc<Runtime>, repo: &str) -> Result<CatalogModel> {
    let cfg = rt.cfg();
    let base = cfg.catalog.hf_endpoint.trim_end_matches('/');
    let mut url = format!("{base}/api/models/{repo}?");
    for e in EXPAND {
        url.push_str(&format!("expand[]={e}&"));
    }
    let info = get_json(rt, &url).await?;
    let mut m = parse_model(&info, &cfg);
    if let Ok(tree) = get_json(rt, &format!("{base}/api/models/{repo}/tree/main?recursive=true")).await {
        apply_tree(&mut m, &tree);
    }
    // config.json gives geometry and context for non-GGUF repos (GGUF geometry comes after download).
    let mut geometry = None;
    if m.format != "gguf" {
        if let Ok(c) = get_json(rt, &format!("{base}/{repo}/resolve/main/config.json")).await {
            geometry = Geometry::from_hf_config(&c);
            if m.ctx.is_none() {
                m.ctx = geometry.as_ref().map(|g| g.ctx_train).filter(|c| *c > 0);
            }
            if c.get("vision_config").is_some() && !m.flags.vision {
                m.flags.vision = true;
                m.flag_sources.insert("vision".into(), "config.json vision_config".into());
            }
        }
        if let Ok(t) = get_json(rt, &format!("{base}/{repo}/resolve/main/tokenizer_config.json")).await {
            if let Some(tpl) = t["chat_template"].as_str() {
                flags_from_template(tpl, &mut m.flags, &mut m.flag_sources);
            }
        }
    }
    let b = rt.budget();
    annotate_fit(&mut m, &cfg, geometry.as_ref(), b.headroom_bytes, b.cap_bytes, 1.0);
    if cfg.catalog.hide_too_big {
        m.quants.retain(|q| q.fit != "too_big");
    }
    m.detailed = true;
    Ok(m)
}

fn urlencode(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_gguf() -> serde_json::Value {
        serde_json::json!({
            "id": "unsloth/gemma-4-12b-it-GGUF", "pipeline_tag": "image-text-to-text",
            "tags": ["gguf", "gemma4", "vision", "base_model:google/gemma-4-12b-it"],
            "downloads": 123456, "likes": 789, "lastModified": "2026-09-01T00:00:00.000Z", "gated": false,
            "gguf": {"total": 12_200_000_000u64, "architecture": "gemma4", "context_length": 262144,
                     "chat_template": "{%- if tools %}...{%- endif %}{{ enable_thinking }}"},
            "siblings": [
                {"rfilename": "gemma-4-12b-it-Q4_K_M.gguf"}, {"rfilename": "gemma-4-12b-it-Q5_K_M.gguf"},
                {"rfilename": "gemma-4-12b-it-UD-Q4_K_XL.gguf"}, {"rfilename": "mmproj-F16.gguf"},
                {"rfilename": "mmproj-BF16.gguf"}, {"rfilename": "README.md"}
            ],
            "cardData": {"base_model": "google/gemma-4-12b-it"}
        })
    }

    fn fixture_tree() -> serde_json::Value {
        serde_json::json!([
            {"type": "file", "path": "gemma-4-12b-it-Q4_K_M.gguf", "size": 7_300_000_000u64, "lfs": {"oid": "aa".repeat(32), "size": 7_300_000_000u64}},
            {"type": "file", "path": "gemma-4-12b-it-Q5_K_M.gguf", "size": 8_400_000_000u64, "lfs": {"oid": "bb".repeat(32), "size": 8_400_000_000u64}},
            {"type": "file", "path": "gemma-4-12b-it-UD-Q4_K_XL.gguf", "size": 7_500_000_000u64, "lfs": {"oid": "cc".repeat(32), "size": 7_500_000_000u64}},
            {"type": "file", "path": "mmproj-F16.gguf", "size": 850_000_000u64, "lfs": {"oid": "dd".repeat(32), "size": 850_000_000u64}},
            {"type": "file", "path": "mmproj-BF16.gguf", "size": 850_000_000u64, "lfs": {"oid": "ee".repeat(32), "size": 850_000_000u64}},
            {"type": "file", "path": "README.md", "size": 1000},
            {"type": "directory", "path": "imatrix"}
        ])
    }

    #[test]
    fn gguf_repo_flags_quants_and_projector() {
        let cfg = Config::default();
        let mut m = parse_model(&fixture_gguf(), &cfg);
        assert_eq!(m.format, "gguf");
        assert_eq!(m.class, "standard");
        assert!(m.flags.vision && m.flags.tools && m.flags.thinking);
        assert_eq!(m.flag_sources["tools"], "chat template");
        assert_eq!(m.quants.len(), 3);
        assert_eq!(m.mmproj.len(), 2);
        assert_eq!(m.ctx, Some(262144));
        assert_eq!(m.base_model.as_deref(), Some("google/gemma-4-12b-it"));
        apply_tree(&mut m, &fixture_tree());
        assert_eq!(m.mmproj[0].path, "mmproj-F16.gguf");
        let q4 = m.quants.iter().find(|q| q.quant == "Q4_K_M").unwrap();
        assert_eq!(q4.size, Some(7_300_000_000));
        assert_eq!(q4.files[0].sha256.as_deref(), Some("aa".repeat(32).as_str()));
        // 12 GB headroom, 16 GB ceiling
        annotate_fit(&mut m, &cfg, None, 12 << 30, 16 << 30, 1.0);
        assert!(m.quants.iter().all(|q| q.estimate_bytes.is_some()));
        assert_eq!(m.quants.iter().filter(|q| q.recommended).count(), 1);
        assert!(m.quants.iter().find(|q| q.recommended).unwrap().quant == "Q4_K_M");
    }

    #[test]
    fn mlx_embed_and_whisper_repos() {
        let cfg = Config::default();
        let mlx = parse_model(
            &serde_json::json!({"id": "mlx-community/Qwen3-30B-A3B-4bit", "tags": ["mlx", "moe"], "safetensors": {"total": 30_500_000_000u64},
            "siblings": [{"rfilename": "config.json"}, {"rfilename": "model-00001-of-00004.safetensors"}]}),
            &cfg,
        );
        assert_eq!(mlx.format, "mlx");
        assert_eq!(mlx.class, "large");
        assert!(mlx.moe);
        assert_eq!(mlx.quants[0].quant, "4bit");
        let emb = parse_model(
            &serde_json::json!({"id": "ggml-org/bge-m3-Q8_0-GGUF", "pipeline_tag": "sentence-similarity", "tags": ["gguf"],
            "siblings": [{"rfilename": "bge-m3-q8_0.gguf"}]}),
            &cfg,
        );
        assert_eq!((emb.kind.as_str(), emb.class.as_str()), ("embed", "embed"));
        let rr = parse_model(
            &serde_json::json!({"id": "gpustack/bge-reranker-v2-m3-GGUF", "tags": ["gguf"], "siblings": [{"rfilename": "bge-reranker-v2-m3-Q4_K_M.gguf"}]}),
            &cfg,
        );
        assert_eq!(rr.kind, "rerank");
        let w = parse_model(
            &serde_json::json!({"id": "ggerganov/whisper.cpp", "pipeline_tag": "automatic-speech-recognition", "siblings": [{"rfilename": "ggml-large-v3-turbo.bin"}]}),
            &cfg,
        );
        assert_eq!((w.kind.as_str(), w.format.as_str()), ("stt", "whisper"));
    }

    #[test]
    fn filters() {
        let cfg = Config::default();
        let m = parse_model(&fixture_gguf(), &cfg);
        assert!(wanted(&m, &SearchQuery { vision: Some(true), class: Some("standard".into()), ..Default::default() }, &cfg));
        assert!(!wanted(&m, &SearchQuery { class: Some("small".into()), ..Default::default() }, &cfg));
        let mut img = m.clone();
        img.pipeline_tag = Some("text-to-image".into());
        assert!(!wanted(&img, &SearchQuery::default(), &cfg), "image generation is excluded");
        let mut gguf_only = cfg.clone();
        gguf_only.catalog.formats = vec!["gguf".into()];
        let mut st = m.clone();
        st.format = "safetensors".into();
        assert!(wanted(&st, &SearchQuery::default(), &cfg), "original weights are offered (MLX loads them)");
        assert!(!wanted(&st, &SearchQuery::default(), &gguf_only));
        let mut rare = m;
        rare.downloads = 3;
        assert!(!wanted(&rare, &SearchQuery::default(), &cfg));
    }
}
