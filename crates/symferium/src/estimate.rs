//! Memory estimates: weights + KV cache at the requested context + compute buffers + overhead.
//!
//! KV geometry comes from the model file itself (GGUF header or config.json), per layer, so hybrid
//! models are counted correctly: sliding-window layers (Gemma 2/3/4, Cohere 2), per-layer KV head
//! counts (recurrent/conv layers with 0 KV heads, e.g. LFM2), and different head dims for local and
//! global layers.

use crate::gguf::{GgufInfo, Value};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerKv {
    pub kv_heads: u32,
    pub k_len: u32,
    pub v_len: u32,
    pub swa: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Geometry {
    pub arch: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub vocab: u32,
    pub ctx_train: u64,
    pub sliding_window: u32,
    pub layers: Vec<LayerKv>,
    /// Encoder-only models (BERT family) keep no KV cache.
    pub encoder_only: bool,
    pub experts: u32,
    pub experts_used: u32,
}

const ENCODER_ARCHS: &[&str] =
    &["bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "t5encoder", "modern-bert", "neo-bert", "xlm-roberta"];

/// Default local:global pattern for archs whose GGUF has a window but no explicit pattern.
fn default_swa_period(arch: &str) -> Option<u32> {
    match arch {
        "gemma2" => Some(2),
        "gemma3" | "gemma3n" | "gemma4" | "gemma-embedding" => Some(6),
        "cohere2" => Some(4),
        _ => None,
    }
}

fn per_layer(v: Option<&Value>, n: usize, default: u32) -> Vec<u32> {
    match v {
        Some(Value::Arr(a)) => {
            let mut out: Vec<u32> = a.iter().map(|x| x.as_u64().unwrap_or(default as u64) as u32).collect();
            out.resize(n, default);
            out
        }
        Some(x) => vec![x.as_u64().map(|v| v as u32).unwrap_or(default); n],
        None => vec![default; n],
    }
}

impl Geometry {
    pub fn from_gguf(g: &GgufInfo) -> Option<Self> {
        let arch = g.arch().to_string();
        let n_layer = g.arch_u64("block_count")? as u32;
        let n_embd = g.arch_u64("embedding_length").unwrap_or(0) as u32;
        let n = n_layer as usize;
        let heads = per_layer(g.arch_val("attention.head_count"), n, 1);
        let head0 = heads.iter().copied().max().unwrap_or(1).max(1);
        let kv_heads = per_layer(g.arch_val("attention.head_count_kv"), n, head0);
        let k_len = g.arch_u64("attention.key_length").map(|v| v as u32).unwrap_or(n_embd / head0);
        let v_len = g.arch_u64("attention.value_length").map(|v| v as u32).unwrap_or(k_len);
        let k_swa = g.arch_u64("attention.key_length_swa").map(|v| v as u32).unwrap_or(k_len);
        let v_swa = g.arch_u64("attention.value_length_swa").map(|v| v as u32).unwrap_or(v_len);
        let window = g.arch_u64("attention.sliding_window").unwrap_or(0) as u32;
        let swa_flags: Vec<bool> = if window == 0 {
            vec![false; n]
        } else {
            match g.arch_val("attention.sliding_window_pattern") {
                Some(Value::Arr(a)) => {
                    let mut f: Vec<bool> = a.iter().map(|x| x.as_u64().unwrap_or(0) != 0).collect();
                    // A short array is a repeating pattern.
                    if !f.is_empty() && f.len() < n {
                        let p = f.clone();
                        f = (0..n).map(|i| p[i % p.len()]).collect();
                    }
                    f.resize(n, false);
                    f
                }
                Some(x) if x.as_u64().is_some() => period_flags(x.as_u64().unwrap() as u32, n),
                _ => match default_swa_period(&arch) {
                    Some(p) => period_flags(p, n),
                    None => vec![false; n],
                },
            }
        };
        let layers = (0..n)
            .map(|i| LayerKv {
                kv_heads: kv_heads[i],
                k_len: if swa_flags[i] { k_swa } else { k_len },
                v_len: if swa_flags[i] { v_swa } else { v_len },
                swa: swa_flags[i],
            })
            .collect();
        let vocab = g.arch_u64("vocab_size").or_else(|| g.get("tokenizer.ggml.tokens").and_then(Value::len)).unwrap_or(0) as u32;
        Some(Self {
            encoder_only: ENCODER_ARCHS.contains(&arch.as_str()),
            ctx_train: g.arch_u64("context_length").unwrap_or(0),
            experts: g.arch_u64("expert_count").unwrap_or(0) as u32,
            experts_used: g.arch_u64("expert_used_count").unwrap_or(0) as u32,
            arch,
            n_layer,
            n_embd,
            vocab,
            sliding_window: window,
            layers,
        })
    }

    /// From a Hugging Face / MLX config.json (VLM configs nest the LM under text_config).
    pub fn from_hf_config(cfg: &serde_json::Value) -> Option<Self> {
        let c = cfg.get("text_config").filter(|t| t.get("num_hidden_layers").is_some()).unwrap_or(cfg);
        let u = |k: &str| c.get(k).and_then(|v| v.as_u64());
        let n_layer = u("num_hidden_layers")? as u32;
        let n = n_layer as usize;
        let n_embd = u("hidden_size").unwrap_or(0) as u32;
        let heads = u("num_attention_heads").unwrap_or(1).max(1) as u32;
        let kv_heads = u("num_key_value_heads").unwrap_or(heads as u64) as u32;
        let head_dim = u("head_dim").map(|v| v as u32).unwrap_or(n_embd / heads);
        let global_head_dim = u("global_head_dim").map(|v| v as u32).unwrap_or(head_dim);
        let global_kv_heads = u("num_global_key_value_heads").map(|v| v as u32).unwrap_or(kv_heads);
        let window = u("sliding_window").unwrap_or(0) as u32;
        let types: Option<Vec<bool>> = c
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|t| t.as_str().is_some_and(|s| s.contains("sliding"))).collect());
        let model_type = c.get("model_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let swa_flags = match (window, types) {
            (0, _) => vec![false; n],
            (_, Some(mut t)) => {
                t.resize(n, false);
                t
            }
            (_, None) => match c.get("sliding_window_pattern").and_then(|v| v.as_u64()) {
                Some(p) => period_flags(p as u32, n),
                None => default_swa_period(&model_type.replace("_text", "")).map(|p| period_flags(p, n)).unwrap_or(vec![false; n]),
            },
        };
        let layers = swa_flags
            .iter()
            .map(|&swa| LayerKv {
                kv_heads: if swa { kv_heads } else { global_kv_heads },
                k_len: if swa { head_dim } else { global_head_dim },
                v_len: if swa { head_dim } else { global_head_dim },
                swa,
            })
            .collect();
        Some(Self {
            arch: model_type,
            n_layer,
            n_embd,
            vocab: u("vocab_size").unwrap_or(0) as u32,
            ctx_train: u("max_position_embeddings").unwrap_or(0),
            sliding_window: window,
            layers,
            encoder_only: false,
            experts: u("num_experts").or_else(|| u("num_local_experts")).unwrap_or(0) as u32,
            experts_used: u("num_experts_per_tok").unwrap_or(0) as u32,
        })
    }

    pub fn swa_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.swa).count()
    }
}

/// period p: layers i with i % p < p-1 are sliding-window, every p-th is global (llama.cpp convention).
fn period_flags(p: u32, n: usize) -> Vec<bool> {
    if p <= 1 {
        return vec![false; n];
    }
    (0..n).map(|i| (i as u32 % p) < p - 1).collect()
}

/// Bytes per element of a KV cache type (block formats include their scale).
pub fn cache_type_bytes(t: &str) -> f64 {
    match t {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        "turbo4" => 4.25 / 8.0,
        "turbo3" => 3.5 / 8.0,
        "turbo2" => 2.5 / 8.0,
        _ => 2.0,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EstimateInput {
    pub adapter: String,
    pub kind: String,
    pub weights_bytes: u64,
    pub projector_bytes: u64,
    pub geometry: Option<Geometry>,
    pub ctx: u64,
    pub parallel: u64,
    pub ubatch: u64,
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// llama.cpp context checkpoints kept for SWA/hybrid models (--ctx-checkpoints).
    pub ctx_checkpoints: u64,
    /// llama.cpp host prompt cache (--cache-ram), MB.
    pub cache_ram_mb: u64,
    pub overhead_mb: f64,
    /// Learned measured/estimated ratio for this adapter (1.0 = none).
    pub correction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Estimate {
    pub weights: u64,
    pub projector: u64,
    pub kv: u64,
    pub kv_full_layers: u64,
    pub kv_swa_layers: u64,
    pub swa_checkpoints: u64,
    pub prompt_cache: u64,
    pub compute: u64,
    pub overhead: u64,
    pub correction: f64,
    pub total: u64,
    pub notes: Vec<String>,
}

const MIB: f64 = 1024.0 * 1024.0;

pub fn estimate(i: &EstimateInput) -> Estimate {
    let mut e = Estimate { weights: i.weights_bytes, projector: i.projector_bytes, ..Default::default() };
    let ctx = i.ctx.max(1);
    let par = i.parallel.max(1);
    let ub = i.ubatch.max(1);
    let bk = cache_type_bytes(&i.cache_type_k);
    let bv = cache_type_bytes(&i.cache_type_v);
    let llama = i.adapter == "llamacpp";
    match i.kind.as_str() {
        "stt" => {
            e.compute = (i.weights_bytes as f64 * 0.2) as u64 + (200.0 * MIB) as u64;
            e.notes.push("speech model: weights + 20 % + 200 MB working memory".into());
        }
        "tts" => {
            e.compute = (i.weights_bytes as f64 * 0.5) as u64 + (300.0 * MIB) as u64;
            e.notes.push("speech model: weights + 50 % + 300 MB runtime".into());
        }
        _ => {
            if let Some(g) = &i.geometry {
                if g.encoder_only {
                    e.notes.push(format!("{}: encoder-only, no KV cache", g.arch));
                } else {
                    let mut swa_bytes_per_cell = 0.0;
                    for l in &g.layers {
                        let per_tok = l.kv_heads as f64 * (l.k_len as f64 * bk + l.v_len as f64 * bv);
                        if l.swa && g.sliding_window > 0 {
                            // llama.cpp keeps window + ubatch cells per sequence for SWA layers.
                            let cells = (ctx).min(par * (g.sliding_window as u64 + ub));
                            e.kv_swa_layers += (per_tok * cells as f64) as u64;
                            swa_bytes_per_cell += per_tok;
                        } else {
                            e.kv_full_layers += (per_tok * ctx as f64) as u64;
                        }
                    }
                    e.kv = e.kv_full_layers + e.kv_swa_layers;
                    let hybrid = g.layers.iter().any(|l| l.kv_heads == 0);
                    if llama && i.ctx_checkpoints > 0 && (g.swa_layers() > 0 || hybrid) {
                        // Each checkpoint copies the SWA (or recurrent) state of one sequence.
                        let one = swa_bytes_per_cell * (g.sliding_window as u64 + ub) as f64;
                        let one = if hybrid && one == 0.0 { 64.0 * MIB } else { one };
                        e.swa_checkpoints = (one * i.ctx_checkpoints as f64) as u64;
                        e.notes.push(format!("{} SWA/hybrid context checkpoints", i.ctx_checkpoints));
                    }
                    if g.ctx_train > 0 && ctx > g.ctx_train {
                        e.notes.push(format!("requested context {ctx} exceeds the trained {}", g.ctx_train));
                    }
                }
                // Logits for a physical batch + activations (rough, learned correction refines it).
                let logits = if g.encoder_only || i.kind == "embed" || i.kind == "rerank" {
                    0.0
                } else if llama {
                    ub as f64 * g.vocab as f64 * 4.0
                } else {
                    g.vocab as f64 * 4.0 * 8.0
                };
                let act = ub.min(if llama { ub } else { 2048 }) as f64 * g.n_embd as f64 * 4.0 * 12.0;
                e.compute = (logits + act) as u64;
            } else {
                // No geometry: assume 15 % of weights for KV + compute at default context.
                e.kv = (i.weights_bytes as f64 * 0.15) as u64;
                e.notes.push("no geometry found: KV assumed at 15 % of weights".into());
            }
            if i.projector_bytes > 0 {
                e.compute += (i.projector_bytes as f64 * 0.25 + 200.0 * MIB) as u64;
            }
            if llama && i.cache_ram_mb > 0 && !(i.kind == "embed" || i.kind == "rerank") {
                e.prompt_cache = (i.cache_ram_mb as f64 * MIB) as u64;
            }
        }
    }
    e.overhead = (i.overhead_mb * MIB) as u64 + if i.adapter == "mlx" || i.adapter == "tts" { (250.0 * MIB) as u64 } else { 0 };
    e.correction = if i.correction > 0.0 { i.correction } else { 1.0 };
    let raw = e.weights + e.projector + e.kv + e.swa_checkpoints + e.prompt_cache + e.compute + e.overhead;
    e.total = (raw as f64 * e.correction) as u64;
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write::*;

    /// Gemma-4 12B geometry as measured from the GGUF header (BRIEF §8):
    /// 48 blocks, 5 local : 1 global, local window 1024 with 8 KV heads × 256, global 1 KV head × 512.
    pub fn gemma4_12b() -> GgufInfo {
        let kv: Vec<u32> = (0..48).map(|i| if i % 6 == 5 { 1 } else { 8 }).collect();
        let bytes = build(
            &[
                ("general.architecture", W::Str("gemma4")),
                ("gemma4.block_count", W::U32(48)),
                ("gemma4.embedding_length", W::U32(3840)),
                ("gemma4.context_length", W::U32(262144)),
                ("gemma4.attention.head_count", W::U32(16)),
                ("gemma4.attention.head_count_kv", W::ArrU32(kv)),
                ("gemma4.attention.key_length", W::U32(512)),
                ("gemma4.attention.value_length", W::U32(512)),
                ("gemma4.attention.key_length_swa", W::U32(256)),
                ("gemma4.attention.value_length_swa", W::U32(256)),
                ("gemma4.attention.sliding_window", W::U32(1024)),
                ("gemma4.attention.sliding_window_pattern", W::ArrBool(vec![true, true, true, true, true, false])),
                ("gemma4.vocab_size", W::U32(262144)),
            ],
            &[],
        );
        crate::gguf::read_from(std::io::Cursor::new(bytes)).unwrap()
    }

    #[test]
    fn gemma4_geometry() {
        let g = Geometry::from_gguf(&gemma4_12b()).unwrap();
        assert_eq!(g.layers.len(), 48);
        assert_eq!(g.swa_layers(), 40);
        let global: Vec<_> = g.layers.iter().filter(|l| !l.swa).collect();
        assert!(global.iter().all(|l| l.kv_heads == 1 && l.k_len == 512));
        assert!(g.layers.iter().filter(|l| l.swa).all(|l| l.kv_heads == 8 && l.k_len == 256));
    }

    #[test]
    fn gemma4_kv_at_256k_f16_matches_hand_math() {
        let g = Geometry::from_gguf(&gemma4_12b()).unwrap();
        let e = estimate(&EstimateInput {
            adapter: "llamacpp".into(),
            kind: "chat".into(),
            geometry: Some(g),
            ctx: 262144,
            parallel: 1,
            ubatch: 512,
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            correction: 1.0,
            ..Default::default()
        });
        // Global: 8 layers × 262144 × 1 head × (512 K + 512 V) × 2 B = 4 GiB when K and V are stored
        // separately (llama.cpp does); 2 GiB if the engine shares K = V (blueprint figure 2.1 GB).
        assert_eq!(e.kv_full_layers, 8 * 262144 * 1024 * 2);
        // Local: 40 layers × (1024 + 512) cells × 8 heads × (256 + 256) × 2 B
        assert_eq!(e.kv_swa_layers, 40 * 1536 * 8 * 512 * 2);
    }

    #[test]
    fn quantised_kv_is_smaller_and_encoders_have_none() {
        let g = Geometry::from_gguf(&gemma4_12b()).unwrap();
        let base = EstimateInput {
            adapter: "llamacpp".into(),
            kind: "chat".into(),
            geometry: Some(g.clone()),
            ctx: 131072,
            parallel: 1,
            ubatch: 512,
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            ..Default::default()
        };
        let f16 = estimate(&base);
        let q8 = estimate(&EstimateInput { cache_type_k: "q8_0".into(), cache_type_v: "q8_0".into(), ..base.clone() });
        assert!(q8.kv < f16.kv * 6 / 10);
        let enc = estimate(&EstimateInput { geometry: Some(Geometry { encoder_only: true, ..g }), kind: "embed".into(), ..base });
        assert_eq!(enc.kv, 0);
    }

    #[test]
    fn hf_config_with_layer_types() {
        let cfg = serde_json::json!({
            "model_type": "gemma3_text", "num_hidden_layers": 6, "hidden_size": 1152, "num_attention_heads": 4,
            "num_key_value_heads": 1, "head_dim": 256, "sliding_window": 512, "vocab_size": 262144,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
            "max_position_embeddings": 32768
        });
        let g = Geometry::from_hf_config(&cfg).unwrap();
        assert_eq!(g.swa_layers(), 5);
        assert_eq!(g.ctx_train, 32768);
    }

    #[test]
    fn correction_scales_total() {
        let i = EstimateInput {
            adapter: "llamacpp".into(),
            kind: "chat".into(),
            weights_bytes: 1 << 30,
            correction: 1.5,
            ..Default::default()
        };
        let e = estimate(&i);
        let raw = e.weights + e.kv + e.compute + e.overhead;
        assert_eq!(e.total, (raw as f64 * 1.5) as u64);
    }
}
