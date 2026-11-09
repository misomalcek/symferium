//! Benchmark: the same prompt on a loaded chat model, prefill and decode tok/s measured at its port.
//! `compare` loads several builds of a model one after another (e.g. GGUF on llama.cpp and MLX) so
//! the faster engine can become that family's default.

use crate::events::now_ms;
use crate::runtime::Runtime;
use crate::state::{BenchResult, LoadRequest};
use anyhow::{anyhow, bail, Result};
use futures::StreamExt;
use std::sync::Arc;
use std::time::Instant;

pub async fn run(
    rt: &Arc<Runtime>,
    instance: &str,
    prompt: Option<String>,
    max_tokens: Option<u32>,
    runs: Option<u32>,
) -> Result<BenchResult> {
    let cfg = rt.cfg();
    let inst = rt.get(instance).ok_or_else(|| anyhow!("{instance} is not loaded"))?;
    if inst.model.kind != "chat" {
        bail!("bench measures chat models; {instance} is {}", inst.model.kind);
    }
    let prompt = prompt.unwrap_or(cfg.bench.prompt.clone());
    let max_tokens = max_tokens.unwrap_or(cfg.bench.max_tokens);
    let runs = runs.unwrap_or(cfg.bench.runs).max(1);
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", inst.port);
    let body = serde_json::json!({
        "model": inst.id, "stream": true, "max_tokens": max_tokens, "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "stream_options": {"include_usage": true}
    });
    let mut acc = BenchResult {
        model: inst.model.id.clone(),
        family: inst.model.family.clone(),
        engine: inst.spec.engine.clone(),
        format: inst.model.format.clone(),
        quant: inst.model.quant.clone(),
        ts_ms: now_ms(),
        runs,
        command: format!("curl -N {url} -d '{}'", body),
        ..Default::default()
    };
    let (mut ttft_sum, mut pre_sum, mut dec_sum) = (0.0, 0.0, 0.0);
    for _ in 0..runs {
        let started = Instant::now();
        let resp = rt.http.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            bail!("bench request: HTTP {}", resp.status());
        }
        let mut s = resp.bytes_stream();
        let mut buf = Vec::new();
        let (mut first, mut last) = (None, None);
        let (mut chunks, mut pt, mut ct) = (0u64, 0u64, 0u64);
        let (mut eng_dec, mut eng_pre) = (0.0, 0.0);
        while let Some(b) = s.next().await {
            buf.extend_from_slice(&b?);
            while let Some(pos) = buf.iter().position(|c| *c == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(d) = line.trim().strip_prefix("data:") else { continue };
                let Ok(v) = serde_json::from_str::<serde_json::Value>(d.trim()) else { continue };
                let delta = &v["choices"][0]["delta"];
                if ["content", "reasoning_content", "reasoning"].iter().any(|k| delta[*k].as_str().is_some_and(|x| !x.is_empty())) {
                    let now = Instant::now();
                    first.get_or_insert(now);
                    last = Some(now);
                    chunks += 1;
                }
                if v["usage"].is_object() {
                    pt = v["usage"]["prompt_tokens"].as_u64().unwrap_or(pt);
                    ct = v["usage"]["completion_tokens"].as_u64().unwrap_or(ct);
                }
                if v["timings"].is_object() {
                    eng_dec = v["timings"]["predicted_per_second"].as_f64().unwrap_or(0.0);
                    eng_pre = v["timings"]["prompt_per_second"].as_f64().unwrap_or(0.0);
                }
            }
        }
        let ct = if ct > 0 { ct } else { chunks };
        let first = first.ok_or_else(|| anyhow!("no tokens came back"))?;
        let ttft = first.duration_since(started).as_secs_f64();
        let gen = last.unwrap_or(first).duration_since(first).as_secs_f64();
        let dec = if eng_dec > 0.0 {
            eng_dec
        } else if gen > 0.0 {
            (ct.saturating_sub(1)) as f64 / gen
        } else {
            0.0
        };
        let pre = if eng_pre > 0.0 {
            eng_pre
        } else if ttft > 0.0 && pt > 0 {
            pt as f64 / ttft
        } else {
            0.0
        };
        ttft_sum += ttft;
        pre_sum += pre;
        dec_sum += dec;
        acc.prompt_tokens = pt;
        acc.completion_tokens += ct;
    }
    let n = runs as f64;
    acc.ttft_ms = ttft_sum / n * 1000.0;
    acc.prefill_tps = pre_sum / n;
    acc.decode_tps = dec_sum / n;
    acc.completion_tokens /= runs as u64;
    rt.state.update(|d| d.bench.push(acc.clone()));
    rt.events.emit("bench.done", serde_json::to_value(&acc).unwrap_or_default());
    Ok(acc)
}

/// Load each model alone, bench it, unload it. Runs in the background; results arrive as events.
pub fn compare(rt: &Arc<Runtime>, models: Vec<String>, ctx: Option<u32>) {
    let rt = rt.clone();
    tokio::spawn(async move {
        let mut results = vec![];
        for m in models {
            let alias = format!("bench-{}", m.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect::<String>());
            let req =
                LoadRequest { model: m.clone(), alias: Some(alias.clone()), ctx, persist: Some(false), wait: true, ..Default::default() };
            match rt.load(req).await {
                Ok(_) => {
                    match run(&rt, &alias, None, None, None).await {
                        Ok(r) => results.push(r),
                        Err(e) => rt.events.emit("bench.failed", serde_json::json!({"model": m, "error": format!("{e:#}")})),
                    }
                    let _ = rt.unload(&alias, true).await;
                }
                Err(e) => rt.events.emit("bench.failed", serde_json::json!({"model": m, "error": e.to_string()})),
            }
        }
        rt.events.emit("bench.compare_done", serde_json::to_value(&results).unwrap_or_default());
    });
}
