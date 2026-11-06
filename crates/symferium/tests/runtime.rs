//! End-to-end tests of the supervisor, proxy, governor, restore set and agent tools, with the
//! mock engine standing in for llama-server / mlx_lm.server.

use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use symferium::config::{Config, GB};
use symferium::gguf::write::{build, to_file, W};
use symferium::probe::{FakeProbe, MachineInfo};
use symferium::runtime::LoadError;
use symferium::state::{LoadRequest, StateStore};
use symferium::Runtime;

const MOCK: &str = env!("CARGO_BIN_EXE_mock-engine");

fn big_machine() -> MachineInfo {
    MachineInfo {
        total_bytes: (64.0 * GB) as u64,
        available_bytes: (50.0 * GB) as u64,
        pressure: "normal".into(),
        perf_cores: 4,
        ..Default::default()
    }
}

fn write_models(dir: &Path) {
    let g = dir.join("tiny");
    std::fs::create_dir_all(&g).unwrap();
    let bytes = build(
        &[
            ("general.architecture", W::Str("llama")),
            ("llama.block_count", W::U32(2)),
            ("llama.embedding_length", W::U32(64)),
            ("llama.attention.head_count", W::U32(4)),
            ("llama.attention.head_count_kv", W::U32(2)),
            ("llama.context_length", W::U32(4096)),
            ("tokenizer.chat_template", W::Str("{% if tools %}{% endif %}")),
        ],
        &[("token_embd.weight", vec![64, 1000])],
    );
    to_file(&g.join("tiny-chat-Q4_K_M.gguf"), &bytes, 1 << 20);
    let mlx = dir.join("mlx-community/tiny-mlx-4bit");
    std::fs::create_dir_all(&mlx).unwrap();
    std::fs::write(mlx.join("config.json"), r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"quantization":{"bits":4}}"#).unwrap();
    std::fs::write(mlx.join("model.safetensors"), vec![0u8; 4096]).unwrap();
}

fn test_config(dir: &Path, port_base: u16) -> Config {
    let mut c = Config::default();
    c.paths.models_dir = dir.join("models").to_string_lossy().into();
    c.paths.extra_model_dirs = vec![];
    c.paths.state_dir = dir.join("state").to_string_lossy().into();
    c.paths.logs_dir = dir.join("logs").to_string_lossy().into();
    c.server.control_port = port_base;
    c.server.model_port_start = port_base + 1;
    c.server.model_port_end = port_base + 20;
    c.server.internal_port_start = port_base + 100;
    c.server.internal_port_end = port_base + 150;
    c.server.proxy_wait_ready_s = 10;
    c.supervisor.health_interval_ms = 100;
    c.supervisor.restart_backoff_ms = 100;
    c.supervisor.startup_timeout_s = 20;
    c.supervisor.max_restarts = 2;
    c.governor.measure_after_s = 1;
    c.governor.min_free_gb = 4.0;
    c.governor.exclusive_large = true;
    for p in c.engines.values_mut() {
        p.binary = MOCK.into();
    }
    c
}

fn runtime(dir: &Path, port_base: u16, machine: MachineInfo, edit: impl FnOnce(&mut Config)) -> Arc<Runtime> {
    write_models(&dir.join("models"));
    let mut cfg = test_config(dir, port_base);
    edit(&mut cfg);
    cfg.validate().unwrap();
    Runtime::new(cfg, None, Arc::new(FakeProbe::new(machine)), StateStore::load(dir.join("state/state.json")))
}

async fn wait_for(mut f: impl FnMut() -> bool, secs: u64) {
    let end = std::time::Instant::now() + Duration::from_secs(secs);
    while !f() {
        assert!(std::time::Instant::now() < end, "condition not met in {secs} s");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chat_model_serves_streams_restarts_and_unloads() {
    let tmp = tempfile::tempdir().unwrap();
    let rt = runtime(tmp.path(), 31700, big_machine(), |c| {
        c.engines.get_mut("llamacpp").unwrap().env.insert("MOCK_EXIT_AFTER".into(), "3".into());
    });
    let lib = rt.library.list();
    let chat = lib.iter().find(|e| e.format == "gguf").expect("tiny gguf in library");
    assert!(chat.flags.tools);
    let v = rt.load(LoadRequest { model: chat.id.clone(), alias: Some("tiny".into()), wait: true, ..Default::default() }).await.unwrap();
    assert_eq!(v.port, 31701);
    assert!(v.command.contains("--ctx-checkpoints"));
    let base = format!("http://127.0.0.1:{}", v.port);
    let http = reqwest::Client::new();

    let h: Value = http.get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(h["status"], "ok");
    let m: Value = http.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(m["data"][0]["id"], "tiny");

    // non-streaming
    let r: Value = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r["choices"][0]["message"]["content"].as_str().unwrap().starts_with("t0"));

    // streaming, passed through as SSE
    let body = http
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({"model": "tiny", "stream": true, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("data: ") && body.contains("[DONE]"), "{body}");
    let inst = rt.get("tiny").unwrap();
    wait_for(|| inst.metrics.lock().requests >= 2, 5).await;
    {
        let m = inst.metrics.lock();
        assert_eq!(m.last_decode_tps, 42.0, "engine timings are used when present");
        assert!(m.completion_tokens >= 16);
    }

    // The third completion makes the mock exit: the supervisor restarts it on the same port.
    let _ = http.post(format!("{base}/v1/chat/completions")).json(&json!({"messages": []})).send().await;
    wait_for(|| inst.status.read().restarts >= 1, 10).await;
    wait_for(|| inst.state() == "ready", 10).await;
    let again = http.post(format!("{base}/v1/chat/completions")).json(&json!({"messages": []})).send().await.unwrap();
    assert!(again.status().is_success());

    // Measured footprint is recorded and the model set is persisted.
    wait_for(|| inst.status.read().measured_bytes.is_some(), 10).await;
    assert_eq!(rt.state.data.lock().model_set.len(), 1);
    let saved = std::fs::read_to_string(tmp.path().join("state/state.json")).unwrap();
    assert!(saved.contains("\"tiny\""));

    rt.unload("tiny", true).await.unwrap();
    assert!(rt.get("tiny").is_none());
    assert!(rt.state.data.lock().model_set.is_empty());
    assert!(http.get(format!("{base}/health")).send().await.is_err(), "port is closed after unload");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn governor_refuses_with_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    let mut m = big_machine();
    m.available_bytes = (4.2 * GB) as u64;
    let rt = runtime(tmp.path(), 31800, m, |_| {});
    let id = rt.library.list().iter().find(|e| e.format == "gguf").unwrap().id.clone();
    match rt.load(LoadRequest { model: id, ..Default::default() }).await {
        Err(LoadError::Refused(d)) => {
            assert!(d.reasons[0].contains("below the minimum 4.00 GB"), "{:?}", d.reasons);
            assert!(d.estimate.total > 0);
        }
        other => panic!("expected refusal, got {other:?}", other = other.map(|v| v.id)),
    }
    assert!(rt.events.history().iter().any(|e| e.kind == "governor.refused"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mlx_profile_rewrites_model_field() {
    let tmp = tempfile::tempdir().unwrap();
    let rt = runtime(tmp.path(), 31900, big_machine(), |_| {});
    let mlx = rt.library.list().into_iter().find(|e| e.format == "mlx").unwrap();
    let v = rt.load(LoadRequest { model: mlx.id.clone(), wait: true, persist: Some(false), ..Default::default() }).await.unwrap();
    assert_eq!(v.engine, "mlx");
    let http = reqwest::Client::new();
    http.post(format!("http://127.0.0.1:{}/v1/chat/completions", v.port))
        .json(&json!({"model": "client-name", "stream": true, "messages": []}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let last: Value = http.get(format!("http://127.0.0.1:{}/debug/last-body", v.internal_port)).send().await.unwrap().json().await.unwrap();
    assert_eq!(last["model"], mlx.path.to_string_lossy().as_ref(), "mlx servers get the served model path");
    assert_eq!(last["stream_options"]["include_usage"], true);
    rt.unload(&v.id, true).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crashing_engine_ends_failed_and_port_says_why() {
    let tmp = tempfile::tempdir().unwrap();
    // An engine that exits at once (a script, so the path exists on Linux and macOS alike).
    let crash = tmp.path().join("crash.sh");
    std::fs::write(&crash, "#!/bin/sh\nexit 1\n").unwrap();
    std::fs::set_permissions(&crash, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let crash = crash.to_string_lossy().into_owned();
    let rt = runtime(tmp.path(), 32000, big_machine(), |c| {
        c.engines.get_mut("llamacpp").unwrap().binary = crash;
        c.server.proxy_wait_ready_s = 1;
    });
    let id = rt.library.list().iter().find(|e| e.format == "gguf").unwrap().id.clone();
    let v = rt.load(LoadRequest { model: id, persist: Some(false), ..Default::default() }).await.unwrap();
    let inst = rt.get(&v.id).unwrap();
    wait_for(|| inst.state() == "failed", 15).await;
    let r = reqwest::get(format!("http://127.0.0.1:{}/v1/models", v.port)).await.unwrap();
    assert!(r.status().is_success(), "/v1/models answers even when failed");
    let r = reqwest::Client::new().post(format!("http://127.0.0.1:{}/v1/chat/completions", v.port)).json(&json!({})).send().await.unwrap();
    assert_eq!(r.status(), 503);
    assert!(r.text().await.unwrap().contains("failed"));
    rt.unload(&v.id, true).await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn agent_uses_tools_only() {
    let tmp = tempfile::tempdir().unwrap();
    let rt = runtime(tmp.path(), 32100, big_machine(), |_| {});
    let listener = tokio::net::TcpListener::bind("127.0.0.1:32100").await.unwrap();
    let app = symferium::api::router(rt.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let api = "http://127.0.0.1:32100/api";
    let http = reqwest::Client::new();
    let tools: Value = http.get(format!("{api}/tools")).send().await.unwrap().json().await.unwrap();
    let names: Vec<&str> = tools["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    for n in ["list_models", "load", "unload", "set_setting", "search_catalog", "metrics"] {
        assert!(names.contains(&n), "{n} missing");
    }
    let call = |name: &'static str, args: Value| {
        let http = http.clone();
        async move { http.post(format!("{api}/tools/{name}")).json(&args).send().await.unwrap() }
    };
    // Not agent-writable.
    let r = call("set_setting", json!({"key": "governor.min_free_gb", "value": 1.0})).await;
    assert_eq!(r.status(), 400);
    // Agent-writable.
    let r = call("set_setting", json!({"key": "classes.small.ctx", "value": 8192})).await;
    assert!(r.status().is_success());
    assert_eq!(rt.cfg().classes["small"].ctx, 8192);

    let list: Value = call("list_models", json!({})).await.json().await.unwrap();
    let id =
        list["result"]["local"].as_array().unwrap().iter().find(|m| m["format"] == "gguf").unwrap()["id"].as_str().unwrap().to_string();
    let est: Value = call("estimate", json!({"model": id})).await.json().await.unwrap();
    assert_eq!(est["result"]["allowed"], true);
    let loaded: Value =
        call("load", json!({"model": id, "alias": "agent-tiny", "wait": true, "persist": false})).await.json().await.unwrap();
    let port = loaded["result"]["port"].as_u64().unwrap();
    let r = http.post(format!("http://127.0.0.1:{port}/v1/chat/completions")).json(&json!({"messages": []})).send().await.unwrap();
    assert!(r.status().is_success());
    let un: Value = call("unload", json!({"id": "agent-tiny"})).await.json().await.unwrap();
    assert_eq!(un["ok"], true);

    // Settings API for the UI: schema + masked values, and validation.
    let s: Value = http.get(format!("{api}/settings")).send().await.unwrap().json().await.unwrap();
    assert!(s["schema"].as_array().unwrap().len() > 80);
    let bad = http.put(format!("{api}/settings")).json(&json!({"key": "server.model_port_start", "value": 80})).send().await.unwrap();
    assert_eq!(bad.status(), 400);
    let m: Value = http.get(format!("{api}/machine")).send().await.unwrap().json().await.unwrap();
    assert!(m["budget"]["headroom_bytes"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admin_and_agent_tokens() {
    let tmp = tempfile::tempdir().unwrap();
    let rt = runtime(tmp.path(), 32200, big_machine(), |c| {
        c.server.admin_token = "adm".into();
        c.server.agent_token = "agt".into();
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:32200").await.unwrap();
    let app = symferium::api::router(rt.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let http = reqwest::Client::new();
    let api = "http://127.0.0.1:32200/api";
    assert_eq!(http.get(format!("{api}/machine")).send().await.unwrap().status(), 401);
    assert_eq!(http.get(format!("{api}/machine")).bearer_auth("agt").send().await.unwrap().status(), 401);
    assert!(http.get(format!("{api}/machine")).bearer_auth("adm").send().await.unwrap().status().is_success());
    assert!(http.get(format!("{api}/tools")).bearer_auth("agt").send().await.unwrap().status().is_success());
    assert!(http.get("http://127.0.0.1:32200/").send().await.unwrap().status().is_success(), "UI shell loads; it asks for the token");
}
