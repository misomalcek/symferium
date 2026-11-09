# Symferium — build plan

Cloud project, 2026-11-06. Built from `docs/BRIEF.md`, `docs/SPEC.md`, `docs/REPOS.md` and the uploaded blueprints and sources. The brief wins where this plan is silent; the spec wins over both.

## 1. The three forks in BRIEF §10, settled

| fork | decision | why |
|---|---|---|
| MLX in v1? | **Yes, in v1, behind the adapter, and chosen per model by measurement.** | On M-series chips MLX usually decodes dense and MoE 4-bit models faster than llama.cpp, and mlx-community ships most of the spec's models. It still trails llama.cpp on vision projectors, embeddings, rerankers and KV-cache options, so it does not replace llama.cpp. `symferium bench` runs the same prompt on both engines and records tok/s; the catalog then offers the faster engine as that model's default. MLX runs as `mlx_lm.server` / `mlx_vlm.server` (or Rapid-MLX when installed) from a Python env that Symferium creates with `uv`. |
| UI stack | **Plain HTML + vanilla JS, embedded in the binary and served by axum.** | Every requirement (all settings as a generated form, live model list, catalog search, load/unload, live memory and tok/s over Server-Sent Events, logs) is a fetch call or an event stream. No build step, no node in the deploy. The settings form is generated from the config schema, so new settings show up without UI work. |
| Ports | **Control API and UI on 7700, model ports from 7701 up, all configurable.** | Free on the Mac; Factorium keeps 8080–8084 untouched. |

## 2. Architecture

```
symferium (one Rust binary, launchd LaunchAgent, nice 10, background I/O)
├── control  :7700  /api/*  + embedded UI  + /api/events (SSE event bus)
├── config   one TOML (~/Library/Application Support/Symferium/symferium.toml), every value has a schema entry:
│            type, default, min/max, unit, description, agent_writable
├── probe    total/free memory, memory pressure, Metal recommendedMaxWorkingSetSize, CPU P/E cores, GPU cores, disk
├── library  scans model folders (incl. ~/.factorium/models, read-only) → local models, no re-download
├── catalog  Hugging Face API → families, quants, mmproj, flags from config.json / chat template / GGUF header
├── download resumable (HTTP Range), sha256 against the HF LFS oid, queue with progress events
├── governor estimate = weights + KV(ctx, cache type, SWA geometry from GGUF/config) + compute buffer + engine overhead
│            × learned correction per engine; refuse with numbers; live pressure watchdog
├── supervisor  one task per loaded model: spawn engine on an internal port, health-check, restart with backoff,
│               stop gracefully, log ring buffer, measured phys_footprint per process
├── proxy    Symferium owns every public model port (7701…). It forwards to the engine's internal port, streams SSE
│            through untouched, normalises routes (whisper /inference → /v1/audio/transcriptions, rerank aliases),
│            answers /health and /v1/models itself, counts tokens and tok/s, and holds the middleware hook (§5)
└── adapters (engine profiles in TOML; adding a profile needs no code)
    ├── llamacpp      llama-server: chat, vision (--mmproj), embeddings, rerank; flash-attn, q8_0 KV, SWA
    ├── llamacpp-tq   same adapter, binary path pointing at a TurboQuant fork (turbo3/turbo4 KV); off by default
    ├── mlx           mlx_lm.server / mlx_vlm.server / rapid-mlx: chat + vision on MLX weights
    ├── whispercpp    whisper-server (Metal): STT
    └── tts           bundled Python sidecar (kokoro-onnx, piper): /v1/audio/speech
```

Why the proxy: clients see a port that never changes while the engine behind it restarts; every engine looks the same (`/health`, `/v1/models`, OpenAI routes); tok/s is measured identically for every engine, which is what the MLX-vs-llama.cpp comparison needs; and the symbiosis layer gets one place to hook requests without touching engines. Cost: one local hop, well under 1 ms.

### Default engine and settings per class (all overridable per model)

| class | default engine | context default | KV cache | notes |
|---|---|---|---|---|
| standard 6–20B | llama.cpp GGUF Q4_K_M/Q5_K_M (+mmproj when vision); MLX 4-bit when `bench` shows it faster and no projector is needed | 32 768 | q8_0 K and V, flash-attn | Gemma-4 12B keeps Q5_K_M + mmproj for Factorium parity |
| large (alone) | MLX 4-bit for MoE (26B-A4B, 30B-A3B), llama.cpp Q4_K_M otherwise | 16 384 | q8_0 | governor admits only when the rest fits |
| small < 5B | MLX 4-bit (text), llama.cpp for VL with mmproj | 32 768 | f16 | cheap enough to keep warm |
| embed / rerank | llama.cpp `--embedding` / `--reranking` | model max (8192 for bge-m3) | f16 | `-ub` = `-b` so long inputs fit in one batch |
| decision / function | llama.cpp (GGUF) or MLX | 4 096 | f16 | logprobs passed through for scoring |
| STT | whisper.cpp large-v3-turbo / small | — | — | |
| TTS | Kokoro (EN), Piper (SK + EN) | — | — | |

Advanced options from the research (TurboQuant KV, speculative decoding with a draft model, MTP) are engine flags in the profile: visible, editable, measured, off by default until `bench` shows a gain on this Mac. Engines that would replace llama.cpp/MLX (turbo-fieldfare, colibri/zunzuncito, uzu, crabml, shimmy) can be added later as profiles; none is needed for v1.

## 3. Governor (the no-bricking rules)

- Budget = min(total − OS reserve, Metal recommended working set) − memory used by everything that is not Symferium. Default OS reserve: max(4 GB, 20 % of total). Shown and editable.
- Estimate per load from the file itself: GGUF header (layers, KV heads, head dims, sliding window, SWA pattern) or `config.json` for MLX. Gemma-4 uses the corrected geometry from BRIEF §8.
- Refuse a load that breaks the budget; the 409 answer lists each term in bytes and the free headroom.
- After the load, measure `phys_footprint` of the engine process (which includes Metal allocations on unified memory), store the error per engine, and fold it into later estimates.
- Watchdog: on "critical" memory pressure, new loads are refused; optional policy unloads the lowest-priority model (off by default, editable).
- Large models: a class flag `exclusive_large` keeps a large model from loading beside another chat model unless the user turns that off.

## 4. Agent API

- `GET /api/tools` returns a manifest (name, description, JSON-schema args) for: list_models, search_catalog, download, load, unload, get_settings, set_setting, metrics, machine.
- `POST /api/tools/{name}` runs a tool. Agents can change only settings marked `agent_writable`, always through the governor.
- Bound to 127.0.0.1. Optional bearer token (off by default) for the control API.

## 5. Symbiosis layer (v2, off by default)

v1 ships the two seams the brief asks for: a request/response middleware chain in the proxy, and the event bus (`/api/events`, also a local Unix socket). The design for Ghost, Whisper, Nexus and Heron and how they reach qdrant, the graph store and the memory folder is in `docs/SYMBIOSIS.md`. It is built after the v1 gates pass.

## 6. Milestones and gates

One branch (`claude/project-thread-oin30t`), one PR, one commit series per milestone, so the Mac seat can check out any milestone.

| | cloud builds and proves | the Mac proves (sonnet seat) |
|---|---|---|
| M0 | daemon, config + schema, probe, control API, UI shell; `cargo test` green on Linux | `/api/machine` shows the real numbers |
| M1 | llama.cpp adapter, proxy, supervisor, restart; tested against a mock engine binary | tiny GGUF answers on its port; kill it, it restarts |
| M2 | HF catalog + downloader against recorded API fixtures; resume + sha256 tests | vision model downloads with its mmproj and answers about an image |
| M3 | governor estimates (GGUF header parser tests), refusal with numbers, persisted set, launchd plist | over-budget load refused; reboot restores the set |
| M4 | embed/rerank, whisper.cpp, TTS sidecar, MLX adapters; `symferium bench` | each OpenAI route answers; MLX vs llama.cpp tok/s on one model |
| M5 | parity preset (gemma-4 12B + mmproj, bge-m3, reranker on spare ports) | Factorium eval and an Una run match; Michal decides cutover |
| M6 | `/api/tools` + agent_writable policy | an agent loads, uses, unloads through tools only |

CI: `cargo test` + clippy on ubuntu for every push; a macOS (Apple Silicon) job that builds, installs llama.cpp from Homebrew and runs the M1 smoke with a ~100 MB GGUF.

## 7. What the cloud cannot do

- No Metal and no huggingface.co from this container (HF answers 403 through the proxy). Catalog code is tested against recorded API responses; the first live HF call happens on the Mac or in macOS CI.
- No real model weights; engine behaviour is tested with a mock engine that speaks the same HTTP.
- Every number in the gates above is produced on the Mac and reported with its command.
