# Symferium — build brief

Prime (Claude Code, Opus), 2026-11-06. Michal's spec is `docs/SPEC.md`; this brief turns it into a build. Where they disagree, the spec wins. Ask before deviating from either.

## 1. What Symferium is

A standalone local AI runtime for one Apple Silicon machine.
- It downloads, loads and unloads many models at once, each served on its own localhost port.
- It is controlled from a thin browser UI and a local API that any local agent can use.
- It is not part of Factorium. Factorium is its first client, but other local apps will use it too.
- It starts clean, from scratch. The current inference setup (llama.cpp under Factorium's launchd jobs) keeps running untouched until Symferium has been verified to replace it. Only then is the inference code taken out of Factorium.

## 2. The machine (measured 2026-11-06)

```
Mac mini M4 (Mac16,10) · 24 GB unified memory · macOS 27.2 · rustc 1.98.1
today, under Factorium's launchd jobs:
  :8080  llama-server  gemma-4-12b-it Q5_K_M + mmproj (vision), -c 131072, -np 1, flash-attn, 1.7 GB RSS + mmapped weights
  :8083  llama-server  bge-m3 Q8_0 embeddings
  :8084  llama-server  bge-reranker-v2-m3 Q4_K_M, --reranking
  plus  Factorium backend :3000, qdrant :6333, postgres :5432, frontend · system memory free ~17 %
models on disk: ~/.factorium/models/{gemma-4-12b-it-Q5_K_M, embed, rerank, tts (Piper/Coqui ONNX, Slovak + English), whisper ggml-large-v3-turbo + small}
```

The memory budget is shared with everything listed above. Symferium must never load a set of models that would push the machine into swap or kill the desktop. The governor in §5 exists for that.

## 3. Architecture for v1: a runtime manager with engine adapters

Michal's spec asks for "the most optimal way of loading each model, based on research". No single engine is best for every model type on this machine. So Symferium v1 is a supervisor written in Rust that runs the best existing engine for each model, behind one control plane. Its own bare-metal engine is a later track (§8), added as one more adapter when it beats the others on a measurement.

```
symferium (Rust daemon, launchd LaunchAgent)
├── control API  :7700 (configurable)   /api/models /api/load /api/unload /api/settings /api/machine /api/metrics /api/tools
├── web UI       served by the daemon   settings, catalog, load/unload, live memory + tok/s, logs
├── catalog      Hugging Face API → model families, quants, projector (mmproj) files, capability flags
├── downloader   resumable, checksummed, in a models dir chosen by the user
├── governor     machine probe + memory budget, refuses unsafe loads, every limit visible and editable
├── supervisor   spawns, health-checks, restarts and stops each engine process; one model = one port
└── adapters
    ├── llama.cpp (llama-server)   GGUF: chat, vision (mmproj), embeddings, reranking   ← v1 workhorse
    ├── MLX (mlx-lm / Rapid-MLX)   MLX weights, where faster on Apple Silicon           ← v1, measured against llama.cpp
    ├── whisper.cpp server          speech to text
    ├── TTS (Kokoro, Piper ONNX)   text to speech
    └── symferium-engine (own)     later, §8
```

Every model port speaks the OpenAI-compatible API for its type:
- chat models: `/v1/chat/completions`, streaming and tool calls;
- embedders: `/v1/embeddings`;
- rerankers: `/v1/rerank`;
- STT: `/v1/audio/transcriptions`;
- TTS: `/v1/audio/speech`.

All of them also answer `/v1/models` and `/health`. Clients never need to know which engine sits behind a port.

## 4. Model classes, from the spec

The class decides the default engine, the quant and the context. The user can override all three.

| class | examples (spec) | notes |
|---|---|---|
| standard 6–20B | gemma-4 12B, ornith 9B, ministral 8B/14B, granite 8B, gpt-oss 20B, phi-4 14B | download the matching mmproj when the model has vision |
| large (alone) | gemma-4 26B-A4B, qwen 27B, muse glimmer 30B, granite 30B, lfm 24B, ornith 35B, tiel 35B | MoE first; the governor allows them only when little else is loaded |
| small < 5B | gemma E2B/E4B, lfm2.5 / lfm3 (thinking, instruct, VL), llama, phi-4-mini, ministral 3B | |
| helpers | bge-m3, nomic, embeddinggemma, all-MiniLM-L6, jina, LFM decision models | embed, rerank, function, decision |
| audio | whisper, kokoro, audio.cpp | STT and TTS |

Each catalog entry shows these flags, read from the model card, `config.json` and the chat template, never guessed: vision, thinking, tool calling, agentic, coding, context length.

Community quants are offered beside the originals: ggml-org, unsloth, bartowski, mlx-community.

## 5. Governor rules (the "no bricking" requirement)

- **Probe at start:** total and free unified memory, the Metal recommended working-set size, the CPU and GPU core counts.
- **Default budget** comes from the probe, for example "keep at least N GB free for the OS and other apps". It is shown in the UI and editable.
- **Before every load,** estimate weights + KV at the requested context + compute buffers + engine overhead.
  - Refuse if the estimate breaks the budget, and give the numbers in the answer.
  - After the load, compare the estimate with the measured RSS and Metal allocation, and keep the error so later estimates improve.
- **Nothing is hardcoded.** Every limit, default, port, path, timeout and engine flag lives in one TOML config, is visible in the UI, and is saved through the API.
- **Autostart:** a launchd LaunchAgent. The daemon starts at low priority, and the saved model set loads in the background after login without blocking it.

## 6. Agent API (the spec: "tooling made available to any local agent")

- `GET /api/tools` returns a JSON tool manifest: name, description and JSON-schema arguments for list, load, unload, set, metrics and catalog search.
- Agents call the tools over HTTP. An optional MCP shim can come later.
- Each setting is marked `agent_writable: true|false` in the config. Agents may change only those, and always within the governor's limits.

## 7. v1 milestones, each with a gate

```
M0  daemon, config, machine probe, control API, UI shell      cargo test green; /api/machine shows the real numbers
M1  llama.cpp adapter: load/unload a GGUF chat model on its port, supervise, logs
                                                              tiny GGUF (≤150 MB) answers /v1/chat/completions; kill it, it restarts
M2  catalog + downloader (HF search, quants, mmproj, resume, checksum)
                                                              a vision model downloads with its projector and answers with an image
M3  governor + launchd autostart + persisted model set         an over-budget load is refused with numbers; reboot restores the set
M4  adapters: embeddings and rerank (llama.cpp), whisper.cpp, TTS, MLX
                                                              each answers its OpenAI route; MLX vs llama.cpp tok/s on one model, same prompt
M5  Factorium parity: serve gemma-4 12B + mmproj, bge-m3 and the reranker on spare ports beside the current ones
                                                              Factorium's retrieval eval and an Una run give equal results on both; then Michal decides the cutover
M6  agent tooling + the agent_writable policy                 an agent loads, uses and unloads a model through /api/tools only
```

Baselines to beat on this machine, from llama.cpp b10327:
- prefill 85–110 tok/s;
- decode 3.1 tok/s with a long context, 8.3 tok/s short;
- a stable 2,539-token prefix: 31 s cold, 0.3 s when reused in memory.

## 8. Later tracks (not v1)

**symferium-engine.** Our own Rust inference for Gemma-4, added as an adapter. It earns its place on one measurable gap.
- llama.cpp cannot restore a saved prompt cache for Gemma-4 after a restart. The slot file holds no sliding-window checkpoints, so every token is prefilled again.
- An engine that persists the KV cache with its SWA checkpoints would make a restart cost under 1 s instead of about 30.
- Geometry, measured from the GGUF header:
  - 48 blocks, 5 local layers to 1 global;
  - local layers: window 1024, 8 KV heads, head_dim 256, separate K and V;
  - global layers: 1 KV head, head_dim 512, K = V;
  - 262,144 context.
- The blueprints say 4 local KV heads with K = V. That is wrong. Their global-layer math holds: about 2.1 GB of KV at 256K in BF16.
- Treat every other blueprint figure as a hypothesis.

**Symbiosis layer (v2, off by default).** Ghost, Whisper, Nexus and Heron as loosely coupled background processes. They keep a model aware of a long agentic run beyond its context window, using local stores (qdrant, the graph, the memory folder) through adapters.
- v1 prepares for it in two ways only: a request/response middleware hook in the supervisor, and an event bus.
- Nothing in v1 depends on it.

## 9. Who does what

- **The cloud project** (Linux, no Metal, no model weights):
  - writes the code and the docs;
  - runs `cargo test` with mocks and tiny models;
  - adds GitHub Actions CI on `macos-latest` (Apple Silicon runner) for build and small-model smoke tests.
- **The Mac:** the hive's sonnet seat builds and runs the gates that need the real models, launchd or the 24 GB budget, and haiku verifies. Results come back as issues or PR comments.
- **Michal** decides the M5 cutover, anything that touches Factorium's running inference, and the forks in §10.
- **Working rules:**
  - every claimed number comes with the command that produced it;
  - a gate that was not run is reported as not run;
  - one PR per milestone, no secrets, English.

## 10. Open forks

1. MLX in v1 (M4) or later? It needs Python (mlx-lm), or Rapid-MLX as a separate server. Proposal: v1, behind the adapter, measured.
2. UI stack: plain HTML + vanilla JS served by axum (proposal), or a small framework.
3. Control port and the port range for models. Proposal: control 7700, models 7701+, all configurable.
