# Symferium

A standalone local AI runtime for Apple Silicon. It downloads, loads and unloads many models at once (chat, vision, embeddings, rerankers, speech), serves each one on its own localhost port with an OpenAI-compatible API, and keeps every load inside a memory budget measured on the machine. It is managed from a browser UI and from a local API that agents can use.

Start with `docs/PLAN.md` (architecture and decisions), `docs/MODELS.md` (what the catalog offers), then `docs/DEPLOY.md` (install on the Mac and run the gates). The brief and Michal's spec are `docs/BRIEF.md` and `docs/SPEC.md`; the symbiosis layer (v2) is designed in `docs/SYMBIOSIS.md`.

## What runs

```
symferium serve                       one Rust daemon (launchd LaunchAgent)
  :7700  UI + control API             /api/machine /api/models /api/load /api/unload /api/settings /api/catalog/*
                                      /api/downloads /api/bench /api/presets /api/events (SSE) /api/tools
  :7701… one port per loaded model    OpenAI routes of the engine behind it, plus /health and /v1/models
```

Engines are profiles in the config (binary + argument templates), so any flag is visible and editable:

| profile | engine | serves |
|---|---|---|
| `llamacpp` | llama-server | GGUF chat, vision (mmproj), embeddings, rerank |
| `llamacpp-turboquant` | a TurboQuant llama.cpp fork (off) | GGUF chat with turbo3/turbo4 KV |
| `mlx`, `mlx-vlm`, `rapid-mlx` (off) | mlx_lm / mlx_vlm / Rapid-MLX servers | MLX chat and vision |
| `whispercpp` | whisper-server | `/v1/audio/transcriptions` |
| `tts` | bundled sidecar (Kokoro, Piper) | `/v1/audio/speech` |

## Install on the Mac

```sh
scripts/install.sh            # Homebrew llama.cpp + whisper-cpp, uv Python env for MLX/TTS, binary, LaunchAgent
open http://127.0.0.1:7700/
```

Factorium's running inference (`:8080`, `:8083`, `:8084`) is not touched. Its model folder (`~/.factorium/models`) is scanned read-only, so nothing is downloaded twice.

## Use it

```sh
curl -s localhost:7700/api/models | jq '.local[].id'
curl -s localhost:7700/api/load -H 'content-type: application/json' \
     -d '{"model":"gemma-4-12b-it-q5_k_m","ctx":32768,"wait":true}'
curl -s localhost:7701/v1/chat/completions -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"hi"}]}'
```

A load that does not fit is refused with HTTP 409 and every number (weights, KV cache, checkpoints, compute, overhead, headroom). Agents read `GET /api/tools` and call `POST /api/tools/{name}`; they may change only settings marked `agent_writable`.

## CLI

```
symferium serve [--no-restore]      run the daemon
symferium probe                     what the probe sees + the budget
symferium models                    models found on disk
symferium estimate <model> [--ctx]  memory estimate without loading
symferium bench <model> <model>…    load each alone, measure tok/s, print the table (daemon must run)
symferium launchd install|uninstall|print
symferium config [--show]
```

## Develop

```sh
cargo test        # unit + end-to-end tests against a mock engine (no weights, no Metal needed)
cargo clippy --all-targets -- -D warnings
```
