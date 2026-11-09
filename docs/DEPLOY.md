# Deploy and gates — for the Mac

Everything below runs on the Mac mini (M4, 24 GB). The cloud built and tested the code with mocks; every gate that needs Metal, real weights, launchd or the 24 GB budget is run here, and each result is reported with the command that produced it. A gate that was not run is reported as not run.

## 0. Install

```sh
git clone https://github.com/misomalcek/symferium && cd symferium
git checkout claude/project-thread-oin30t   # until the PR is merged
scripts/install.sh --no-launchd             # add the LaunchAgent at M3
symferium serve                             # foreground for the first gates; UI at http://127.0.0.1:7700/
```

Needs: Rust (rustc ≥ 1.85), Homebrew. The install script runs `brew install llama.cpp whisper-cpp uv` and creates the Python env for MLX and TTS. To use the existing llama.cpp b10327 build instead of Homebrew's, set `engines.llamacpp.binary` to its `llama-server` path in Settings.

Config: `~/Library/Application Support/Symferium/symferium.toml` (0600). Logs: `~/Library/Logs/Symferium/`. Hugging Face token for gated repos (Gemma): Settings → `catalog.hf_token`.

## Gates

| gate | command | pass when |
|---|---|---|
| M0 | `symferium probe` | total 24 GB, Metal working set and GPU cores filled in, P/E cores 4/6 |
| M1 + M2 | `scripts/smoke.sh` | prints `SMOKE OK` (downloads a 135M GGUF through the catalog, answers, survives `kill -9`, governor refuses with numbers) |
| M2 vision | UI → Catalog → `gemma-4` (or another vision GGUF) → Download a quant (projector comes along) → Load → send an image to `/v1/chat/completions` | the answer describes the image |
| M3 refuse | Load gemma-4 12B with `ctx` 262144 and f16 KV while bge-m3 + reranker are loaded | 409 with the breakdown; nothing swaps |
| M3 learn | after any load, `curl localhost:7700/api/machine \| jq .corrections` | a correction per engine appears (estimate vs measured `phys_footprint`) |
| M3 reboot | `symferium launchd install`, load a set, reboot | after login + `autostart.delay_s` the set is back, login was not slowed |
| M4 routes | load bge-m3, the reranker, whisper `ggml-large-v3-turbo`, a Piper voice | `/v1/embeddings`, `/v1/rerank`, `/v1/audio/transcriptions`, `/v1/audio/speech` answer on their ports |
| M4 MLX vs llama.cpp | download `mlx-community/<model>-4bit` and the same model as GGUF Q4_K_M, then `symferium bench <gguf-id> <mlx-id>` | the table shows prefill / decode tok/s for both; Models tab shows the faster engine |
| M5 parity | UI → Models → set `factorium-parity` → Load set (ports 7701–7703 beside Factorium's 8080/8083/8084) | Factorium's retrieval eval and an Una run give equal results against both; then Michal decides the cutover |
| M6 agent | an agent uses only `GET /api/tools` + `POST /api/tools/{name}` to load, use and unload a model | it works, and `set_setting` on a non-agent key is refused |

Baselines to beat (llama.cpp b10327, from the brief): prefill 85–110 tok/s; decode 3.1 tok/s long context, 8.3 tok/s short.

## Things to verify on first contact

Already proven on the GitHub Apple Silicon runner (VirtualMac, M2 Pro, 14 GB): Metal working-set probe, real Hugging Face download with sha256, real llama-server load and answer, restart after `kill -9`, governor refusal. Still from documentation only:

- `llama-server` flags in the default profile (`-fa on`, `--ctx-checkpoints`, `--cache-ram`, `--no-webui`): **confirmed** with Homebrew llama.cpp 0.7.0 on the Apple Silicon CI runner (smoke passed). An older build such as b10327 may still reject one; edit `engines.llamacpp.args` then.
- `mlx_lm.server` answers `GET /v1/models` (health path of the `mlx` profile); `mlx_vlm.server` answers `/health`.
- `whisper-server --inference-path /v1/audio/transcriptions` and that `GET /` answers 200 (its health path).
- The Kokoro sidecar needs a `voices*.bin` next to the `kokoro*.onnx` file.
- Memory pressure levels map from `kern.memorystatus_vm_pressure_level` (1 normal, 2 warn, 4 critical).
