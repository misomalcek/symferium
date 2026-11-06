# Files to upload to the Claude Project with the brief

`docs/BRIEF.md`, `docs/SPEC.md` (Michal's spec) and `docs/REPOS.md` (the public inference repositories) are already in this repo. The project reads them from the repo, so they need no upload.

Upload only these. Each one is here for a reason; everything else in the folders was left out on purpose.

## Blueprints (`~/content/inference-blueprints`) — hypotheses, not measurements

| file | why |
|---|---|
| `implementation/03_GEMMA4_KV_MATH_AND_BENCHMARKS.md` | the KV math. BRIEF §8 corrects the local-layer figures |
| `implementation/02_CODEMASTER_SPEC_GROK.md` | the candle + axum skeleton for the later engine track |
| `implementation/04_SOURCE_REGISTRY_AND_CITATIONS.md` | which engines and papers the design leaned on |
| `engine-autonomy/engine-level-autonomy-1.md` | the clearest map of Weaver, Nexus, Ghost and Heron for the v2 symbiosis layer |
| `03/Step0-Rust Holonic Architecture.md` | the holonic module idea behind loosely coupled processes |
| `03/Step4-Heron Dream Engine.md` | Heron and the integration of steps 1–4 |

## Sources (`~/sources/inference`) — the evidence

| file | why |
|---|---|
| `TheTomturboquant_plus.md` | TurboQuant KV compression, validated on a MacBook |
| `42-kv-cache - add SWA support by ggerganov · Pull Request #13194 · ggml-org-llama.cpp.md` | how llama.cpp handles SWA caches, which is the restore gap in BRIEF §8 |
| `My biggest Issue with the Gemma-4 Models is the Massive KV Cache!!.md` | real-world Gemma-4 memory behaviour |
| `56-08-Gemma 4 Mac RAM.md.md` | Gemma-4 on Mac memory |
| `01-Rust Production Patterns.md` | Rust patterns we want in the daemon |

## Left out, and why

- `architecture/`, `01/`, `02/` (earlier NotebookLM designs): they repeat each other and carry the uncorrected KV numbers.
- `metal-shaders.md`, `turboquant.metal`, `ghost_inject.metal`, `cargo-workspace.md`: engine-track detail. They are worth it only when that track starts, and they live in the hive.
- The rest of `~/sources/inference`: mostly Codex chapters and manifestos, too broad for a build brief.
