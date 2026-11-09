# Which models Symferium offers, and how the catalog filters Hugging Face

The catalog shows only models that an engine profile on this Mac can load and that can fit in its memory. Every rule below is a setting under `catalog.*` (UI → Settings), so nothing is fixed in code.

## Formats that load

| repo format | how it is recognised | engine | notes |
|---|---|---|---|
| GGUF | `.gguf` files or the `gguf` tag | llama.cpp | chat, vision (+ `mmproj-*.gguf`, downloaded with the model), embeddings, rerankers, decision models |
| MLX | `mlx-community/*` or the `mlx` tag | mlx-lm / mlx-vlm (Rapid-MLX optional) | quantised 4/8-bit; vision when `config.json` has `vision_config` |
| safetensors (originals) | anything else with `config.json` + `.safetensors` | mlx-lm | full-precision weights: practical for small models only; the fit check hides the rest |
| whisper.cpp | `ggml-*.bin` | whisper-server | speech to text |
| ONNX voices | `kokoro*.onnx` + `voices*.bin`, or Piper `*.onnx` + `.onnx.json` | TTS sidecar | text to speech |

`catalog.formats` lists the formats offered. Remove `safetensors` to see only quantised builds.

## Always filtered out

- Pipelines with no engine here (`catalog.exclude_pipeline_tags`): text-to-image, image-to-image, video generation, segmentation, detection, depth, 3D, RL, robotics.
- Repos with fewer than `catalog.min_downloads` downloads (default 50): abandoned or test uploads.
- Quants that can never fit this machine (`catalog.hide_too_big`): estimate > min(total − `governor.min_free_gb`, Metal working set). On the 24 GB M4 that is roughly a 16 GB ceiling.
- Models above `catalog.large_max_b` parameters (default 40 B).

## Classes (from parameter count and pipeline tag)

| class | rule (defaults) | spec examples |
|---|---|---|
| small | < `catalog.small_max_b` (5 B) | Gemma E2B/E4B, LFM2.5/3, Phi-4-mini, Ministral 3B |
| standard | `standard_min_b`–`standard_max_b` (5–22 B, so gpt-oss-20b's 21 B counts) | Gemma-4 12B, Ornith 9B, Ministral 8B/14B, Granite 8B, Phi-4 14B, gpt-oss 20B |
| large | up to `large_max_b` (40 B), loads alone | Gemma-4 26B-A4B, Qwen 27B, Granite 30B, LFM 24B, Ornith/Tiel 35B |
| embed | pipeline `feature-extraction` / `sentence-similarity`, or "embed" in the name | bge-m3, nomic-embed, embeddinggemma, all-MiniLM-L6, jina-embeddings |
| rerank | pipeline `text-ranking`, or "rerank" in the name | bge-reranker-v2-m3, jina-reranker |
| decision | "decision" / "function-call" in the name | LFM decision models |
| stt / tts | ASR / TTS pipelines | whisper large-v3-turbo / small; Kokoro, Piper |

## Capability flags (each shown with its source)

- **vision**: pipeline `image-text-to-text`, a `vision` tag, `vision_config` in `config.json`, or a projector file in the repo.
- **tools**: the chat template handles `tools` / `tool_call`, or a `function-calling` / `tool-use` tag.
- **thinking**: the template has `enable_thinking`, `<think>` or `reasoning_content`, or a `reasoning` tag.
- **agentic**: an `agent` / `agentic` tag only (never inferred).
- **coding**: a `code` tag, or "coder" / "devstral" / "codestral" in the name.
- **context**: GGUF metadata or `max_position_embeddings`.

## Quant choice

`catalog.quant_preference` (default Q4_K_M, UD-Q4_K_XL, Q5_K_M, Q4_K_S, IQ4_XS, Q6_K, Q8_0, MXFP4): the first one that fits is marked *recommended*. Each quant shows its estimated memory at the class's default context and whether it fits now, needs another model unloaded first, or is too big.

## Publishers searched beside the originals

`catalog.authors`: ggml-org, unsloth, bartowski, mlx-community, lmstudio-community. The search box can also target any other author.

## Gated models

Gemma and some Llama repos are gated. Accept the licence on huggingface.co, then put a read token in `catalog.hf_token`. It is stored in the config file with 0600 permissions and is masked in the UI and API.
