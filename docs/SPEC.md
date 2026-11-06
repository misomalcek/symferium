build end-to-end high-fidelity hybrid symbiotic LLM inference & runtime squeezing maximum potential while staying stable and reliable on apple sillicon (M4, 24GB RAM machine). capable of loading/unloading multiple types of models simultaneously and making them available via localhost (each type of model on its own port). 

for each model type offered to download (from updated hugging face list, filtered based on supported attributes) the most optimal way of loading the model should be selected and implemented based on preliminary research (rust, c++, mlx, gguf, gpu, turbofieldflare, turboquant kv cache, colibri, mtp, vlm, litert, tensorsharp, rapid-mlx, crablm...). 

model types: 
standard: gemma4 12B and any supported model between 6B - 20B parameters (inclusive), like ornith 9B, ministral 8B & 14B, granite 8B, gpt-oss 20B, phi-4 14B (also indicate is model supports vision, thinking, agentic, tool calling, coding and our inference should also download related projector files and load them for full functionality of each model)...
large: MoE and other models fitting with reliable performance when loaded alone, like gemma4 26B, qwen 27B, muse glimmer 30B, granite 30B, lfm 24B, ornith 35B, tiel 35B...
small: under 5B parameters like gemma e2b, e4b, lfm2.5 & 3 (thinking, instruct, vl), llama, phi4-mini, ministral 3B...
embeding, reranker, function, decision: all popular "helper" models for various use cases like bge3-micro, nomic, embeddinggemma , all-minilm-l6, jina, lfm...
audio (stt, tts): kokoro, whisper, audio.cpp ...

comunity models and various quantisations shall be offered

app consits of robust backend, be able to auto start with mac os, loading models set while not blocking the start of the os, it provides a thin frontend (via browser) where user can set and monitor everything the app abd its models needs and does. no setting or limits, values, variables should be hardcoded (all visible to user, all controled against the machine performance to avoid bricking), defaults could be loaded, settings possible to save.

app shall be controlable also via local api endpoint, tooling made available to any local agent to adjust allowed settings, parameters as needed, also to load/unload models (within the allowed memory limits)

in the next version i would also like to consider a sort of symbiotic/awareness layer (option to turn on/off) which will ensure that any of the models (standard, large, small) will sustain really long agentic runs, can work beyond their context window limits and keep a clue (an awareness) of what they are/were doing by utilising loosely coupled processes (more like processes, kind of autonomous backend agents): ghost, whisper, nexus, heron, and we should figure out how will it connect to available local memory storage and tooling (e.g our qdrant database, graph database, raw memories folder) for it's symbiotic operations under the hood. this will be highly custom for this machine and for my system and apps that i use with AI on local)