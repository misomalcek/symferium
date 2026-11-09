# Symbiosis layer — design (v2, off by default)

Goal from the spec: any chat model (small, standard, large) sustains very long agentic runs, works past its context window, and keeps an awareness of what it is and was doing. It does this through loosely coupled background processes — Ghost, Whisper, Nexus, Heron — that use the local memory stores under the hood.

The engine-level ideas in the blueprints (logit masking, residual-stream injection) need our own engine (BRIEF §8). This design works with any engine behind the proxy, today, and leaves the engine-level primitives as a later upgrade path.

## 1. Where it plugs in

```
client ──► model port (proxy) ──► middleware chain ──► engine
                │                      ▲      │
                │ events               │      │ request.completed
                ▼                      │      ▼
         event bus (/api/events, unix socket) ◄── processes publish/subscribe
                │
   ┌────────────┼──────────────┬──────────────┐
   ▼            ▼              ▼              ▼
 Ghost       Whisper         Nexus          Heron
 (record)    (bridge ctx)    (run ledger)   (consolidate, idle/night)
   │            │              │              │
   └──── store adapters: qdrant · graph · memory folder · embed/rerank ports ────┘
```

- A run is identified by the `X-Symferium-Run` header (or the `user` field). Requests without it pass through untouched.
- Symbiosis is switched on globally, per model, and per request (`X-Symferium-Symbiosis: off`).
- Each process is its own OS process (`symferium agent ghost`, …), supervised like a model, restartable, and optional. If one is down, requests still pass; the layer degrades, it never blocks inference.
- The processes use Symferium's own model ports for their thinking: a small model (e.g. LFM2.5 / Gemma E4B) for summaries, the embedder for vectors, the reranker for retrieval, a decision model for gates.

## 2. The four processes

**Ghost — memory and continuity.** Listens to `request.completed`. Writes the raw turn to the memory folder (append-only JSONL per run + a Markdown digest), extracts decisions, failures, facts and open threads with the small model, embeds them and upserts into qdrant. Scores entries with the BUM formula from the blueprints (`use×0.4 + e^(−Δt/30)×0.3 + pinned×10`); decisions and failures never decay.

**Whisper — context bridge (the "beyond the window" part).** A pre-request middleware. When a run's prompt passes a threshold of the model's context (default 70 %), Whisper:
1. keeps the system prompt and the last N turns verbatim;
2. replaces older turns with a rolling summary (small model, cached per run);
3. injects the top-k retrieved memories ("whispers") from Ghost's store, reranked, as one bounded system block;
4. prepends Nexus's awareness card.
The model sees a prompt that always fits, and the full history stays recoverable from the folder.

**Nexus — run ledger and integrity.** Keeps per-run state: goal, plan, done steps, open todos, tool calls made. Renders it as a short awareness card ("you are X, working on Y, done A/B, next C"). Detects loops (the same tool call or near-duplicate output repeated), flags them in the card, and can veto via the decision model. Checks that the stores Ghost and Whisper need are reachable before turning them on for a run.

**Heron — consolidation.** Runs when the machine is idle or on a schedule (default 03:00). Reservoir-samples the qdrant collections, clusters related memories, writes insights and links to the graph store, revives pruned nodes that resonate with recent activity, and does nothing at all when the day's activity is below a threshold ("stillness").

## 3. Store adapters

| store | adapter | used by |
|---|---|---|
| qdrant (:6333) | HTTP; reads `symbiosis.qdrant_read_collections` (e.g. Factorium's), writes `symbiosis.qdrant_write_collection` (default `symferium_ghost`) | Ghost, Whisper, Heron |
| graph | Postgres + Apache AGE (`symbiosis.age_dsn`, `symbiosis.age_graph`), openCypher through `ag_catalog.cypher()` | Heron, Nexus |
| raw memories folder | plain files, `symbiosis.memory_dir` (set by Michal) | Ghost (write), Heron (read) |
| postgres (:5432) | optional, for the run ledger if the folder is not enough | Nexus |

All endpoints, collections, paths, thresholds and schedules live in the same TOML and UI as the rest of Symferium.

## 4. What v1 already provides

- `Middleware` trait in the proxy (`before_request`, `after_response`) with an empty chain.
- Event bus with `model.*`, `request.started`, `request.completed` (prompt and completion token counts, run id, latency).
- Config section `[symbiosis]` with `enabled = false`.
