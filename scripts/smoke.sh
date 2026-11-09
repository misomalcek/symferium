#!/usr/bin/env bash
# Gates M1 + M2 on a real Mac (or the macOS CI runner):
#   1. download a tiny GGUF through Symferium's own catalog + downloader (sha256 checked)
#   2. load it on llama.cpp, get a chat completion through its port
#   3. kill the engine; the supervisor restarts it on the same port; answer again
#   4. ask for an absurd context and check the governor refuses with numbers
# Uses a throwaway config and models folder; the user's config and models are not touched.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${SMOKE_REPO:-bartowski/SmolLM2-135M-Instruct-GGUF}"
QUANT="${SMOKE_QUANT:-Q8_0}"
SYM="${SYM:-target/release/symferium}"
PORT="${SMOKE_PORT:-7790}"
WORK="$(mktemp -d)"
trap 'curl -s -XPUT "http://127.0.0.1:$PORT/api/shutdown" >/dev/null 2>&1 || true; sleep 1; rm -rf "$WORK"' EXIT

cat > "$WORK/symferium.toml" <<TOML
[server]
control_port = $PORT
model_port_start = $((PORT + 1))
model_port_end = $((PORT + 9))
[paths]
models_dir = "$WORK/models"
extra_model_dirs = []
state_dir = "$WORK/state"
logs_dir = "$WORK/logs"
sidecar_dir = "$WORK/sidecars"
${SMOKE_EXTRA_TOML:-}
TOML

py() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }
api() { curl -sf -H 'content-type: application/json' "$@"; }
step() { echo; echo "==> $*"; }

"$SYM" --config "$WORK/symferium.toml" serve --no-restore > "$WORK/daemon.log" 2>&1 &
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.2; done
B="http://127.0.0.1:$PORT/api"

step "machine"
api "$B/machine" | py 'm=d["machine"]; b=d["budget"]; print(m["model_identifier"], m["cpu_brand"], "total %.1f GB, metal ws %s, headroom %.1f GB" % (m["total_bytes"]/2**30, m["metal_working_set_bytes"], b["headroom_bytes"]/2**30))'

step "download $REPO $QUANT"
api -XPOST "$B/downloads" -d "{\"repo\":\"$REPO\",\"quant\":\"$QUANT\"}"; echo
for _ in $(seq 600); do
  S="$(api "$B/downloads" | py 'print(d[-1]["status"]["state"])')"
  [[ "$S" == done ]] && break
  [[ "$S" == failed ]] && { api "$B/downloads"; exit 1; }
  sleep 1
done
MODEL="$(api "$B/models" | py 'print([m["id"] for m in d["local"] if m["kind"]=="chat"][0])')"
echo "library id: $MODEL"

step "load"
api -XPOST "$B/load" -d "{\"model\":\"$MODEL\",\"alias\":\"smoke\",\"wait\":true,\"persist\":false}" | py 'print(d["url"], d["engine"], "estimate %.2f GB" % (d["estimate"]["total"]/2**30)); print(d["command"])'
MP="$(api "$B/metrics" | py 'print([i for i in d["instances"] if i["id"]=="smoke"][0]["port"])')"
CHAT="{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in three words.\"}],\"max_tokens\":16}"
api "http://127.0.0.1:$MP/v1/chat/completions" -d "$CHAT" | py 'print("answer:", d["choices"][0]["message"]["content"].strip()[:80])'

step "kill the engine; it must come back on the same port"
PID="$(api "$B/metrics" | py 'print([i for i in d["instances"] if i["id"]=="smoke"][0]["status"]["pid"])')"
kill -9 "$PID"
for _ in $(seq 120); do
  R="$(api "$B/metrics" | py 'i=[i for i in d["instances"] if i["id"]=="smoke"][0]; print(i["status"]["state"], i["status"]["restarts"])')"
  [[ "$R" == "ready 1" ]] && break; sleep 0.5
done
echo "state/restarts: $R"; [[ "$R" == "ready 1" ]]
api "http://127.0.0.1:$MP/v1/chat/completions" -d "$CHAT" | py 'print("answer after restart:", d["choices"][0]["message"]["content"].strip()[:80])'

step "governor refuses an over-budget load with numbers"
CODE="$(curl -s -o "$WORK/refused.json" -w '%{http_code}' -H 'content-type: application/json' -XPOST "$B/estimate" -d "{\"model\":\"$MODEL\",\"params\":{\"cache_ram_mb\":\"900000\"}}")"
py 'print("allowed:", d["allowed"]); [print(" -", r) for r in d["reasons"]]' < "$WORK/refused.json"
python3 -c "import json; assert not json.load(open('$WORK/refused.json'))['allowed']"

step "unload"
api -XPOST "$B/unload" -d '{"id":"smoke"}'; echo
echo; echo "SMOKE OK"
