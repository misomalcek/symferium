#!/usr/bin/env bash
# Install Symferium on this Mac: engines, Python env for MLX/TTS, the binary, and (optionally) the
# LaunchAgent. Touches nothing of Factorium: its launchd jobs and ports 8080-8084 keep running.
#
#   scripts/install.sh                 # everything
#   scripts/install.sh --no-launchd    # do not start at login
#   scripts/install.sh --no-engines    # skip Homebrew/uv steps (engines already installed)
set -euo pipefail
cd "$(dirname "$0")/.."

LAUNCHD=1; ENGINES=1
for a in "$@"; do
  case "$a" in
    --no-launchd) LAUNCHD=0 ;;
    --no-engines) ENGINES=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || { echo "Symferium targets Apple Silicon macOS"; exit 1; }

BIN_DIR="${SYMFERIUM_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$BIN_DIR"

echo "==> building symferium (release)"
cargo build --release --locked
install -m 0755 target/release/symferium "$BIN_DIR/symferium"
SYM="$BIN_DIR/symferium"

echo "==> config (created from the machine probe on first run)"
CONFIG="$("$SYM" config)"
echo "    $CONFIG"
PYENV="$("$SYM" config --show | awk -F' = ' '/^python_env/ {gsub(/"/,"",$2); print $2; exit}')"
PYENV="${PYENV/#\~/$HOME}"

if [[ $ENGINES == 1 ]]; then
  command -v brew >/dev/null || { echo "Homebrew is required for llama.cpp and whisper.cpp (https://brew.sh)"; exit 1; }
  echo "==> engines from Homebrew: llama.cpp, whisper-cpp, uv"
  brew install llama.cpp whisper-cpp uv
  echo "==> Python env for MLX and TTS: $PYENV"
  uv venv --python 3.12 "$PYENV"
  uv pip install --python "$PYENV/bin/python" --upgrade mlx-lm mlx-vlm kokoro-onnx piper-tts soundfile numpy
  echo "    optional, faster MLX server:  uv pip install --python $PYENV/bin/python rapid-mlx"
  echo "    then enable it:               engines.rapid-mlx.enabled = true (UI → Settings)"
fi

echo "==> probe"
"$SYM" probe

if [[ $LAUNCHD == 1 ]]; then
  echo "==> LaunchAgent (starts at login; models restore in the background after autostart.delay_s)"
  "$SYM" launchd install
fi

PORT="$("$SYM" config --show | awk -F' = ' '/^control_port/ {print $2; exit}')"
echo
echo "Done. UI and control API: http://127.0.0.1:${PORT:-7700}/"
[[ $LAUNCHD == 1 ]] || echo "Start it by hand with: $SYM serve"
