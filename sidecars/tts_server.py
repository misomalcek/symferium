#!/usr/bin/env python3
"""Symferium TTS sidecar: OpenAI-compatible /v1/audio/speech for Kokoro (kokoro-onnx) and Piper voices.

Started by Symferium with:  python tts_server.py --model <file.onnx> --host 127.0.0.1 --port <port>
Standard library HTTP server; the only third-party packages are the engines themselves
(kokoro-onnx, piper-tts, soundfile, numpy), installed by scripts/install-engines.sh.
"""
import argparse
import io
import json
import os
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--voices", default=None, help="Kokoro voices file (default: voices*.bin next to the model)")
args = ap.parse_args()

MODEL = os.path.abspath(args.model)
NAME = os.path.splitext(os.path.basename(MODEL))[0]
IS_PIPER = os.path.exists(MODEL + ".json")
lock = threading.Lock()


def load_engine():
    if IS_PIPER:
        from piper import PiperVoice  # piper-tts
        return PiperVoice.load(MODEL)
    from kokoro_onnx import Kokoro
    voices = args.voices
    if not voices:
        d = os.path.dirname(MODEL)
        cands = sorted(f for f in os.listdir(d) if f.startswith("voices") and f.endswith((".bin", ".npz", ".json")))
        if not cands:
            raise SystemExit("no Kokoro voices file next to the model; pass --voices")
        voices = os.path.join(d, cands[0])
    return Kokoro(MODEL, voices)


ENGINE = load_engine()


def to_wav(samples, rate):
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def synth(text, voice, speed):
    with lock:
        if IS_PIPER:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                if hasattr(ENGINE, "synthesize_wav"):
                    ENGINE.synthesize_wav(text, w)
                else:
                    ENGINE.synthesize(text, w, length_scale=1.0 / max(speed, 0.1))
            return buf.getvalue()
        lang = "en-us"
        samples, rate = ENGINE.create(text, voice=voice or "af_heart", speed=speed, lang=lang)
        return to_wav(samples, rate)


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok", "engine": "piper" if IS_PIPER else "kokoro"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("content-length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            text = req.get("input", "")
            if not text:
                return self._json(400, {"error": {"message": "input is required"}})
            fmt = req.get("response_format", "wav")
            if fmt not in ("wav", "pcm"):
                return self._json(400, {"error": {"message": "response_format must be wav or pcm"}})
            audio = synth(text, req.get("voice"), float(req.get("speed", 1.0)))
            if fmt == "pcm":
                audio = audio[44:]
            self.send_response(200)
            self.send_header("content-type", "audio/wav" if fmt == "wav" else "audio/pcm")
            self.send_header("content-length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
        except Exception as e:  # report, keep serving
            self._json(500, {"error": {"message": str(e)}})

    def log_message(self, fmt, *a):
        print("tts:", fmt % a, flush=True)


print(f"tts sidecar {NAME} ({'piper' if IS_PIPER else 'kokoro'}) on {args.host}:{args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
