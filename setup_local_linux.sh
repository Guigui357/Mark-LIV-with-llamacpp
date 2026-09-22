#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/models"

echo "[1/3] Checking llama.cpp..."
command -v llama >/dev/null || {
  echo "llama não encontrado. Instale/build o llama.cpp primeiro."
  exit 1
}

echo "[2/3] Installing/building whisper.cpp..."
if [ ! -d "$ROOT/whisper.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$ROOT/whisper.cpp"
fi
cmake -S "$ROOT/whisper.cpp" -B "$ROOT/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/whisper.cpp/build" --config Release -j2

echo "[3/3] Downloading tiny Whisper model..."
if [ ! -f "$ROOT/models/ggml-tiny.bin" ]; then
  bash "$ROOT/whisper.cpp/models/download-ggml-model.sh" tiny
  cp "$ROOT/whisper.cpp/models/ggml-tiny.bin" "$ROOT/models/ggml-tiny.bin"
fi

echo
echo "OK. Runtime local:"
echo "  LLM : llama serve"
echo "  STT : whisper.cpp"
echo "  TTS : Piper (se MARK_LIV_PIPER_MODEL estiver definido) ou engine local"
echo
echo "Depois: ./start_local.sh"
