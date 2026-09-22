#!/usr/bin/env bash
set -e
MODEL="${MARK_LIV_MODEL:-Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M}"
echo "[Mark-LIV] Starting local llama.cpp server..."
exec llama serve -hf "$MODEL" --jinja --alias mark-liv --host 127.0.0.1 --port 8080 -c 2048 -np 1
