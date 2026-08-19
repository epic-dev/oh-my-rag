#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-mlx-community/Qwen3-1.7B-4bit}"
PORT="${LLM_PORT:-11434}"

echo "Starting mlx-lm server: model=$MODEL port=$PORT"
mlx_lm.server --model "$MODEL" --port "$PORT"
