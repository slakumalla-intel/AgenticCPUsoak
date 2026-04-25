#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-NousResearch/Hermes-3-Llama-3.2-1B}"
OUTPUT_DIR="${2:-./runs/run_36h}"
REPORT_EVERY_SEC="${3:-300}"

if [[ ! -x "./.venv/bin/python" ]]; then
  echo "Virtual environment not found. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

./.venv/bin/python ./agentic_cpu_runner.py \
  --model-id "${MODEL_ID}" \
  --duration-hours 36 \
  --report-every-sec "${REPORT_EVERY_SEC}" \
  --output-dir "${OUTPUT_DIR}"
