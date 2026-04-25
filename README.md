# CPU Agentic Validation Harness (Llama 3.x 1B)

This project runs a CPU-only agentic workload for up to 36 hours using a small Llama model from Hugging Face.

## What this does

- Downloads a 1B Llama-family model snapshot from Hugging Face
- Runs an agent loop (`plan -> act(tool) -> observe -> answer`)
- Evaluates correctness on synthetic tasks
- Logs latency, tokens/sec, CPU%, memory, and pass rate over time

## Default model

`NousResearch/Hermes-3-Llama-3.2-1B`

This is an open Llama 3.2 1B family model and is usually available without a gated access flow.
If you want the official Meta checkpoint, use `--model-id meta-llama/Llama-3.2-1B-Instruct` and pass `HF_TOKEN`.

## Setup (Linux x86_64)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick smoke test (5 minutes)

```bash
python ./agentic_cpu_runner.py --duration-hours 0.083 --report-every-sec 60 --output-dir ./runs/smoke_linux
```

## 36-hour run

```bash
chmod +x ./run_36h.sh
./run_36h.sh
```

Equivalent direct command:

```bash
python ./agentic_cpu_runner.py --duration-hours 36 --report-every-sec 300 --output-dir ./runs/run_36h
```

## Optional Hugging Face token

```bash
export HF_TOKEN="<your_token>"
```

## Output artifacts

- `summary.json` : aggregate run metrics
- `events.jsonl` : per-iteration events
- `report.csv` : periodic metrics snapshots
- `runner.log` : runtime logs
