import argparse
import csv
import json
import logging
import math
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a reliable planning agent.
You must strictly return valid JSON with keys:
- plan: list[str]
- action: either \"CALC:<expression>\" or \"NONE\"
- final_answer: string
Do not include markdown or extra text.
"""

ACTION_RE = re.compile(r"^CALC:(.+)$")
NUMBER_RE = re.compile(r"-?\d+")


@dataclass
class Task:
    task_id: str
    prompt: str
    expected: str


@dataclass
class IterationResult:
    task_id: str
    ok: bool
    expected: str
    predicted: str
    latency_sec: float
    generated_tokens: int
    tokens_per_sec: float
    cpu_percent: float
    mem_rss_mb: float


class CalcTool:
    ALLOWED = set("0123456789+-*/(). ")

    @staticmethod
    def run(expression: str) -> str:
        safe = expression.strip()
        if not safe or any(ch not in CalcTool.ALLOWED for ch in safe):
            return "ERROR: invalid expression"
        try:
            value = eval(safe, {"__builtins__": {}}, {})  # noqa: S307 - restricted by allow-list
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            return str(value)
        except Exception as exc:
            return f"ERROR: {exc}"


class AgenticValidator:
    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        max_new_tokens: int,
        report_every_sec: int,
        hf_token: Optional[str],
        seed: int,
    ) -> None:
        self.model_id = model_id
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.report_every_sec = report_every_sec
        self.hf_token = hf_token
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.report_path = self.output_dir / "report.csv"
        self.summary_path = self.output_dir / "summary.json"
        self.log_path = self.output_dir / "runner.log"

        self.logger = self._build_logger()
        self.process = psutil.Process(os.getpid())
        random.seed(self.seed)

        self.logger.info("Downloading model snapshot for %s", self.model_id)
        snapshot_download(
            repo_id=self.model_id,
            token=self.hf_token,
            local_dir=str(self.output_dir / "hf_cache"),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        self.logger.info("Loading tokenizer/model on CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to("cpu")
        self.model.eval()

        self._initialize_report_csv()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("agentic_runner")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _initialize_report_csv(self) -> None:
        with self.report_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp",
                    "iterations",
                    "pass_rate",
                    "avg_latency_sec",
                    "avg_tokens_per_sec",
                    "avg_cpu_percent",
                    "avg_mem_rss_mb",
                ],
            )
            writer.writeheader()

    def _task_stream(self) -> Task:
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        c = random.randint(2, 9)
        mode = random.choice(["two_step", "difference", "combo"])

        if mode == "two_step":
            prompt = (
                f"Compute this carefully: first add {a} and {b}, then multiply the result by {c}. "
                "Return only the final number."
            )
            expected = str((a + b) * c)
        elif mode == "difference":
            prompt = (
                f"Find the absolute difference between {a * c} and {b * c}. "
                "Return only the final number."
            )
            expected = str(abs((a * c) - (b * c)))
        else:
            prompt = (
                f"Calculate ({a} * {c}) + ({b} // {c}) and return only the final number."
            )
            expected = str((a * c) + (b // c))

        return Task(task_id=f"task_{int(time.time() * 1000)}_{random.randint(1000,9999)}", prompt=prompt, expected=expected)

    def _build_prompt(self, task: Task, tool_observation: Optional[str]) -> str:
        base = f"System:\n{SYSTEM_PROMPT}\nUser:\n{task.prompt}\n"
        if tool_observation is not None:
            base += f"Tool observation: {tool_observation}\n"
        return base

    def _generate(self, prompt: str) -> Tuple[str, int, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cpu")
        attn_mask = inputs["attention_mask"].to("cpu")

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end = time.time()

        generated = outputs[0][input_ids.shape[-1] :]
        gen_tokens = int(generated.shape[-1])
        latency = end - start
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip(), gen_tokens, latency

    @staticmethod
    def _parse_json(text: str) -> Dict[str, object]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        blob = text[start : end + 1]
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_answer(value: str) -> str:
        matches = NUMBER_RE.findall(value)
        if not matches:
            return value.strip()
        return matches[-1]

    def run_iteration(self) -> IterationResult:
        task = self._task_stream()

        step1_prompt = self._build_prompt(task, tool_observation=None)
        step1_text, step1_tokens, step1_latency = self._generate(step1_prompt)
        step1_json = self._parse_json(step1_text)
        action = str(step1_json.get("action", "NONE"))

        tool_output: Optional[str] = None
        if action != "NONE":
            match = ACTION_RE.match(action)
            if match:
                tool_output = CalcTool.run(match.group(1))
            else:
                tool_output = "ERROR: malformed action"

        step2_tokens = 0
        step2_latency = 0.0
        final_answer = str(step1_json.get("final_answer", "")).strip()

        if tool_output is not None:
            step2_prompt = self._build_prompt(task, tool_observation=tool_output)
            step2_text, step2_tokens, step2_latency = self._generate(step2_prompt)
            step2_json = self._parse_json(step2_text)
            final_answer = str(step2_json.get("final_answer", final_answer)).strip()

        predicted = self._extract_answer(final_answer)
        ok = predicted == task.expected
        total_latency = step1_latency + step2_latency
        total_tokens = step1_tokens + step2_tokens
        tps = 0.0 if total_latency == 0 else total_tokens / total_latency

        cpu_pct = psutil.cpu_percent(interval=None)
        mem_rss_mb = self.process.memory_info().rss / (1024 * 1024)

        return IterationResult(
            task_id=task.task_id,
            ok=ok,
            expected=task.expected,
            predicted=predicted,
            latency_sec=total_latency,
            generated_tokens=total_tokens,
            tokens_per_sec=tps,
            cpu_percent=cpu_pct,
            mem_rss_mb=mem_rss_mb,
        )

    def _write_event(self, result: IterationResult) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": result.task_id,
            "ok": result.ok,
            "expected": result.expected,
            "predicted": result.predicted,
            "latency_sec": result.latency_sec,
            "generated_tokens": result.generated_tokens,
            "tokens_per_sec": result.tokens_per_sec,
            "cpu_percent": result.cpu_percent,
            "mem_rss_mb": result.mem_rss_mb,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def _write_periodic_report(self, results: List[IterationResult]) -> None:
        if not results:
            return

        pass_rate = sum(1 for r in results if r.ok) / len(results)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": len(results),
            "pass_rate": f"{pass_rate:.6f}",
            "avg_latency_sec": f"{statistics.fmean(r.latency_sec for r in results):.6f}",
            "avg_tokens_per_sec": f"{statistics.fmean(r.tokens_per_sec for r in results):.6f}",
            "avg_cpu_percent": f"{statistics.fmean(r.cpu_percent for r in results):.6f}",
            "avg_mem_rss_mb": f"{statistics.fmean(r.mem_rss_mb for r in results):.6f}",
        }
        with self.report_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _write_summary(self, results: List[IterationResult], wall_clock_sec: float) -> None:
        if not results:
            payload = {
                "status": "no_results",
                "wall_clock_sec": wall_clock_sec,
                "model_id": self.model_id,
            }
        else:
            pass_rate = sum(1 for r in results if r.ok) / len(results)
            payload = {
                "status": "ok",
                "model_id": self.model_id,
                "wall_clock_sec": wall_clock_sec,
                "iterations": len(results),
                "pass_rate": pass_rate,
                "avg_latency_sec": statistics.fmean(r.latency_sec for r in results),
                "p95_latency_sec": _percentile([r.latency_sec for r in results], 95),
                "avg_tokens_per_sec": statistics.fmean(r.tokens_per_sec for r in results),
                "avg_cpu_percent": statistics.fmean(r.cpu_percent for r in results),
                "avg_mem_rss_mb": statistics.fmean(r.mem_rss_mb for r in results),
            }
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def run(self, duration_hours: float) -> None:
        end_ts = time.time() + (duration_hours * 3600)
        last_report = 0.0
        results: List[IterationResult] = []
        started = time.time()

        self.logger.info("Starting validation run for %.3f hours", duration_hours)

        while time.time() < end_ts:
            result = self.run_iteration()
            results.append(result)
            self._write_event(result)

            elapsed = time.time() - started
            if elapsed - last_report >= self.report_every_sec:
                last_report = elapsed
                self._write_periodic_report(results)
                self.logger.info(
                    "iters=%d pass_rate=%.3f avg_latency=%.3fs avg_tps=%.3f",
                    len(results),
                    sum(1 for r in results if r.ok) / len(results),
                    statistics.fmean(r.latency_sec for r in results),
                    statistics.fmean(r.tokens_per_sec for r in results),
                )

        wall_clock = time.time() - started
        self._write_periodic_report(results)
        self._write_summary(results, wall_clock)
        self.logger.info("Run completed. Iterations=%d, wall_clock_sec=%.2f", len(results), wall_clock)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only agentic validation runner")
    parser.add_argument("--model-id", type=str, default="NousResearch/Hermes-3-Llama-3.2-1B")
    parser.add_argument("--duration-hours", type=float, default=36.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report-every-sec", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "run_latest")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.hf_token if args.hf_token else None

    runner = AgenticValidator(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        report_every_sec=args.report_every_sec,
        hf_token=token,
        seed=args.seed,
    )
    runner.run(args.duration_hours)


if __name__ == "__main__":
    main()
