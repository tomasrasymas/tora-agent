#!/usr/bin/env python3
"""
Benchmark runner for llama.cpp (OpenAI-compatible).

Flow:
  1. Read currently-loaded model from /v1/models.
  2. For each test in benchmarks.TESTS, do `warmup` discarded calls + `runs`
     measured calls. Optionally score the response with Claude Sonnet 4.6.
  3. Save results JSON to results/<model>-<timestamp>.json.

Run with:
  uv run benchmarks/run_benchmarks.py --runs 5 --warmup 2
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from benchmarks import CATEGORY_WEIGHTS, JUDGE_SYSTEM, TESTS, build_judge_prompt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
ENV_FILE = SCRIPT_DIR.parent / ".env"
JUDGE_MODEL = "claude-sonnet-4-6"


# ─── ANSI palette + print helpers ─────────────────────────────────────
# R/G/Y/B/C/W = red/green/yellow/blue/cyan/white. DIM = dimmed gray.
# RESET turns formatting off.

R, G, Y, B, C, W = (
    "\033[91m",
    "\033[92m",
    "\033[93m",
    "\033[94m",
    "\033[96m",
    "\033[97m",
)
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _hdr(t):
    """Top-level header with rule lines above and below."""
    print(f"\n{B}{'─' * 64}{RESET}\n{BOLD}{W}  {t}{RESET}\n{B}{'─' * 64}{RESET}")


def _sec(t):
    """Section heading — cyan arrow."""
    print(f"\n{C}▸ {t}{RESET}")


def _ok(t):
    """Green check."""
    print(f"  {G}✓{RESET}  {t}")


def _fail(t):
    """Red cross."""
    print(f"  {R}✗{RESET}  {t}")


def _info(t):
    """Dimmed info line — descriptions, paths, low-importance detail."""
    print(f"  {DIM}{t}{RESET}")


def _warn(t):
    """Yellow warning."""
    print(f"  {Y}⚠{RESET}  {t}")


def _met(label, value):
    """Metric row — yellow label, white value."""
    print(f"    {Y}{label:<24}{RESET} {W}{value}{RESET}")


def _print_prompt(test: dict) -> None:
    """Show the chat messages (and tool names) so it's obvious what we're asking."""
    for msg in test["messages"]:
        role, content = msg["role"], msg["content"]
        # Cap per-message length so very long prompts don't drown the console.
        if len(content) > 400:
            content = content[:400] + "…"
        if "\n" in content:
            print(f"  {DIM}[{role}]{RESET}")
            for line in content.split("\n"):
                print(f"    {DIM}{line}{RESET}")
        else:
            print(f"  {DIM}[{role}] {content}{RESET}")
    if test.get("tools"):
        names = [t["function"]["name"] for t in test["tools"]]
        print(f"  {DIM}tools: {', '.join(names)}{RESET}")


# ─── env ──────────────────────────────────────────────────────────────


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# ─── llama.cpp calls (via OpenAI SDK — llama.cpp speaks the same protocol) ──

_openai: OpenAI | None = None


def _get_openai_client(base_url: str) -> OpenAI:
    """Lazy OpenAI client pointed at the llama.cpp server."""
    global _openai
    if _openai is None:
        # OpenAI SDK expects base_url to already include /v1
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        # llama.cpp ignores the key, but the SDK requires a non-empty string.
        _openai = OpenAI(base_url=url, api_key="not-needed", timeout=600.0)
    return _openai


def get_loaded_model(base_url: str) -> str:
    """Return the model id currently loaded in the llama.cpp server."""
    try:
        models = _get_openai_client(base_url).models.list().data
    except Exception as e:
        sys.exit(f"{R}Cannot reach llama.cpp at {base_url}: {e}{RESET}")
    if not models:
        sys.exit(
            f"{R}No model loaded on {base_url}. Load one in llama.cpp first.{RESET}"
        )
    return models[0].id


def call_model(base_url: str, model: str, test: dict) -> dict:
    """Send one chat completion request. Returns text, tool_calls, timing, tokens."""
    kwargs = {
        "model": model,
        "messages": test["messages"],
        "temperature": 0.0,
        "stream": False,
        "max_tokens": 2048,
        # llama.cpp-specific: disable KV cache reuse so each run measures cold
        # prompt processing, not a cache hit from the previous identical request.
        "extra_body": {"cache_prompt": False},
    }
    if test.get("tools"):
        kwargs["tools"] = test["tools"]
        kwargs["tool_choice"] = "auto"
    if test.get("response_format"):
        kwargs["response_format"] = test["response_format"]

    t0 = time.perf_counter()
    resp = _get_openai_client(base_url).chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    message = resp.choices[0].message

    tool_calls = []
    for tc in message.tool_calls or []:
        raw_args = tc.function.arguments
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        tool_calls.append(
            {"id": tc.id, "function": {"name": tc.function.name, "arguments": args}}
        )

    ct = resp.usage.completion_tokens if resp.usage else 0
    return {
        "text": message.content or "",
        "tool_calls": tool_calls,
        "elapsed": elapsed,
        "completion_tokens": ct,
        "tokens_per_sec": ct / elapsed if elapsed > 0 and ct else 0,
    }


# ─── LLM judge (Claude Sonnet 4.6) ────────────────────────────────────

_anthropic = None


def call_judge(judge_cfg: dict, text: str, tool_calls: list, api_key: str) -> dict:
    """Score the response on a 0.0–1.0 scale. Returns score, reason, passed."""
    global _anthropic
    if _anthropic is None:
        import anthropic

        _anthropic = anthropic.Anthropic(api_key=api_key)

    threshold = judge_cfg.get("threshold", 0.7)
    try:
        msg = _anthropic.messages.create(
            model=JUDGE_MODEL,
            max_tokens=120,
            system=JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": build_judge_prompt(
                        judge_cfg["prompt"], text, tool_calls
                    ),
                }
            ],
        )
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", msg.content[0].text.strip())
        parsed = json.loads(raw)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        return {
            "score": round(score, 3),
            "reason": parsed.get("reason", ""),
            "passed": score >= threshold,
        }
    except Exception as e:
        return {"score": 0.0, "reason": f"judge error: {e}", "passed": False}


# ─── stats ────────────────────────────────────────────────────────────


def stats(values: list[float]) -> dict:
    """Mean / median / stdev / p95 / min / max for a list of numbers."""
    if not values:
        return {}
    sv = sorted(values)
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0,
        "p95": round(sv[max(0, int(len(values) * 0.95) - 1)], 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


# ─── one test ─────────────────────────────────────────────────────────


def run_test(
    base_url: str,
    model: str,
    name: str,
    test: dict,
    runs: int,
    warmup: int,
    api_key: str | None,
) -> dict:
    """Run warmup + measured calls for a single test. Returns aggregated result."""
    eval_fn = test.get("eval")
    judge_cfg = test.get("judge")
    use_judge = judge_cfg is not None and api_key is not None

    latencies, tps, passes, judge_scores, judge_results = [], [], [], [], []
    last = None

    for i in range(warmup + runs):
        is_warmup = i < warmup
        label = (
            f"warmup {i + 1}/{warmup}" if is_warmup else f"run {i - warmup + 1}/{runs}"
        )

        try:
            result = call_model(base_url, model, test)
        except Exception as e:
            _fail(f"[{label}] error: {e}")
            if not is_warmup:
                passes.append(False)
            continue

        last = result
        if is_warmup:
            print(
                f"    {DIM}· [{label}] {result['elapsed']:.3f}s  "
                f"{result['tokens_per_sec']:.0f} tok/s  (discarded){RESET}"
            )
            continue

        latencies.append(result["elapsed"])
        if result["tokens_per_sec"] > 0:
            tps.append(result["tokens_per_sec"])

        det_ok = bool(eval_fn(result)) if eval_fn else True
        judge_out = None
        if use_judge:
            judge_out = call_judge(
                judge_cfg, result["text"], result["tool_calls"], api_key
            )
            judge_scores.append(judge_out["score"])
            judge_results.append(judge_out)

        passed = det_ok and (judge_out["passed"] if judge_out else True)
        passes.append(passed)

        sym = f"{G}✓{RESET}" if passed else f"{R}✗{RESET}"
        det_tag = (
            f"  det:{G if det_ok else R}{'✓' if det_ok else '✗'}{RESET}"
            if eval_fn
            else ""
        )
        judge_tag = ""
        if judge_out:
            jc = G if judge_out["passed"] else R
            judge_tag = f"  judge:{jc}{judge_out['score']:.2f}{RESET}"
        print(
            f"    {sym} [{label}] "
            f"{result['elapsed']:.3f}s  {result['tokens_per_sec']:.0f} tok/s"
            f"{det_tag}{judge_tag}"
        )
        if not passed and result["text"]:
            _info(f"output: {result['text'][:300]}")
        if not passed and result["tool_calls"]:
            _info(f"tool_calls: {json.dumps(result['tool_calls'])[:300]}")

    accuracy = round(sum(passes) / len(passes) * 100, 1) if passes else 0.0
    return {
        "test": name,
        "category": test["category"],
        "description": test["description"],
        "accuracy": accuracy,
        "latency": stats(latencies),
        "tokens_per_sec": stats(tps),
        "judge_score": stats(judge_scores) if judge_scores else None,
        "judge_results": judge_results,
        "last_text": (last or {}).get("text", "")[:400],
        "last_tool_calls": (last or {}).get("tool_calls"),
    }


# ─── composite score ──────────────────────────────────────────────────


def composite_score(results: dict) -> dict:
    """Weighted average accuracy across categories (weights from benchmarks.py)."""
    by_cat: dict[str, list[float]] = {}
    for r in results.values():
        by_cat.setdefault(r["category"], []).append(r["accuracy"])

    weighted_sum = total_weight = 0.0
    breakdown = {}
    for cat, accs in by_cat.items():
        avg = statistics.mean(accs)
        weight = CATEGORY_WEIGHTS.get(cat, 1.0)
        weighted_sum += avg * weight
        total_weight += weight
        breakdown[cat] = {"avg_accuracy": round(avg, 1), "weight": weight}

    return {
        "score": round(weighted_sum / total_weight, 1) if total_weight else 0,
        "breakdown": breakdown,
    }


# ─── output path ──────────────────────────────────────────────────────


def output_path(model: str, timestamp: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")
    safe_ts = timestamp.replace(":", "-")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{safe_model}-{safe_ts}.json"


# ─── main ─────────────────────────────────────────────────────────────


def main():
    load_dotenv(ENV_FILE)

    parser = argparse.ArgumentParser(
        description="LLM benchmark — llama.cpp / OpenAI-compatible."
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8080", help="llama.cpp server URL"
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per test")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (discarded)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = get_loaded_model(args.base_url)
    timestamp = datetime.now().isoformat(timespec="seconds")

    _hdr("LLM Benchmark — llama.cpp / OpenAI-compatible")
    _met("Model", model)
    _met("Server", args.base_url)
    _met("Tests", str(len(TESTS)))
    _met("Runs", f"{args.runs} measured, {args.warmup} warmup")
    if api_key:
        _met("Judge", JUDGE_MODEL)
    else:
        _warn("No ANTHROPIC_API_KEY in .env — judge evals skipped.")

    # Run all tests in random order to spread out any environmental drift.
    test_names = list(TESTS.keys())
    random.shuffle(test_names)

    results = {}
    total = len(test_names)
    for idx, name in enumerate(test_names, start=1):
        test = TESTS[name]
        _sec(f"({idx}/{total}) [{test['category']}] {name}")
        _info(test["description"])
        _print_prompt(test)
        results[name] = run_test(
            args.base_url, model, name, test, args.runs, args.warmup, api_key
        )
        r = results[name]
        ac_color = G if r["accuracy"] >= 80 else (Y if r["accuracy"] >= 50 else R)
        _met("Accuracy", f"{ac_color}{r['accuracy']}%{RESET}")
        if r["latency"].get("mean"):
            _met(
                "Latency mean",
                f"{r['latency']['mean']:.3f}s  (p95 {r['latency'].get('p95', 0):.3f}s)",
            )
        if r["tokens_per_sec"].get("mean"):
            _met("Tokens/sec", f"{r['tokens_per_sec']['mean']:.1f}")

    composite = composite_score(results)
    _hdr("Composite Score")
    score_color = (
        G if composite["score"] >= 80 else (Y if composite["score"] >= 50 else R)
    )
    _met("Score", f"{score_color}{composite['score']:.1f}%{RESET}")
    for cat, bd in composite["breakdown"].items():
        _met(f"  {cat}", f"{bd['avg_accuracy']:.1f}%  ×{bd['weight']}")

    output = {
        "meta": {
            "model": model,
            "timestamp": timestamp,
            "base_url": args.base_url,
            "runs": args.runs,
            "warmup": args.warmup,
            "judge_model": JUDGE_MODEL if api_key else None,
        },
        "composite": composite,
        "tests": results,
    }
    path = output_path(model, timestamp)
    path.write_text(json.dumps(output, indent=2))
    _ok(f"Results → {path}")


if __name__ == "__main__":
    main()
