#!/usr/bin/env python3
"""Evaluate CODI modes on a random GSM8K test subset.

Modes:
  - latent: continuous latent reasoning (hidden CoT)
  - verbalized: visible chain-of-thought ("reasoning" mode)

Reports accuracy, token cost (output tokens), and latency statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from innerthink.client import InnerThinkClient, InnerThinkClientError

ANSWER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
HASH_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def normalize_number(text: str) -> str | None:
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isfinite(value) and value == int(value):
        return str(int(value))
    return str(value)


def gold_answer(answer_field: str) -> str:
    match = HASH_ANSWER_RE.search(answer_field)
    if not match:
        raise ValueError(f"GSM8K answer missing #### marker: {answer_field!r}")
    normalized = normalize_number(match.group(1))
    if normalized is None:
        raise ValueError(f"Could not normalize gold answer: {match.group(1)!r}")
    return normalized


def extract_prediction(text: str) -> str | None:
    hash_match = HASH_ANSWER_RE.search(text)
    if hash_match:
        return normalize_number(hash_match.group(1))
    numbers = ANSWER_RE.findall(text)
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                {
                    "index": index,
                    "question": row["question"].strip(),
                    "gold": gold_answer(row["answer"]),
                    "gold_solution": row["answer"],
                }
            )
    return examples


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    latencies = [float(row["elapsed_ms"]) for row in rows]
    tokens = [int(row["output_tokens"]) for row in rows]
    visible = [int(row["visible_reasoning_tokens"]) for row in rows]
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 1) if latencies else None,
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
            "total": round(sum(latencies), 1) if latencies else None,
        },
        "token_cost_output": {
            "mean": round(statistics.fmean(tokens), 2) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "p95": round(percentile([float(t) for t in tokens], 0.95), 2) if tokens else None,
            "total": sum(tokens),
        },
        "visible_reasoning_tokens": {
            "mean": round(statistics.fmean(visible), 2) if visible else None,
            "median": statistics.median(visible) if visible else None,
            "total": sum(visible),
        },
    }


def format_prompt(question: str) -> str:
    return f"{question}\nOutput only the answer."


def run_one(
    client: InnerThinkClient,
    *,
    example: dict[str, Any],
    mode: str,
    max_new_tokens: int,
    retries: int = 3,
) -> dict[str, Any]:
    prompt = format_prompt(example["question"])
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.generate(
                prompt,
                mode=mode,
                max_new_tokens=max_new_tokens,
                include_latent_metrics=False,
            )
            break
        except InnerThinkClientError as error:
            last_error = error
            print(f"  retry {attempt}/{retries} after error: {error}", flush=True)
            time.sleep(min(5 * attempt, 20))
    else:
        assert last_error is not None
        raise last_error

    prediction = extract_prediction(response["answer"])
    return {
        "index": example["index"],
        "mode": mode,
        "gold": example["gold"],
        "prediction": prediction,
        "correct": prediction is not None and prediction == example["gold"],
        "answer_text": response["answer"],
        "elapsed_ms": response["elapsed_ms"],
        "prompt_tokens": response["prompt_tokens"],
        "output_tokens": response["output_tokens"],
        "visible_reasoning_tokens": response["visible_reasoning_tokens"],
        "latent_iterations": response.get("latent_iterations", 0),
    }


def print_summary(label: str, stats: dict[str, Any]) -> None:
    latency = stats["latency_ms"]
    tokens = stats["token_cost_output"]
    visible = stats["visible_reasoning_tokens"]
    print(f"\n=== {label} (n={stats['n']}) ===")
    print(f"accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['n']})")
    print(
        "token_cost (output_tokens): "
        f"mean={tokens['mean']} median={tokens['median']} "
        f"p95={tokens['p95']} total={tokens['total']}"
    )
    print(
        "visible_reasoning_tokens: "
        f"mean={visible['mean']} median={visible['median']} total={visible['total']}"
    )
    print(
        "latency_ms: "
        f"mean={latency['mean']} median={latency['median']} "
        f"p95={latency['p95']} min={latency['min']} max={latency['max']} "
        f"total={latency['total']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gsm8k" / "test.jsonl",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of random test questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["latent", "verbalized"],
        choices=["direct", "latent", "verbalized"],
    )
    parser.add_argument("--max-new-tokens-latent", type=int, default=64)
    parser.add_argument("--max-new-tokens-verbalized", type=int, default=256)
    parser.add_argument("--max-new-tokens-direct", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "evals" / "gsm8k_eval.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed (index, mode) pairs found in --out",
    )
    args = parser.parse_args()

    examples = load_examples(args.data)
    if args.n > len(examples):
        raise SystemExit(f"Requested {args.n} examples but only {len(examples)} available")
    rng = random.Random(args.seed)
    sample = rng.sample(examples, args.n)

    token_limits = {
        "latent": args.max_new_tokens_latent,
        "verbalized": args.max_new_tokens_verbalized,
        "direct": args.max_new_tokens_direct,
    }

    # Verbalized GSM8K runs can exceed the default 10-minute client timeout.
    client = InnerThinkClient(
        base_url=os.getenv("INNERTHINK_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        timeout_seconds=float(os.getenv("INNERTHINK_API_TIMEOUT_SECONDS", "1800")),
    )
    health = client.health()
    if not health.get("ready"):
        raise SystemExit(f"InnerThink API is not ready: {health}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[int, str], dict[str, Any]] = {}
    if args.resume and args.out.exists():
        previous = json.loads(args.out.read_text())
        for row in previous.get("results", []):
            done[(row["index"], row["mode"])] = row
        print(f"Resuming with {len(done)} completed runs from {args.out}")

    results: list[dict[str, Any]] = list(done.values())
    total = args.n * len(args.modes)
    completed = sum(1 for ex in sample for mode in args.modes if (ex["index"], mode) in done)
    started = time.perf_counter()

    print(
        f"Evaluating {args.n} GSM8K questions × {len(args.modes)} modes "
        f"(seed={args.seed}, remaining={total - completed})"
    )

    try:
        for ex_i, example in enumerate(sample, start=1):
            for mode in args.modes:
                key = (example["index"], mode)
                if key in done:
                    continue
                print(
                    f"[{completed + 1}/{total}] q={example['index']} "
                    f"mode={mode} gold={example['gold']}",
                    flush=True,
                )
                try:
                    row = run_one(
                        client,
                        example=example,
                        mode=mode,
                        max_new_tokens=token_limits[mode],
                    )
                except InnerThinkClientError as error:
                    raise SystemExit(f"API error on index={example['index']} mode={mode}: {error}")
                results.append(row)
                done[key] = row
                completed += 1
                print(
                    f"  -> pred={row['prediction']} correct={row['correct']} "
                    f"tokens={row['output_tokens']} latency_ms={row['elapsed_ms']:.0f}",
                    flush=True,
                )
                payload = {
                    "config": {
                        "n": args.n,
                        "seed": args.seed,
                        "modes": args.modes,
                        "sample_indices": [ex["index"] for ex in sample],
                        "model": health,
                        "max_new_tokens": token_limits,
                    },
                    "results": results,
                    "summary": {
                        mode: summarize([r for r in results if r["mode"] == mode])
                        for mode in args.modes
                    },
                    "elapsed_wall_seconds": round(time.perf_counter() - started, 1),
                }
                args.out.write_text(json.dumps(payload, indent=2))
    except KeyboardInterrupt:
        print("\nInterrupted; partial results saved.", flush=True)

    summary = {
        mode: summarize([r for r in results if r["mode"] == mode]) for mode in args.modes
    }
    for mode in args.modes:
        label = "reasoning (verbalized)" if mode == "verbalized" else mode
        print_summary(label, summary[mode])

    if "latent" in summary and "verbalized" in summary:
        lat = summary["latent"]
        verb = summary["verbalized"]
        if lat["n"] and verb["n"]:
            token_delta = verb["token_cost_output"]["mean"] - lat["token_cost_output"]["mean"]
            latency_delta = verb["latency_ms"]["mean"] - lat["latency_ms"]["mean"]
            print("\n=== latent vs reasoning (verbalized) ===")
            print(
                f"accuracy delta (latent - reasoning): "
                f"{lat['accuracy'] - verb['accuracy']:+.2%}"
            )
            print(f"mean output token delta (reasoning - latent): {token_delta:+.2f}")
            print(f"mean latency_ms delta (reasoning - latent): {latency_delta:+.1f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
