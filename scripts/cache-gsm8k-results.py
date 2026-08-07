#!/usr/bin/env python3
"""Cache three CODI inference modes for a deterministic GSM8K test sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "gsm8k" / "test.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "gsm8k-test-random-500-seed-11.jsonl"
MODES = ("direct", "verbalized", "latent")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(mode in row.get("modes", {}) for mode in MODES):
                ids.add(row["id"])
    return ids


def final_answer(reference: str) -> str:
    return reference.rsplit("####", 1)[-1].strip()


def normalize_answer(value: str) -> str:
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", value)
    normalized = matches[-1] if matches else value.strip().lower()
    return normalized.replace(",", "").replace("$", "").strip()


def generate(
    base_url: str,
    question: str,
    mode: str,
    *,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": f"{question}\nOutput only the answer and nothing else.",
            "mode": mode,
            "max_new_tokens": 160 if mode == "verbalized" else 48,
            "greedy": True,
            "include_latent_metrics": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/generate",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt >= retries:
                raise RuntimeError(f"{mode} request failed: {error}") from error
            time.sleep(2**attempt)
    raise RuntimeError(f"{mode} request failed")


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    if args.limit > len(dataset):
        raise SystemExit(f"Requested {args.limit} rows from a {len(dataset)}-row dataset.")
    sampled = random.Random(args.seed).sample(list(enumerate(dataset)), args.limit)
    done = completed_ids(args.output)
    print(f"sample=test random seed={args.seed} size={args.limit} already_cached={len(done)}")

    failures = 0
    for sample_position, (dataset_index, example) in enumerate(sampled):
        result_id = f"gsm8k-test-{dataset_index:04d}"
        if result_id in done:
            continue
        started = time.perf_counter()
        modes: dict[str, Any] = {}
        try:
            for mode in MODES:
                modes[mode] = generate(
                    args.base_url,
                    example["question"],
                    mode,
                    timeout=args.timeout,
                    retries=args.retries,
                )
        except RuntimeError as error:
            failures += 1
            print(f"FAILED position={sample_position + 1} id={result_id} error={error}")
            continue

        expected = final_answer(example["answer"])
        normalized = {
            mode: normalize_answer(response["answer"]) for mode, response in modes.items()
        }
        row = {
            "id": result_id,
            "sample_position": sample_position,
            "dataset_index": dataset_index,
            "sample_seed": args.seed,
            "question": example["question"],
            "reference_solution": example["answer"],
            "expected_answer": expected,
            "modes": modes,
            "normalized_answers": normalized,
            "correct": {
                mode: answer == normalize_answer(expected) for mode, answer in normalized.items()
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }
        append_result(args.output, row)
        elapsed = time.perf_counter() - started
        print(f"CACHED {sample_position + 1}/{args.limit} id={result_id} seconds={elapsed:.2f}")

    cached = min(len(completed_ids(args.output)), args.limit)
    print(f"DONE cached={cached}/{args.limit} failures={failures}")
    if cached < args.limit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
