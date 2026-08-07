#!/usr/bin/env python3
"""Download GSM8K question splits for local eval."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

# Canonical OpenAI release (same content as Hugging Face openai/gsm8k main).
RAW_BASE = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data"
)
SPLITS = ("train", "test")


def download_split(split: str, dest: Path) -> int:
    url = f"{RAW_BASE}/{split}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return sum(1 for _ in data.splitlines() if _.strip())


def validate_jsonl(path: Path) -> None:
    with path.open() as f:
        first = json.loads(next(f))
    if "question" not in first or "answer" not in first:
        raise SystemExit(f"{path} is missing question/answer fields")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "gsm8k",
        help="Directory for train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Which splits to download (default: train test)",
    )
    args = parser.parse_args()

    for split in args.splits:
        out = args.out_dir / f"{split}.jsonl"
        n = download_split(split, out)
        validate_jsonl(out)
        print(f"{out}: {n} examples")


if __name__ == "__main__":
    main()
