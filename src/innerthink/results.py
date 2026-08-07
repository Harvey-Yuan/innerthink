import json
from pathlib import Path
from typing import Any

from innerthink.config import Settings


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                # A reader may catch the final line while the batch process is appending it.
                continue
    return results


def result_index(settings: Settings) -> dict[str, Any]:
    rows = sorted(
        load_results(settings.results_path),
        key=lambda row: row.get("sample_position", 0),
    )[: settings.results_expected]
    summaries = []
    completed = 0
    mode_names = ("direct", "verbalized", "latent")
    correct_counts = {mode: 0 for mode in mode_names}
    latency_values: dict[str, list[float]] = {mode: [] for mode in mode_names}
    token_savings: list[int] = []
    for row in rows:
        modes = row.get("modes", {})
        complete = all(mode in modes for mode in mode_names)
        if complete:
            completed += 1
        correctness = row.get("correct", {})
        for mode in mode_names:
            if correctness.get(mode):
                correct_counts[mode] += 1
            if mode in modes:
                latency_values[mode].append(float(modes[mode].get("elapsed_ms", 0)))
        answers = row.get("normalized_answers", {})
        verbal_tokens = modes.get("verbalized", {}).get("output_tokens", 0)
        latent_tokens = modes.get("latent", {}).get("output_tokens", 0)
        if complete:
            token_savings.append(max(0, verbal_tokens - latent_tokens))
        summaries.append(
            {
                "id": row["id"],
                "sample_position": row["sample_position"],
                "dataset_index": row["dataset_index"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "complete": complete,
                "correct": correctness,
                "normalized_answers": answers,
            }
        )
    summaries.sort(key=lambda item: item["sample_position"])
    return {
        "sample": {
            "dataset": "gsm8k",
            "split": "test",
            "strategy": "random",
            "seed": 11,
            "expected": settings.results_expected,
        },
        "progress": {
            "completed": completed,
            "expected": settings.results_expected,
            "percent": round(completed / settings.results_expected * 100, 1),
        },
        "aggregate": {
            "accuracy": {
                mode: round(correct_counts[mode] / completed * 100, 1)
                if completed
                else None
                for mode in mode_names
            },
            "latency": {
                mode: {
                    "average_ms": round(sum(values) / len(values), 1) if values else None,
                    "total_ms": round(sum(values), 1) if values else None,
                }
                for mode, values in latency_values.items()
            },
            "average_visible_tokens_saved": round(sum(token_savings) / len(token_savings), 1)
            if token_savings
            else None,
        },
        "questions": summaries,
    }


def result_detail(settings: Settings, result_id: str) -> dict[str, Any] | None:
    rows = sorted(
        load_results(settings.results_path),
        key=lambda row: row.get("sample_position", 0),
    )[: settings.results_expected]
    for row in rows:
        if row.get("id") == result_id:
            return row
    return None
