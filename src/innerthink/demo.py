import json
import random
import re
from pathlib import Path
from typing import Any

from innerthink.client import InnerThinkClient

DEFAULT_DATASET = Path("data/gsm8k/test.jsonl")


def _reference_answer(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _normalized_answer(answer: str) -> str:
    value = answer.strip().replace(",", "").replace("$", "")
    boxed = re.search(r"\\boxed\{([^{}]+)\}", value)
    if boxed:
        value = boxed.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", value)
    return numbers[-1] if numbers else value.lower().rstrip(".")


def load_problem(
    dataset: Path,
    *,
    index: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not dataset.is_file():
        raise ValueError(
            f"Dataset not found at {dataset}. Run: uv run python scripts/download-gsm8k.py"
        )
    with dataset.open(encoding="utf-8") as handle:
        problems = [json.loads(line) for line in handle if line.strip()]
    if not problems:
        raise ValueError(f"Dataset is empty: {dataset}")
    selected_index = random.Random(seed).randrange(len(problems)) if index is None else index
    if not 0 <= selected_index < len(problems):
        raise ValueError(f"Problem index {selected_index} is outside 0..{len(problems) - 1}")
    problem = problems[selected_index]
    if "question" not in problem or "answer" not in problem:
        raise ValueError("Dataset rows must contain question and answer fields")
    return {
        "index": selected_index,
        "count": len(problems),
        "question": problem["question"],
        "expected_answer": _reference_answer(problem["answer"]),
    }


def run_dataset_demo(
    client: InnerThinkClient,
    *,
    dataset: Path = DEFAULT_DATASET,
    index: int | None = None,
    seed: int = 0,
    step: int = 3,
    scale: float = 0.0,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    problem = load_problem(dataset, index=index, seed=seed)
    prompt = f"{problem['question']} Output only the final answer and nothing else."
    comparison = client.compare(prompt, max_new_tokens=max_new_tokens)
    changed = client.generate(
        prompt,
        mode="latent",
        max_new_tokens=max_new_tokens,
        intervention_step=step,
        intervention_scale=scale,
    )
    expected = _normalized_answer(problem["expected_answer"])
    direct = comparison["direct"]
    latent = comparison["latent"]
    return {
        "problem": problem,
        "results": {
            "direct": {
                **direct,
                "correct": _normalized_answer(direct["answer"]) == expected,
            },
            "latent": {
                **latent,
                "correct": _normalized_answer(latent["answer"]) == expected,
            },
            "intervened": {
                **changed,
                "correct": _normalized_answer(changed["answer"]) == expected,
            },
        },
        "intervention": {"step": step, "scale": scale},
        "direct_and_latent_match": comparison["answers_match"],
        "intervention_changed_answer": (
            _normalized_answer(latent["answer"]) != _normalized_answer(changed["answer"])
        ),
    }
