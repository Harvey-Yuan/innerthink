import json
from pathlib import Path
from unittest.mock import Mock

from innerthink.demo import load_problem, run_dataset_demo


def _dataset(path: Path) -> None:
    path.write_text(
        json.dumps({"question": "What is 2 + 2?", "answer": "Work. #### 4"}) + "\n",
        encoding="utf-8",
    )


def test_load_problem_extracts_gsm8k_reference_answer(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    _dataset(dataset)

    problem = load_problem(dataset, index=0)

    assert problem["expected_answer"] == "4"
    assert problem["index"] == 0


def test_dataset_demo_scores_direct_latent_and_intervention(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    _dataset(dataset)
    client = Mock()
    client.compare.return_value = {
        "direct": {"answer": "4", "elapsed_ms": 10, "output_tokens": 1},
        "latent": {"answer": "\\boxed{4}", "elapsed_ms": 12, "output_tokens": 1},
        "answers_match": False,
    }
    client.generate.return_value = {"answer": "5", "elapsed_ms": 12, "output_tokens": 1}

    result = run_dataset_demo(client, dataset=dataset, index=0)

    assert result["results"]["direct"]["correct"] is True
    assert result["results"]["latent"]["correct"] is True
    assert result["results"]["intervened"]["correct"] is False
    assert result["intervention_changed_answer"] is True
