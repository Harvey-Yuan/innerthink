import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from innerthink.api import create_app
from innerthink.config import Settings
from innerthink.interventions import ScaleStepHook
from innerthink.runtime import InferenceResult


class FakeRuntime:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = False

    def load(self) -> None:
        self.ready = True

    def unload(self) -> None:
        self.ready = False

    def model_info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_id": self.settings.model_id,
            "base_model_id": self.settings.base_model_id,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "latent_iterations": self.settings.latent_iterations,
            "load_elapsed_seconds": 0.01,
        }

    def generate(self, prompt: str, **kwargs: Any) -> InferenceResult:
        type(self).last_kwargs = kwargs
        mode = kwargs["mode"]
        is_latent = mode == "latent"
        return InferenceResult(
            answer="38",
            mode=mode,
            elapsed_ms=12.5 if is_latent else 10.0,
            prompt_tokens=len(prompt.split()),
            output_tokens=1,
            token_pieces=["38"],
            visible_reasoning_tokens=0,
            latent_iterations=6 if is_latent else 0,
            latent_states=7 if is_latent else 0,
            latent_metrics=(
                [{"step": 0, "l2_norm": 1.0, "cosine_from_previous": None}] if is_latent else []
            ),
        )


def test_health_and_generate() -> None:
    app = create_app(
        settings=Settings(device="cpu"),
        runtime_factory=FakeRuntime,
    )
    with TestClient(app) as client:
        demo = client.get("/")
        health = client.get("/health")
        response = client.post(
            "/v1/generate",
            json={"prompt": "What is 19 + 26 - 7?", "mode": "latent"},
        )

    assert demo.status_code == 200
    assert "latent oscilloscope" in demo.text
    assert demo.headers["cache-control"] == "no-store, max-age=0"
    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert response.status_code == 200
    assert response.json()["answer"] == "38"
    assert response.json()["latent_states"] == 7


def test_compare_is_deterministic() -> None:
    app = create_app(
        settings=Settings(device="cpu"),
        runtime_factory=FakeRuntime,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/compare",
            json={"prompt": "What is 19 + 26 - 7?"},
        )

    assert response.status_code == 200
    assert response.json()["answers_match"] is True
    assert response.json()["latent_minus_direct_ms"] == 2.5


def test_rejects_latent_iterations_in_direct_mode() -> None:
    app = create_app(
        settings=Settings(device="cpu"),
        runtime_factory=FakeRuntime,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "What is 2 + 2?",
                "mode": "direct",
                "latent_iterations": 3,
            },
        )

    assert response.status_code == 422


def test_applies_latent_scale_intervention() -> None:
    app = create_app(
        settings=Settings(device="cpu"),
        runtime_factory=FakeRuntime,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "What is 2 + 2?",
                "mode": "latent",
                "intervention": {"type": "scale", "step": 3, "scale": 0},
            },
        )

    assert response.status_code == 200
    hook = FakeRuntime.last_kwargs["latent_hook"]
    assert isinstance(hook, ScaleStepHook)
    assert hook.step == 3
    assert hook.scale == 0


def test_reads_cached_result_index_and_detail(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    row = {
        "id": "gsm8k-test-0007",
        "sample_position": 0,
        "dataset_index": 7,
        "question": "What is 2 + 2?",
        "expected_answer": "4",
        "reference_solution": "2 + 2 = 4\n#### 4",
        "modes": {
            mode: {"answer": "4", "output_tokens": 1} for mode in ("direct", "verbalized", "latent")
        },
        "normalized_answers": {
            "direct": "4",
            "verbalized": "4",
            "latent": "4",
        },
        "correct": {"direct": True, "verbalized": True, "latent": True},
    }
    results_path.write_text(json.dumps(row) + "\n")
    app = create_app(
        settings=Settings(
            device="cpu",
            results_path=results_path,
            results_expected=250,
        ),
        runtime_factory=FakeRuntime,
    )

    with TestClient(app) as client:
        index = client.get("/v1/results")
        detail = client.get("/v1/results/gsm8k-test-0007")

    assert index.status_code == 200
    assert index.json()["progress"]["completed"] == 1
    assert index.json()["aggregate"]["accuracy"]["direct"] == 100.0
    assert detail.status_code == 200
    assert detail.json()["question"] == "What is 2 + 2?"
