from typing import Any

from fastapi.testclient import TestClient

from innerthink.api import create_app
from innerthink.config import Settings
from innerthink.runtime import InferenceResult


class FakeRuntime:
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
        mode = kwargs["mode"]
        is_latent = mode == "latent"
        return InferenceResult(
            answer="38",
            mode=mode,
            elapsed_ms=12.5 if is_latent else 10.0,
            prompt_tokens=len(prompt.split()),
            output_tokens=1,
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
        health = client.get("/health")
        response = client.post(
            "/v1/generate",
            json={"prompt": "What is 19 + 26 - 7?", "mode": "latent"},
        )

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
