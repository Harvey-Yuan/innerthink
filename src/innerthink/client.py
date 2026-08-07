import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class InnerThinkClientError(RuntimeError):
    """Raised when the local InnerThink API cannot satisfy a request."""


@dataclass(frozen=True)
class InnerThinkClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> "InnerThinkClient":
        return cls(
            base_url=os.getenv("INNERTHINK_API_URL", "http://127.0.0.1:8000").rstrip("/"),
            timeout_seconds=float(os.getenv("INNERTHINK_API_TIMEOUT_SECONDS", "600")),
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise InnerThinkClientError(
                f"InnerThink API returned HTTP {error.code}: {body}"
            ) from error
        except TimeoutError as error:
            raise InnerThinkClientError(
                f"InnerThink request timed out after {self.timeout_seconds}s"
            ) from error
        except URLError as error:
            raise InnerThinkClientError(
                f"Cannot reach InnerThink at {self.base_url}. "
                f"Start innerthink-api first: {error.reason}"
            ) from error

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def generate(
        self,
        prompt: str,
        *,
        mode: str = "latent",
        max_new_tokens: int | None = None,
        latent_iterations: int | None = None,
        intervention_step: int | None = None,
        intervention_scale: float | None = None,
        include_latent_metrics: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt, "mode": mode}
        optional = {
            "max_new_tokens": max_new_tokens,
            "latent_iterations": latent_iterations,
            "intervention_step": intervention_step,
            "intervention_scale": intervention_scale,
            "include_latent_metrics": include_latent_metrics,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return self._request("POST", "/v1/generate", payload)

    def compare(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        latent_iterations: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if latent_iterations is not None:
            payload["latent_iterations"] = latent_iterations
        return self._request("POST", "/v1/compare", payload)

    def intervene(
        self,
        prompt: str,
        *,
        step: int,
        scale: float,
        max_new_tokens: int | None = None,
        latent_iterations: int | None = None,
    ) -> dict[str, Any]:
        baseline = self.generate(
            prompt,
            mode="latent",
            max_new_tokens=max_new_tokens,
            latent_iterations=latent_iterations,
        )
        changed = self.generate(
            prompt,
            mode="latent",
            max_new_tokens=max_new_tokens,
            latent_iterations=latent_iterations,
            intervention_step=step,
            intervention_scale=scale,
        )
        return {
            "baseline": baseline,
            "intervention": {"step": step, "scale": scale},
            "changed": changed,
            "answer_changed": baseline["answer"].strip() != changed["answer"].strip(),
            "elapsed_delta_ms": round(changed["elapsed_ms"] - baseline["elapsed_ms"], 3),
        }
