from dataclasses import asdict, dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class LatentHook(Protocol):
    """Transform one projected latent before it is consumed by the next step."""

    def __call__(self, step: int, latent: torch.Tensor) -> torch.Tensor: ...


class IdentityHook:
    def __call__(self, step: int, latent: torch.Tensor) -> torch.Tensor:
        del step
        return latent


@dataclass(frozen=True)
class ScaleStepHook:
    """Small example intervention for future causal experiments."""

    step: int
    scale: float

    def __call__(self, step: int, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.scale if step == self.step else latent


@dataclass(frozen=True)
class LatentStepMetrics:
    step: int
    l2_norm: float
    cosine_from_previous: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


class RecordingHook:
    """Record compact metrics while optionally applying another hook."""

    def __init__(self, inner: LatentHook | None = None) -> None:
        self.inner = inner or IdentityHook()
        self.steps: list[LatentStepMetrics] = []
        self._previous: torch.Tensor | None = None

    def __call__(self, step: int, latent: torch.Tensor) -> torch.Tensor:
        transformed = self.inner(step, latent)
        current = transformed.detach().float().reshape(transformed.shape[0], -1).cpu()
        norm = torch.linalg.vector_norm(current, dim=-1).mean().item()
        cosine = None
        if self._previous is not None:
            cosine = F.cosine_similarity(current, self._previous, dim=-1).mean().item()
        self.steps.append(
            LatentStepMetrics(
                step=step,
                l2_norm=round(norm, 6),
                cosine_from_previous=None if cosine is None else round(cosine, 6),
            )
        )
        self._previous = current
        return transformed
