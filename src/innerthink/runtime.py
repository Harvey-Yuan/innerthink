import gc
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from innerthink.codi import CodiQwen, GenerationMode
from innerthink.config import Settings
from innerthink.interventions import LatentHook, RecordingHook

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    answer: str
    mode: GenerationMode
    elapsed_ms: float
    prompt_tokens: int
    output_tokens: int
    visible_reasoning_tokens: int
    latent_iterations: int
    latent_states: int
    latent_metrics: list[dict[str, float | int | None]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodiRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: CodiQwen | None = None
        self.load_elapsed_seconds: float | None = None
        self.checkpoint_path: Path | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def device(self) -> str:
        return self.settings.device

    def _torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.settings.dtype == "float16" else torch.float32

    def _validate_device(self) -> torch.device:
        if self.settings.device == "mps":
            if not torch.backends.mps.is_built():
                raise RuntimeError("This PyTorch build does not include MPS support.")
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS is unavailable. Use a native arm64 Python and a current PyTorch build."
                )
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> None:
        if self.model is not None:
            return

        started = time.perf_counter()
        device = self._validate_device()
        dtype = self._torch_dtype()
        cache_dir = str(self.settings.cache_dir) if self.settings.cache_dir else None
        logger.info("Initializing %s on %s with %s", self.settings.model_id, device, dtype)

        model = CodiQwen(
            base_model_id=self.settings.base_model_id,
            checkpoint_id=self.settings.model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            token=self.settings.hf_token,
        )
        checkpoint = Path(
            hf_hub_download(
                repo_id=self.settings.model_id,
                filename="pytorch_model.bin",
                cache_dir=cache_dir,
                token=self.settings.hf_token,
            )
        )
        self.checkpoint_path = checkpoint
        logger.info("Memory-mapping CODI checkpoint at %s", checkpoint)
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            missing = incompatible.missing_keys[:10]
            unexpected = incompatible.unexpected_keys[:10]
            raise RuntimeError(
                "CODI checkpoint does not match the local architecture. "
                f"Missing ({len(incompatible.missing_keys)}): {missing}; "
                f"unexpected ({len(incompatible.unexpected_keys)}): {unexpected}"
            )

        model.codi.tie_weights()
        model.eval()
        model.requires_grad_(False)
        model.to(device=device, dtype=dtype)
        del state_dict
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        self.model = model
        self.load_elapsed_seconds = round(time.perf_counter() - started, 3)
        logger.info("Model ready in %.3f seconds", self.load_elapsed_seconds)

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def model_info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_id": self.settings.model_id,
            "base_model_id": self.settings.base_model_id,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "latent_iterations": self.settings.latent_iterations,
            "load_elapsed_seconds": self.load_elapsed_seconds,
        }

    def generate(
        self,
        prompt: str,
        *,
        mode: GenerationMode = "latent",
        max_new_tokens: int | None = None,
        latent_iterations: int | None = None,
        greedy: bool = True,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.95,
        include_latent_metrics: bool = True,
        latent_hook: LatentHook | None = None,
    ) -> InferenceResult:
        if self.model is None:
            raise RuntimeError("Model has not been loaded.")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        max_tokens = max_new_tokens or self.settings.max_new_tokens
        iterations = (
            self.settings.latent_iterations if latent_iterations is None else latent_iterations
        )
        encoded = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens > self.settings.max_prompt_tokens:
            raise ValueError(
                f"Prompt has {prompt_tokens} tokens; limit is {self.settings.max_prompt_tokens}."
            )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        recorder = RecordingHook(latent_hook) if mode == "latent" else None
        started = time.perf_counter()
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode=mode,
            max_new_tokens=max_tokens,
            latent_iterations=iterations,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            latent_hook=recorder,
            return_latent_vectors=False,
        )
        if self.model.device.type == "mps":
            torch.mps.synchronize()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

        sequence = output.sequences[0].detach().cpu()
        answer = self.model.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        special_ids = set(self.model.tokenizer.all_special_ids)
        output_tokens = int(sum(token_id not in special_ids for token_id in sequence.tolist()))
        metrics = (
            [step.to_dict() for step in recorder.steps]
            if recorder is not None and include_latent_metrics
            else []
        )
        return InferenceResult(
            answer=answer,
            mode=mode,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            visible_reasoning_tokens=output_tokens if mode == "verbalized" else 0,
            latent_iterations=iterations if mode == "latent" else 0,
            latent_states=len(recorder.steps) if recorder is not None else 0,
            latent_metrics=metrics,
        )
