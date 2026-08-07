from typing import Literal

from pydantic import BaseModel, Field, model_validator

GenerationMode = Literal["direct", "latent", "verbalized"]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    mode: GenerationMode = "latent"
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    latent_iterations: int | None = Field(default=None, ge=0, le=32)
    greedy: bool = True
    temperature: float = Field(default=0.1, gt=0, le=5)
    top_k: int = Field(default=40, ge=0, le=500)
    top_p: float = Field(default=0.95, gt=0, le=1)
    include_latent_metrics: bool = True
    intervention_step: int | None = Field(default=None, ge=0, le=32)
    intervention_scale: float | None = Field(default=None, ge=-4, le=4)

    @model_validator(mode="after")
    def validate_mode_options(self) -> "GenerateRequest":
        if self.mode != "latent" and self.latent_iterations is not None:
            raise ValueError("latent_iterations is only valid in latent mode")
        has_step = self.intervention_step is not None
        has_scale = self.intervention_scale is not None
        if has_step != has_scale:
            raise ValueError("intervention_step and intervention_scale must be provided together")
        if has_step and self.mode != "latent":
            raise ValueError("latent intervention is only valid in latent mode")
        if (
            self.intervention_step is not None
            and self.latent_iterations is not None
            and self.intervention_step > self.latent_iterations
        ):
            raise ValueError("intervention_step cannot exceed latent_iterations")
        return self


class LatentStepResponse(BaseModel):
    step: int
    l2_norm: float
    cosine_from_previous: float | None


class GenerateResponse(BaseModel):
    answer: str
    mode: GenerationMode
    elapsed_ms: float
    prompt_tokens: int
    output_tokens: int
    visible_reasoning_tokens: int
    latent_iterations: int
    latent_states: int
    latent_metrics: list[LatentStepResponse]


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=512)
    latent_iterations: int | None = Field(default=None, ge=0, le=32)
    include_latent_metrics: bool = True


class CompareResponse(BaseModel):
    direct: GenerateResponse
    latent: GenerateResponse
    answers_match: bool
    latent_minus_direct_ms: float
    visible_token_reduction: int


class HealthResponse(BaseModel):
    ready: bool
    model_id: str
    base_model_id: str
    device: str
    dtype: str
    latent_iterations: int
    load_elapsed_seconds: float | None
