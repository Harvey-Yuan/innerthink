import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from innerthink.config import Settings, get_settings
from innerthink.interventions import ScaleStepHook
from innerthink.results import result_detail, result_index
from innerthink.runtime import CodiRuntime, InferenceResult
from innerthink.schemas import (
    CompareRequest,
    CompareResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)
from innerthink.telemetry import SnowflakeTelemetry, record_safely

logger = logging.getLogger(__name__)
DEMO_PATH = Path(__file__).with_name("static") / "index.html"


class RuntimeProtocol(Protocol):
    def load(self) -> None: ...

    def unload(self) -> None: ...

    def model_info(self) -> dict[str, Any]: ...

    def generate(self, prompt: str, **kwargs: Any) -> InferenceResult: ...


RuntimeFactory = Callable[[Settings], RuntimeProtocol]


def _response(result: InferenceResult) -> GenerateResponse:
    return GenerateResponse.model_validate(result.to_dict())


def create_app(
    *,
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = CodiRuntime,
) -> FastAPI:
    app_settings = settings or get_settings()
    inference_lock = asyncio.Lock()
    telemetry = SnowflakeTelemetry.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory(app_settings)
        app.state.runtime = runtime
        if app_settings.preload_model:
            await asyncio.to_thread(runtime.load)
        yield
        await asyncio.to_thread(runtime.unload)

    application = FastAPI(
        title="InnerThink CODI API",
        version="0.1.0",
        description="Local direct and continuous-latent reasoning with CODI-Qwen3-8B.",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> dict[str, Any]:
        return request.app.state.runtime.model_info()

    @application.get("/", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(
            DEMO_PATH,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @application.get("/v1/results")
    async def cached_results() -> dict[str, Any]:
        return await asyncio.to_thread(result_index, app_settings)

    @application.get("/v1/results/{result_id}")
    async def cached_result(result_id: str) -> dict[str, Any]:
        result = await asyncio.to_thread(result_detail, app_settings, result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Cached result not found.")
        return result

    async def infer(
        request: Request,
        prompt: str,
        **kwargs: Any,
    ) -> InferenceResult:
        runtime: RuntimeProtocol = request.app.state.runtime
        try:
            async with inference_lock:
                result = await asyncio.to_thread(runtime.generate, prompt, **kwargs)
            if telemetry is not None:
                await asyncio.to_thread(
                    record_safely,
                    telemetry,
                    result,
                    prompt=prompt,
                    model_id=app_settings.model_id,
                    intervention_step=(
                        kwargs["latent_hook"].step if kwargs.get("latent_hook") else None
                    ),
                    intervention_scale=(
                        kwargs["latent_hook"].scale if kwargs.get("latent_hook") else None
                    ),
                )
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            logger.exception("Inference failed")
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.post("/v1/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        latent_hook = None
        if payload.intervention is not None:
            latent_hook = ScaleStepHook(
                step=payload.intervention.step,
                scale=payload.intervention.scale,
            )
        result = await infer(
            request,
            payload.prompt,
            mode=payload.mode,
            max_new_tokens=payload.max_new_tokens,
            latent_iterations=payload.latent_iterations,
            greedy=payload.greedy,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            include_latent_metrics=payload.include_latent_metrics,
            latent_hook=latent_hook,
        )
        return _response(result)

    @application.post("/v1/compare", response_model=CompareResponse)
    async def compare(payload: CompareRequest, request: Request) -> CompareResponse:
        common = {
            "max_new_tokens": payload.max_new_tokens,
            "greedy": True,
            "include_latent_metrics": payload.include_latent_metrics,
        }
        direct = await infer(request, payload.prompt, mode="direct", **common)
        latent = await infer(
            request,
            payload.prompt,
            mode="latent",
            latent_iterations=payload.latent_iterations,
            **common,
        )
        return CompareResponse(
            direct=_response(direct),
            latent=_response(latent),
            answers_match=direct.answer.strip() == latent.answer.strip(),
            latent_minus_direct_ms=round(latent.elapsed_ms - direct.elapsed_ms, 3),
            visible_token_reduction=direct.output_tokens - latent.output_tokens,
        )

    return application


app = create_app()


def run() -> None:
    uvicorn.run("innerthink.api:app", host="127.0.0.1", port=8000, workers=1)
