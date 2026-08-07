import asyncio
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer

from innerthink.client import InnerThinkClient
from innerthink.demo import DEFAULT_DATASET, run_dataset_demo
from innerthink.memory import EverOSClient
from innerthink.telemetry import SnowflakeTelemetry

mcp = MCPServer(
    "InnerThink",
    instructions=(
        "Use the local CODI-Qwen3-8B service for math and reasoning tasks. "
        "The local model is the default InnerThink route. Use qwen_intervene only when "
        "the user explicitly wants to inspect or modify a latent reasoning step."
    ),
)


async def _call(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    client = InnerThinkClient.from_env()
    function = getattr(client, method)
    return await asyncio.to_thread(function, *args, **kwargs)


async def _memory_call(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    client = EverOSClient.from_env()
    function = getattr(client, method)
    return await asyncio.to_thread(function, *args, **kwargs)


@mcp.tool()
async def qwen_reason(
    prompt: str,
    mode: Literal["direct", "latent", "verbalized"] = "latent",
    max_new_tokens: int | None = None,
    latent_iterations: int | None = None,
) -> dict[str, Any]:
    """Answer a math or reasoning prompt with the local CODI-Qwen3-8B model."""
    return await _call(
        "generate",
        prompt,
        mode=mode,
        max_new_tokens=max_new_tokens,
        latent_iterations=latent_iterations,
    )


@mcp.tool()
async def qwen_compare(
    prompt: str,
    max_new_tokens: int | None = None,
    latent_iterations: int | None = None,
) -> dict[str, Any]:
    """Compare direct generation with continuous latent reasoning on local Qwen."""
    return await _call(
        "compare",
        prompt,
        max_new_tokens=max_new_tokens,
        latent_iterations=latent_iterations,
    )


@mcp.tool()
async def qwen_intervene(
    prompt: str,
    step: int,
    scale: float,
    max_new_tokens: int | None = None,
    latent_iterations: int | None = None,
) -> dict[str, Any]:
    """Scale one local CODI latent step, rerun the prompt, and compare with its baseline."""
    return await _call(
        "intervene",
        prompt,
        step=step,
        scale=scale,
        max_new_tokens=max_new_tokens,
        latent_iterations=latent_iterations,
    )


@mcp.tool()
async def recall_reasoning_feedback(
    user_id: str,
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Recall relevant prior reasoning feedback for a user from local EverOS memory."""
    return await _memory_call("recall", user_id, query, top_k=top_k)


@mcp.tool()
async def remember_reasoning_feedback(
    user_id: str,
    prompt: str,
    answer: str,
    feedback: str,
) -> dict[str, Any]:
    """Persist verified user feedback about a reasoning result in local EverOS memory."""
    return await _memory_call(
        "remember_feedback",
        user_id,
        prompt,
        answer,
        feedback,
    )


@mcp.tool()
async def snowflake_economy_summary() -> dict[str, Any]:
    """Summarize local inference runs, tokens, latency, and interventions in Snowflake."""
    telemetry = SnowflakeTelemetry.from_env()
    if telemetry is None:
        raise RuntimeError("Set INNERTHINK_SNOWFLAKE_ENABLED=true before requesting a summary")
    return await asyncio.to_thread(telemetry.summary)


@mcp.tool()
async def qwen_dataset_demo(
    index: int | None = None,
    seed: int = 0,
    step: int = 3,
    scale: float = 0.0,
    dataset: str = str(DEFAULT_DATASET),
) -> dict[str, Any]:
    """Run one GSM8K case through direct, latent, and intervened local Qwen paths."""
    return await asyncio.to_thread(
        run_dataset_demo,
        InnerThinkClient.from_env(),
        dataset=Path(dataset),
        index=index,
        seed=seed,
        step=step,
        scale=scale,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
