import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from innerthink.client import InnerThinkClient, InnerThinkClientError
from innerthink.demo import DEFAULT_DATASET, run_dataset_demo
from innerthink.memory import EverOSClient, EverOSClientError
from innerthink.telemetry import SnowflakeTelemetry


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="innerthink",
        description="Use the local CODI-Qwen3 reasoning service from a terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check whether the local model is ready.")

    generate = subparsers.add_parser("generate", help="Generate with local Qwen reasoning.")
    generate.add_argument("prompt")
    generate.add_argument("--mode", choices=("direct", "latent", "verbalized"), default="latent")
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--latent-iterations", type=int)

    compare = subparsers.add_parser("compare", help="Compare direct and latent local inference.")
    compare.add_argument("prompt")
    compare.add_argument("--max-new-tokens", type=int)
    compare.add_argument("--latent-iterations", type=int)

    intervene = subparsers.add_parser(
        "intervene",
        help="Scale one recurrent latent state and compare it with the baseline.",
    )
    intervene.add_argument("prompt")
    intervene.add_argument("--step", type=int, required=True)
    intervene.add_argument("--scale", type=float, required=True)
    intervene.add_argument("--max-new-tokens", type=int)
    intervene.add_argument("--latent-iterations", type=int)

    recall = subparsers.add_parser("recall", help="Recall reasoning feedback from EverOS.")
    recall.add_argument("user_id")
    recall.add_argument("query")
    recall.add_argument("--top-k", type=int, default=5)

    remember = subparsers.add_parser("remember", help="Store reasoning feedback in EverOS.")
    remember.add_argument("user_id")
    remember.add_argument("prompt")
    remember.add_argument("answer")
    remember.add_argument("feedback")

    subparsers.add_parser("cost-report", help="Summarize inference economics in Snowflake.")
    subparsers.add_parser(
        "snowflake-check",
        help="Verify Snowflake credentials and prepare the inference telemetry table.",
    )

    demo = subparsers.add_parser(
        "demo",
        help="Run one reproducible GSM8K direct/latent/intervention comparison.",
    )
    demo.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    demo.add_argument("--index", type=int)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--step", type=int, default=3)
    demo.add_argument("--scale", type=float, default=0.0)
    demo.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    client = InnerThinkClient.from_env()
    try:
        if args.command == "health":
            result = client.health()
        elif args.command == "generate":
            result = client.generate(
                args.prompt,
                mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                latent_iterations=args.latent_iterations,
            )
        elif args.command == "compare":
            result = client.compare(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                latent_iterations=args.latent_iterations,
            )
        elif args.command == "intervene":
            result = client.intervene(
                args.prompt,
                step=args.step,
                scale=args.scale,
                max_new_tokens=args.max_new_tokens,
                latent_iterations=args.latent_iterations,
            )
        elif args.command == "recall":
            result = EverOSClient.from_env().recall(
                args.user_id,
                args.query,
                top_k=args.top_k,
            )
        elif args.command == "remember":
            result = EverOSClient.from_env().remember_feedback(
                args.user_id,
                args.prompt,
                args.answer,
                args.feedback,
            )
        elif args.command == "cost-report":
            telemetry = SnowflakeTelemetry.from_env()
            if telemetry is None:
                raise SystemExit(
                    "Set INNERTHINK_SNOWFLAKE_ENABLED=true and configure SNOWFLAKE_* first"
                )
            result = telemetry.summary()
        elif args.command == "snowflake-check":
            telemetry = SnowflakeTelemetry.from_env()
            if telemetry is None:
                raise SystemExit(
                    "Set INNERTHINK_SNOWFLAKE_ENABLED=true and configure SNOWFLAKE_* first"
                )
            result = telemetry.check_connection()
        else:
            result = run_dataset_demo(
                client,
                dataset=args.dataset,
                index=args.index,
                seed=args.seed,
                step=args.step,
                scale=args.scale,
                max_new_tokens=args.max_new_tokens,
            )
    except (InnerThinkClientError, EverOSClientError) as error:
        raise SystemExit(str(error)) from error
    _print_json(result)
