#!/usr/bin/env python3
import json

from innerthink.config import get_settings
from innerthink.runtime import CodiRuntime

PROMPT = (
    "Mom bought a set of pots for $19 and garden soil for $26, then used a "
    "$7-off coupon. How much did she spend? Output only the answer and nothing else."
)


def main() -> None:
    runtime = CodiRuntime(get_settings())
    runtime.load()
    print(json.dumps(runtime.model_info(), indent=2))
    for mode in ("direct", "latent"):
        result = runtime.generate(
            PROMPT,
            mode=mode,
            max_new_tokens=16,
            greedy=True,
        )
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
