# InnerThink

Local Apple Silicon inference for
[`cds-jb/codi_qwen3-8b-answer_only`](https://huggingface.co/cds-jb/codi_qwen3-8b-answer_only).
The API can answer directly, run CODI's continuous latent reasoning, or emit the
model's separately trained verbalized chain of thought.

This is a research model trained mainly on grade-school math and CommonsenseQA. It
is not a general-purpose chat model.

## Requirements

- Apple Silicon Mac with 32 GB unified memory recommended
- Roughly 40 GB free disk space
- Native arm64 Python 3.12 or 3.13
- macOS with PyTorch MPS support

The CODI checkpoint is about 17 GB and the Qwen3-8B base model is another roughly
17 GB. They are cached by Hugging Face, not copied into this repository. Runtime
uses FP16 and generally occupies 17–20 GB of unified memory.

## Start the endpoint

```bash
./scripts/setup-mac.sh
./scripts/serve.sh
```

The first command creates `.venv`, installs the package, and downloads both models.
Use `./scripts/setup-mac.sh --skip-download` to install only; the server will then
download missing files on first startup.

The server listens on `http://127.0.0.1:8000`. Model loading can take several
minutes before the port becomes available.

Open `http://127.0.0.1:8000` for the live three-path comparison and latent
oscilloscope. The page runs direct, verbalized, and latent inference, then lets you
scale or zero one latent state and compare the regenerated trace and answer.

The **Cached 250** tab browses a deterministic random sample of GSM8K test
questions (`seed=11`) with all three model responses, correctness, activation
fingerprints, aggregate accuracy, and latency comparisons. Populate or resume it
while the server is running:

```bash
.venv/bin/python scripts/cache-gsm8k-results.py
```

```bash
curl http://127.0.0.1:8000/health
```

## Generate

Latent reasoning:

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Janet has 16 eggs. She uses 7 and sells the rest for $2 each. How much does she make? Output only the answer.",
    "mode": "latent"
  }'
```

Direct answer without latent reasoning:

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is 19 + 26 - 7? Output only the answer.","mode":"direct"}'
```

Run both modes:

```bash
curl -s http://127.0.0.1:8000/v1/compare \
  -H 'content-type: application/json' \
  -d '{"prompt":"What is 19 + 26 - 7? Output only the answer."}'
```

`mode` accepts:

- `direct`: bypasses the continuous thought loop.
- `latent`: runs the published six recurrent latent iterations.
- `verbalized`: uses the checkpoint's visible-CoT path. Its text can be noisy because
  this checkpoint was trained in answer-only format.

Greedy decoding is the default so comparisons are repeatable. Sampling is available
with `greedy: false`, `temperature`, `top_k`, and `top_p`.

## CLI and Cursor MCP

The CLI and MCP server are lightweight clients of the local API. Start the model
endpoint on the 32 GB Apple Silicon demo machine, then run these commands from any
machine that can reach it. The default URL is `http://127.0.0.1:8000`; override it
with `INNERTHINK_API_URL` when the model runs on another host.

For a Windows Cursor client and a Mac model host, keep the model API bound to
loopback and create an SSH tunnel from Windows:

```powershell
ssh -L 8000:127.0.0.1:8000 mac-user@mac-host
```

Cursor and the CLI can then keep using `http://127.0.0.1:8000` locally. This is
safer and faster to set up than publicly deploying the model endpoint.

```bash
innerthink health
innerthink generate "What is 19 + 26 - 7? Output only the answer."
innerthink compare "What is 19 + 26 - 7? Output only the answer."
innerthink intervene "What is 19 + 26 - 7? Output only the answer." --step 3 --scale 0
innerthink recall demo-user "What mistakes have I corrected in arithmetic?"
innerthink remember demo-user "What is 2 + 2?" "4" "Correct; keep answers concise."
innerthink cost-report
```

The project-level [`.cursor/mcp.json`](.cursor/mcp.json) starts `innerthink-mcp`
over stdio. After dependencies are installed, restart Cursor and enable the
`innerthink` MCP server. Cursor Auto remains the coding model and can call:

- `qwen_reason` for local math/reasoning;
- `qwen_compare` for direct-versus-latent measurements;
- `qwen_intervene` to scale one CODI latent state and compare the result.
- `recall_reasoning_feedback` and `remember_reasoning_feedback` for EverOS memory.
- `snowflake_economy_summary` for aggregate token and latency economics.

The intervention changes the local CODI computation. MCP cannot inspect or modify
Cursor Auto, Claude, or OpenAI private hidden states; those models can only invoke
the local experiment as a tool.

## EverOS memory service

EverOS runs as a separate local process on port `8001`, avoiding a collision with
the model API on port `8000`:

```bash
./scripts/start-everos.sh
./scripts/demo-memory.sh
```

Provider keys stay in `.env`. Do not expose either local service without an
authenticated gateway.

## Snowflake telemetry

Set `INNERTHINK_SNOWFLAKE_ENABLED=true` and fill the `SNOWFLAKE_*` values in
`.env` to log inference economics. The API creates `INFERENCE_EVENTS` on first
use and records model, mode, latency, token counts, latent iterations, and
interventions. Prompts and answers are stored only as SHA-256 hashes. Telemetry
failures are logged without discarding a successful model result.

## What latent metrics mean

Latent responses include the L2 norm of each projected state and cosine similarity
to the previous state. The published generator has an initial projected state plus
six recurrent passes, so `latent_iterations` is 6 while `latent_states` is 7.

Each state also returns a 64-bin activation fingerprint. The runtime groups the
4096 hidden dimensions into fixed bins, averages each group, and standardizes the
64 values within that state. The local demo renders these real measurements beside
the model's individual verbalized token pieces to show reasoning-route compression.

These values and any token-neighborhood projections are measurements of hidden
states. They are not translations of what the model is "thinking."

## Modify latent reasoning

[`src/innerthink/interventions.py`](src/innerthink/interventions.py) defines the
`LatentHook` seam. A hook receives every projected latent immediately before the
next transformer pass and may return a modified tensor:

```python
from innerthink.interventions import ScaleStepHook

result = runtime.generate(
    prompt,
    mode="latent",
    latent_hook=ScaleStepHook(step=3, scale=0.0),
)
```

The same propagated intervention is available over HTTP:

```bash
curl -s http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "What is 19 + 26 - 7? Output only the answer.",
    "mode": "latent",
    "intervention": {"type": "scale", "step": 3, "scale": 0}
  }'
```

This supports propagated ablations, scaling, noise injection, learned adapters, or
activation collection without changing the API or checkpoint loader. Keep tuning
code separate from the serving process; optimizer state for an 8B model will not fit
comfortably in 32 GB, while projection-only or quantized-base LoRA experiments are
more realistic.

## Configuration

Copy `.env.example` to `.env` to override model IDs, token limits, device, or dtype.
For a private Hugging Face cache, set `INNERTHINK_HF_TOKEN`.

The service intentionally uses one Uvicorn worker and serializes inference. More
workers would load additional 17+ GB copies of the model.

### Troubleshooting

- `MPS is unavailable`: confirm `uname -m` prints `arm64`; do not run Python under
  Rosetta.
- Out of memory: close other GPU-heavy apps and keep one server process. Do not
  disable Metal's memory safety watermark.
- Unsupported MPS operation: `scripts/serve.sh` enables PyTorch's CPU fallback for
  individual unsupported operations.
- Slow first request: model loading and Metal kernel warm-up are one-time costs.

## Hackathon directions

1. **Latent thought oscilloscope (recommended, medium lift).** Animate the seven
   measured states, then let the audience zero or scale step 3 or 4 and watch the
   answer change. Pair it with direct and verbalized outputs.
2. **Hidden-reasoning scoreboard (low lift).** Run 20 curated GSM8K questions and
   show correctness, latency, output tokens, and answer agreement for each mode.
3. **Middle-step ablation lab (low-to-medium lift).** Sweep a zero/scale intervention
   across steps and visualize the answer or accuracy delta. It connects directly to
   the model card's claim that middle latents are most load-bearing.
4. **Thought transplant (medium lift).** Capture states from two problems and swap
   selected states in a two-pass experimental runner. The surprising result to test
   is that foreign thoughts disrupt an answer rather than transplanting the donor's
   answer.

For a five-minute demo, combine ideas 1 and 2: start with a credible aggregate
scoreboard, then perform one live causal intervention.
