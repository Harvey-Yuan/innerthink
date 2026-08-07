# InnerThink deck

The 3-minute CODI presentation. One self-contained file — `index.html` — no build step, no dependencies. Fonts load from Google; everything else is inline.

## Run

```bash
./scripts/serve-deck.sh
```

→ http://localhost:8080/ (override with `PORT=9000`). `.claude/launch.json` has the same server as the `deck` config.

The server points at `./deck`, **not** the project root, and that is deliberate: the root holds `.env` with the OpenRouter and DeepInfra keys, and a static server on the root would publish them at `/.env`. Don't widen the served directory — put deck assets inside `deck/`.

## Present

| Key | |
|---|---|
| `←` `→` `space` | move |
| `Home` `End` | first / last |
| `r` | replay the run on slide 03 |
| `f` | fullscreen |

The current slide is mirrored into the URL hash, so `#3` deep-links to slide 03. The deck fills any browser window; there is no fixed stage.

## The four slides

1. **Identity** — wordmark, the one-line thesis, three spec cards.
2. **Mechanism** — the CODI inference loop animated six times, then self-distillation.
3. **Side by side** — GSM8K's Natalia problem run against explicit CoT and latent CoT.
4. **Generalisation** — the adapter is 4.3% of the weights, and it moves between bases.

## Design language

**Paper is language. Apertures are thought.** That single idea drives every choice.

- The page is cool grey paper with strictly ink/graphite type. The only dark objects anywhere are *apertures* — windows into the model.
- **Colour exists only inside an aperture**, as a continuous indigo → violet → magenta → amber ramp. CODI's thoughts are continuous vectors, so the encoding is a continuous ramp, not a flat brand colour. Outside apertures the sole accent is `--sig` (`#E8237E`), spent on active nav, the recurrence path, and one number per slide.
- Discrete tokens always render as monospace chips; latent state always renders as a heatmap. The two registers must never be drawn the same way.
- Signature element: the wordmark, where the field flows through the glyphs of **INNER** while **THINK** sits in solid ink. Both words are drawn into one canvas using `destination-in` compositing, so they cannot drift apart and there is no paper-overlay seam at any window size.

Type: Bricolage Grotesque (display) · IBM Plex Sans (body) · IBM Plex Mono (every label, chip and figure).

### Sizing

There is no fixed stage and no transform scaling. `--u` is one design pixel against a 1600×900 reference:

```css
--u: max(0.46px, min(0.0625vw, 0.11111vh));
```

Every dimension is `calc(N * var(--u))`, so type scales with the smaller axis while flex/grid layout absorbs the rest. Small text carries a `max(Npx, …)` floor so footnotes stay readable in a short window. Slides are flex columns; bottom-pinned elements use `margin-top:auto`, never absolute offsets.

### Adding to the deck

Ask what is language (paper, mono chips, ink) and what is thought (aperture, ramp, heatmap), and put chroma only in the second. Don't introduce a second accent or a flat gradient outside an aperture. Size in `var(--u)`, not pixels.

## Where the numbers come from

CODI = *Compressing Chain-of-Thought into Continuous Space via Self-Distillation*, Shen et al., EMNLP 2025 — [paper](https://arxiv.org/abs/2502.21074) · [ACL](https://aclanthology.org/2025.emnlp-main.36/) · [code](https://github.com/zhenyi4/codi) · [checkpoint](https://huggingface.co/cds-jb/codi_qwen3-8b-answer_only).

| Claim on the slides | Source |
|---|---|
| 6 latent tokens; LoRA r=128, α=32 on q,k,v,o,gate,up,down | model card |
| GSM8k-Aug-NL + CommonsenseQA ≈388k; 5 epochs, 8×H100, lr 2e-4 | model card |
| 49.3% latent vs 49.9% verbalised (Qwen3-8B, answer-only) | model card |
| 43.7% at GPT-2 scale, +28.2 pts over Coconut; 55.6% on Llama-3.2-1B | paper |
| 3.1× compression, 2.7–5.9× speed-up | paper |
| 0.35 B of 8.2 B trainable (4.3%) | computed from the Qwen3-8B config: 4096 hidden, 36 layers, 32/8 GQA heads, 12288 FFN, at rank 128 |

Two honesty constraints the deck already states in footnotes, and that must survive any edit:

- The model rows on slide 04 use **different GSM8k variants and setups** and are not directly comparable; each is read against its own verbalised baseline.
- The 3.1s → 0.9s decode on slide 03 is an **illustration of a single item**, not a measurement from this repo. The token counts *are* real — the counter counts exactly the words rendered on screen. If you measure real latency, replace `A_END`/`B_END` in `runS3()` and drop the caveat.
