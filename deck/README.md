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
| `r` | replay the current slide's animation |
| `f` | fullscreen |

The current slide is mirrored into the URL hash, so `#3` deep-links to slide 03. The deck fills any browser window; there is no fixed stage.

## The four slides

The hackathon theme is *save tokens*, and the deck is built as one argument: here is the bill → here is where the waste comes from → here is how you train it away.

1. **Identity** — wordmark, the one-line thesis, three spec cards.
2. **The token bill** — a live chat turn, with the token counter as the hero: a 58u display number over a full-width meter, pink on the left because it is money burning. A reasoning model spends 109 tokens of thinking on a trivial question and replies with one line; then the pane splits and InnerThink runs the same turn in the *same chat shape* — question, a "thinking silently…" row, the reply — so the only difference on screen is that its thinking never becomes text. Ends on the bill: 117 → 14, −88%. Both meters share one scale, so the right-hand stub is directly comparable to the full left bar; keep that if you change the numbers.
3. **Where the thinking lives** — the neural-net diagram. In explicit mode a token chip physically *leaves* the aperture, arcs across the paper and re-enters as input, and the emitted-token counter climbs. In latent mode the loop runs from the last hidden layer back to the first, entirely inside the dark, and the counter stays at 0.
4. **How it learns** — self-distillation. One network, two branches; the teacher's words and the student's vectors are aligned at the token before the answer. The closing beat dissolves the teacher branch and replaces it with "Same answers. None of the words."

Slides 02–04 auto-play on entry and slides 03–04 loop; `r` restarts the current one.

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
| 3.1× compression, 2.7–5.9× speed-up | paper |
| the ⟨bot⟩ / 6 latent steps / ⟨eot⟩ loop and the L1-on-hidden-state objective | paper §3 |
| 0.35 B of 8.2 B trainable (4.3%) | computed from the Qwen3-8B config: 4096 hidden, 36 layers, 32/8 GQA heads, 12288 FFN, at rank 128 |

Three honesty constraints the deck states in footnotes, and that must survive any edit:

- Slide 02 is an **illustrative single turn**, not a logged run. The 109 thinking tokens are honest in the only sense that matters — the counter counts exactly the words rendered on screen — and a real Qwen3-8B trace on that question is typically *longer*, not shorter. The 14-token latent side is 6 latent steps + 8 answer tokens. If you log a real turn, replace `THINK` and `ANSWER_TOKENS` in `runS2()` and drop the caveat.
- Slide 03's **layer counts are illustrative**: 4 hidden layers of 5 units stands in for 36 transformer layers, and the diagram's recurrence stands in for the last hidden state passed through the 2-layer MLP + LayerNorm.
- Slide 04's "27 tokens" for the teacher branch is the length of the three reasoning chips shown, matching the emitted-token count on slide 03. Keep those two numbers in sync if you change either.
