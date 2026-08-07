# InnerThink — EverOS Memory Layer

Local-first [EverOS](https://github.com/EverMind-AI/EverOS) memory service for this repo: Markdown + SQLite + LanceDB, no Mongo/Elasticsearch/Milvus/Redis.

## Prerequisites

- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))
- API keys for the real memory pipeline:
  - [OpenRouter](https://openrouter.ai/) → `EVEROS_LLM__API_KEY` / `EVEROS_MULTIMODAL__API_KEY`
  - [DeepInfra](https://deepinfra.com/) → `EVEROS_EMBEDDING__API_KEY` / `EVEROS_RERANK__API_KEY`

Without keys you can still run the educational CLI demo (`everos demo`); `/api/v2/memory/*` needs keys.

## Setup

```bash
uv sync
cp .env.example .env
# edit .env and paste the four API keys
```

First start also creates `./data/everos` (gitignored) from `config/*.example` if needed.

## Start the server

EverOS validates LLM/embedding/rerank keys at startup — empty keys will exit before bind.

```bash
./scripts/start-everos.sh
```

Default bind: `http://127.0.0.1:8000` (loopback only; EverOS has no built-in auth).

In another terminal:

```bash
./scripts/healthcheck.sh
# → {"status":"ok"}
```

## First memory loop

With the server running and keys configured:

```bash
./scripts/demo-memory.sh
```

This calls `/api/v2/memory/add` → `/flush` → `/search`. Extracted memories land as Markdown under `data/everos/`.

## Educational demo (no server / no keys)

```bash
uv run everos demo --plain
```

## Layout

| Path | Purpose |
|------|---------|
| `pyproject.toml` / `uv.lock` | Dependencies (`everos`) |
| `.env.example` | Env template (committed) |
| `.env` | Secrets (gitignored) |
| `config/*.example` | Bootstrap `everos.toml` / `ome.toml` |
| `data/everos/` | Runtime memory root (gitignored) |
| `scripts/start-everos.sh` | Start API |
| `scripts/healthcheck.sh` | `GET /health` |
| `scripts/demo-memory.sh` | add → flush → search |

## Notes

- Memory root is project-local (`EVEROS_ROOT=./data/everos`), not `~/.everos`.
- Raise `ulimit -n` if LanceDB hits too-many-open-files (the start script tries `4096`).
- Do not bind `0.0.0.0` without your own auth gateway in front.
