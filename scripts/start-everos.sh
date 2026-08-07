#!/usr/bin/env bash
# Start the local EverOS memory server for this repo.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EVEROS_BIN="${ROOT_DIR}/.venv/bin/everos"
MEMORY_ROOT="${ROOT_DIR}/data/everos"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -x "$EVEROS_BIN" ]]; then
  echo "error: everos not found at $EVEROS_BIN" >&2
  echo "Run: uv sync" >&2
  exit 1
fi

if [[ ! -f "${MEMORY_ROOT}/everos.toml" ]]; then
  echo "Initializing EverOS memory root at ${MEMORY_ROOT} ..."
  mkdir -p "$MEMORY_ROOT"
  if [[ -f "${ROOT_DIR}/config/everos.toml.example" ]]; then
    cp "${ROOT_DIR}/config/everos.toml.example" "${MEMORY_ROOT}/everos.toml"
    cp "${ROOT_DIR}/config/ome.toml.example" "${MEMORY_ROOT}/ome.toml"
  else
    "$EVEROS_BIN" init --root "$MEMORY_ROOT"
  fi
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "error: no .env found — run: cp .env.example .env" >&2
  exit 1
fi

export EVEROS_ROOT="$MEMORY_ROOT"

missing=()
[[ -z "${EVEROS_LLM__API_KEY:-}" ]] && missing+=("EVEROS_LLM__API_KEY")
[[ -z "${EVEROS_EMBEDDING__API_KEY:-}" ]] && missing+=("EVEROS_EMBEDDING__API_KEY")
[[ -z "${EVEROS_RERANK__API_KEY:-}" ]] && missing+=("EVEROS_RERANK__API_KEY")
if ((${#missing[@]})); then
  echo "error: EverOS refuses to start without provider keys." >&2
  echo "Fill these in .env: ${missing[*]}" >&2
  echo "  OpenRouter → EVEROS_LLM__API_KEY (and optionally EVEROS_MULTIMODAL__API_KEY)" >&2
  echo "  DeepInfra  → EVEROS_EMBEDDING__API_KEY + EVEROS_RERANK__API_KEY" >&2
  echo "Meanwhile you can run: uv run everos demo --plain" >&2
  exit 1
fi

# LanceDB opens many segment files under concurrent search + indexing.
ulimit -n 4096 2>/dev/null || true

export EVEROS_API__PORT="${EVEROS_API__PORT:-8001}"

echo "starting everos (root=${MEMORY_ROOT}) on ${EVEROS_API__HOST:-127.0.0.1}:${EVEROS_API__PORT}"
exec "$EVEROS_BIN" server start --root "$MEMORY_ROOT"
