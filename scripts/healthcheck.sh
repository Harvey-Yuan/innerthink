#!/usr/bin/env bash
# Health-check the local EverOS server.
set -euo pipefail

HOST="${EVEROS_API__HOST:-127.0.0.1}"
PORT="${EVEROS_API__PORT:-8001}"
URL="http://${HOST}:${PORT}/health"

echo "GET ${URL}"
curl -sf "$URL"
echo
