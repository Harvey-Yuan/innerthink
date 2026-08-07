#!/usr/bin/env bash
# Serve the InnerThink deck on localhost.
# Only ./deck is exposed — the project root holds .env, keep it off the wire.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8080}"
echo "InnerThink deck → http://localhost:${PORT}/"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory deck
