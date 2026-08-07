#!/usr/bin/env bash
# Smoke-test: add → flush → search against the local EverOS API.
set -euo pipefail

HOST="${EVEROS_API__HOST:-127.0.0.1}"
PORT="${EVEROS_API__PORT:-8001}"
BASE="http://${HOST}:${PORT}"

echo "1) health"
curl -sf "${BASE}/health"
echo

TS=$(($(date +%s) * 1000))
SESSION_ID="innerthink-demo-$(date +%s)"

echo "2) memory/add (session=${SESSION_ID})"
curl -sf -X POST "${BASE}/api/v2/memory/add" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"${SESSION_ID}\",
    \"app_id\": \"innerthink\",
    \"project_id\": \"default\",
    \"messages\": [
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": ${TS}, \"content\": \"I love climbing in Yosemite every spring.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS + 10000)), \"content\": \"My favorite coffee shop is Blue Bottle in SOMA.\"}
    ]
  }"
echo

echo "3) memory/flush"
curl -sf -X POST "${BASE}/api/v2/memory/flush" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"app_id\":\"innerthink\",\"project_id\":\"default\"}"
echo

echo "4) memory/search"
curl -sf -X POST "${BASE}/api/v2/memory/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "innerthink",
    "project_id": "default",
    "query": "Where do I like to climb?",
    "top_k": 5
  }'
echo

echo "done — inspect Markdown under data/everos/ if extraction succeeded"
