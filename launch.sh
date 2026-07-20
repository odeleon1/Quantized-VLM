#!/usr/bin/env bash
# Starts the VLM server, waits for it to be ready, then opens the browser.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

cd "$REPO/backend"
uv run --project "$REPO" uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait up to 30s for the server to answer its unauthenticated health probe.
# /status needs a Bearer token, so polling it always returned 401 and the
# script waited the full 30s every time. /health needs no auth.
for i in $(seq 1 30); do
  sleep 1
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    break
  fi
done

# Open default browser (xdg-open works on JetPack / Ubuntu desktop)
xdg-open http://localhost:8000 2>/dev/null || true

# Keep terminal alive so Ctrl-C kills the server
wait "$SERVER_PID"
