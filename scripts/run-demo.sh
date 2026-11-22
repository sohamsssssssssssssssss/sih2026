#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python || ! -f frontend/node_modules/next/dist/bin/next ]]; then
  echo 'Install the backend and frontend dependencies first. See README → Run the demo locally.' >&2
  exit 1
fi

npm --prefix frontend run copy-maplibre-worker
unset SATQUERY_ALLOW_MPS
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
export SATQUERY_VQA_MLX=${SATQUERY_VQA_MLX:-1}
export NEXT_TELEMETRY_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1

.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
backend=$!
frontend=
cleanup() {
  if [[ -n "$frontend" ]]; then
    kill "$frontend" 2>/dev/null || true
    wait "$frontend" 2>/dev/null || true
  fi
  kill "$backend" 2>/dev/null || true
  wait "$backend" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

until curl --max-time 1 -sS http://127.0.0.1:8000/api/health >/dev/null 2>&1; do
  if ! kill -0 "$backend" 2>/dev/null || (( SECONDS > 75 )); then
    echo 'The demo backend did not become ready. Check the server output above.' >&2
    exit 1
  fi
  sleep 1
done
(
  cd frontend
  exec node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 --port 3000
) &
frontend=$!

echo 'SatQuery Ask: http://127.0.0.1:3000/ask'
while kill -0 "$backend" 2>/dev/null && kill -0 "$frontend" 2>/dev/null; do sleep 1; done
echo 'A demo server stopped. Check the server output above.' >&2
exit 1
