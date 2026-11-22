#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
case "$*" in
  "") frontend_command=start ;;
  --dev) frontend_command=dev ;;
  *) echo 'Usage: bash scripts/run-demo.sh [--dev]' >&2; exit 2 ;;
esac

if [[ ! -x .venv/bin/python || ! -f frontend/node_modules/next/dist/bin/next ]]; then
  echo 'Install the backend and frontend dependencies first. See README → Run the demo locally.' >&2
  exit 1
fi

require_free_ports() {
  if lsof -nP -iTCP:3000 -iTCP:8000 -sTCP:LISTEN; then
    echo 'Ports 3000 and 8000 must be free. Stop the existing servers before launching the demo.' >&2
    exit 1
  fi
}

require_free_ports
unset SATQUERY_ALLOW_MPS
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
export SATQUERY_VQA_MLX=${SATQUERY_VQA_MLX:-1}
export NEXT_TELEMETRY_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ "$frontend_command" == start ]]; then
  npm --prefix frontend run build
  if [[ ! -s frontend/.next/BUILD_ID ]]; then
    echo 'Production build missing frontend/.next/BUILD_ID. Check the build output above.' >&2
    exit 1
  fi
else
  npm --prefix frontend run copy-maplibre-worker
fi
require_free_ports

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

backend_started=$SECONDS
until curl --max-time 1 -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; do
  if ! kill -0 "$backend" 2>/dev/null || (( SECONDS - backend_started > 75 )); then
    echo 'The demo backend did not become ready. Check the server output above.' >&2
    exit 1
  fi
  sleep 1
done
(
  cd frontend
  exec node node_modules/next/dist/bin/next "$frontend_command" --hostname 127.0.0.1 --port 3000
) &
frontend=$!

frontend_started=$SECONDS
until curl --max-time 2 -fsS http://127.0.0.1:3000/ask >/dev/null 2>&1; do
  if ! kill -0 "$frontend" 2>/dev/null || ! kill -0 "$backend" 2>/dev/null || (( SECONDS - frontend_started > 60 )); then
    echo 'The demo frontend did not become ready. Check the server output above.' >&2
    exit 1
  fi
  sleep 1
done

echo 'SatQuery Ask: http://127.0.0.1:3000/ask'
while kill -0 "$backend" 2>/dev/null && kill -0 "$frontend" 2>/dev/null; do sleep 1; done
echo 'A demo server stopped. Check the server output above.' >&2
exit 1
