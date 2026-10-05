#!/usr/bin/env bash
# End-to-end deployment smoke through the public URL (the Caddy proxy).
#
#   deploy/smoke.sh cpu   # stack from docker-compose.yml
#   deploy/smoke.sh gpu   # stack from docker-compose.yml + docker-compose.gpu.yml
#
# BASE_URL defaults to https://$DOMAIN when DOMAIN is set (also read from ./.env),
# else http://localhost:${HTTP_PORT:-80}.
#
# gpu mode additionally requires /api/health status "ready", runs
# scripts/gpu_smoke.py INSIDE the backend container (one real inference per
# provider, report kept in the state volume), then sends one live VQA and one
# live grounding request through the proxy. Run it on a freshly (re)started
# backend: models load lazily and stay resident, and gpu_smoke loads its own
# copies in child processes, so a backend that already holds Qwen would have to
# share VRAM with a second copy. The live API calls at the end double as the
# demo warm-up (the backend keeps both models loaded afterwards).
#
# Needs: docker compose, curl, python3 (JSON checks) on the host.
set -euo pipefail

MODE="${1:-cpu}"
case "$MODE" in cpu|gpu) ;; *) echo "usage: $0 [cpu|gpu]" >&2; exit 2 ;; esac

cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi

COMPOSE=(docker compose -f docker-compose.yml)
[ "$MODE" = gpu ] && COMPOSE+=(-f docker-compose.gpu.yml)

if [ -n "${BASE_URL:-}" ]; then :
elif [ -n "${DOMAIN:-}" ]; then BASE_URL="https://${DOMAIN}"
else BASE_URL="http://localhost:${HTTP_PORT:-80}"; fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass() { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*" >&2; exit 1; }
# json FILE PY_EXPR -> prints the expression evaluated against the parsed body `j`
json() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print($2)" "$1"; }

echo "SatQuery deploy smoke ($MODE) against $BASE_URL"

echo "1. /api/health through the proxy"
code=000
for _ in $(seq 1 60); do
  code=$(curl -sS -o "$WORK/health.json" -w '%{http_code}' "$BASE_URL/api/health" || true)
  [ "$code" = 200 ] && break
  sleep 3
done
[ "$code" = 200 ] || fail "/api/health returned HTTP $code (503 = trace unusable; see checks.trace.detail)"
status=$(json "$WORK/health.json" 'j["status"]')
python3 - "$WORK/health.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
for name, check in j["checks"]["capabilities"].items():
    print(f"        {name:18} available={check['available']!s:5} reason={check['reason_code']}")
PY
for cap in change_vqa optical_sar; do
  [ "$(json "$WORK/health.json" "j['checks']['capabilities']['$cap']['available']")" = True ] || fail "$cap unavailable"
done
if [ "$MODE" = gpu ]; then
  [ "$status" = ready ] || fail "status is '$status'; gpu mode needs 'ready' (every capability available)"
fi
pass "health status=$status"

echo "2. frontend through the proxy"
code=$(curl -sS -L -o "$WORK/index.html" -w '%{http_code}' "$BASE_URL/workspace")
[ "$code" = 200 ] && grep -qi '<html' "$WORK/index.html" || fail "/workspace returned HTTP $code"
pass "/workspace HTTP 200"

echo "3. explicit cached replay (no GPU involved)"
code=$(curl -sS -o "$WORK/cached.json" -w '%{http_code}' -X POST "$BASE_URL/api/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"scene_id":"loveda_LoveDA_images_png_0_gsd0.3","question":"Is there a building in this image?","sensor":"optical","execution_mode":"cached_result"}')
[ "$code" = 200 ] || fail "cached replay HTTP $code"
[ "$(json "$WORK/cached.json" 'j["execution_mode"]')" = cached_result ] || fail "cached replay mode"
pass "cached_result answer=$(json "$WORK/cached.json" 'repr(j["answer"])')"

if [ "$MODE" = gpu ]; then
  echo "4. scripts/gpu_smoke.py inside the backend container"
  report="/app/state/smoke/gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ).json"
  if "${COMPOSE[@]}" exec -T backend sh -c "mkdir -p /app/state/smoke && python scripts/gpu_smoke.py --out $report"; then
    pass "gpu_smoke exit 0 (report: $report in volume satquery_satquery-state)"
  else
    "${COMPOSE[@]}" exec -T backend cat "$report" >&2 || true
    fail "gpu_smoke failed; report above"
  fi
  "${COMPOSE[@]}" exec -T backend python -c "import json; r=json.load(open('$report')); print('        runtime:', r['runtime']['gpu_name'], 'torch', r['runtime']['torch'], 'cuda', r['runtime']['cuda_runtime']); [print('       ', c['provider'], 'success=%s latency=%ss' % (c['success'], c['latency_seconds'])) for c in r['cases']]"

  echo "5. live single-image VQA through the proxy (loads Qwen into the backend)"
  code=$(curl -sS -m 300 -o "$WORK/vqa.json" -w '%{http_code}' -X POST "$BASE_URL/api/analyze" \
    -H 'Content-Type: application/json' \
    -d '{"scene_id":"loveda_LoveDA_images_png_0_gsd0.3","question":"Is there a building in this image?","sensor":"optical","capability":"single_image_vqa"}')
  [ "$code" = 200 ] || { cat "$WORK/vqa.json" >&2; fail "live VQA HTTP $code"; }
  [ "$(json "$WORK/vqa.json" 'j["execution_mode"]')" = live ] || fail "VQA did not run live"
  pass "live VQA answer=$(json "$WORK/vqa.json" 'repr(j["answer"])')"

  echo "6. live grounding through the proxy (loads Grounding DINO into the backend)"
  code=$(curl -sS -o "$WORK/scene.json" -w '%{http_code}' -X POST "$BASE_URL/api/scenes" \
    -F "file=@data/demo/grounding/07272.jpg")
  [ "$code" = 201 ] || fail "scene upload HTTP $code"
  scene=$(json "$WORK/scene.json" 'j["scene_id"]')
  code=$(curl -sS -m 300 -o "$WORK/ground.json" -w '%{http_code}' -X POST "$BASE_URL/api/analyze" \
    -H 'Content-Type: application/json' \
    -d "{\"scene_id\":\"$scene\",\"question\":\"A yellow ship\",\"sensor\":\"optical\",\"capability\":\"grounding\"}")
  [ "$code" = 200 ] || { cat "$WORK/ground.json" >&2; fail "live grounding HTTP $code"; }
  [ "$(json "$WORK/ground.json" 'j["execution_mode"]')" = live ] || fail "grounding did not run live"
  pass "live grounding answer=$(json "$WORK/ground.json" 'repr(j["answer"])') boxes=$(json "$WORK/ground.json" 'len(j.get("evidence") or [])')"
fi

echo "7. audit trace chain"
code=$(curl -sS -o "$WORK/verify.json" -w '%{http_code}' -X POST "$BASE_URL/api/traces/verify")
[ "$code" = 200 ] && [ "$(json "$WORK/verify.json" 'j["verified"]')" = True ] || fail "trace verify: $(cat "$WORK/verify.json")"
pass "$(json "$WORK/verify.json" 'j["message"]')"

echo "Deploy smoke ($MODE) passed."
