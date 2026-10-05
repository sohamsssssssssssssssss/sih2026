# Deploy on one GPU VM behind one public URL

This runs the whole SatQuery stack (FastAPI backend, Next.js frontend, Caddy
reverse proxy) with Docker Compose on one rented NVIDIA GPU VM, using any
provider. The browser sees one origin: Caddy sends `/api/*` to the backend and
everything else to the frontend.

```
browser ──https──▶ Caddy :80/:443 ─┬─ /api/* ─▶ backend :8000  (uvicorn, 1 worker; GPU)
                                   └─ /*     ─▶ frontend :3000 (next start)
        model artifacts: host dir ──read-only──▶ backend:/models
        audit trace + uploads: named volumes satquery_satquery-state, satquery_satquery-runtime
```

| File | Purpose |
|---|---|
| `Dockerfile.backend` | Targets `cpu` (python:3.11-slim + `backend/requirements.txt`) and `gpu` (+ `deploy/requirements-gpu.txt`) |
| `frontend/Dockerfile` | `npm ci` → `next build` → `next start` |
| `docker-compose.yml` | CPU stack, works on any Docker host |
| `docker-compose.gpu.yml` | Override: `gpus: all`, `gpu` target, read-only `/models` mount |
| `deploy/Caddyfile` | Reverse proxy; automatic HTTPS when `DOMAIN` is set |
| `deploy/provision_models.py` | One-time, explicit download of the artifacts into the host dir, then an offline check |
| `deploy/smoke.sh` | End-to-end check through the public URL (`make deploy-smoke`) |

Make targets: `make docker-cpu`, `make docker-gpu`, `make deploy-smoke`
(`SMOKE_MODE=cpu` for the CPU stack), `make docker-down`.

## What was and was not tested

There was no GPU available when this was written. Read this before demo day.

- **Built and run (CPU, x86_64):** the backend `cpu` image, the frontend image and
  the CPU compose stack behind Caddy. Checked: `/api/health` through the proxy
  (`degraded`, with change/optical-SAR available), the frontend pages, the
  scene image, cached replay, upload, trace verify, `deploy/smoke.sh cpu`, the
  non-root user and read-only code, the trace symlink, and torn-tail recovery
  across restarts.
- **Partially built:** the backend `gpu` target. The `deps-gpu` stage ran to
  completion: pip resolved and installed `backend/requirements.txt` together
  with `deploy/requirements-gpu.txt`, and the `groundingdino-py` sdist built
  without compiling anything. Exporting the final image then failed because
  the build sandbox ran out of disk, so nothing in the finished GPU image has
  been checked, including whether its libraries import. CUDA, real weights,
  `scripts/gpu_smoke.py` and the `gpu` mode of `deploy/smoke.sh` have **not**
  been run. The same library versions passed the Kaggle T4 smoke
  (`docs/gpu-smoke.md`), but not inside this container. The first job on the
  VM is `make docker-gpu`, then `deploy/provision_models.py --verify-only`,
  then `make deploy-smoke`.
- **Not run at all:** `docker-compose.gpu.yml` on a real NVIDIA host,
  `deploy/provision_models.py` (no artifact download was attempted), automatic
  HTTPS with a real `DOMAIN`, and a browser session against the deployed stack.
  The frontend bundle was checked to contain relative `/api/...` URLs.
- The build sandbox had no Debian mirror, so its builds used host copies of
  the `apt-get` packages (`libexpat1`, `libgl1`, `libglib2.0-0`). The `apt-get`
  lines in `Dockerfile.backend` themselves have not been run.

## Facts the setup is based on (from the code)

- **Where artifacts are looked up.** `models/artifacts.py::validate_artifact`
  reads `configs/model_artifacts.json`. For each provider it uses the env var
  `local_path_env` if set, otherwise `default_local_path` (the Kaggle paths),
  otherwise the local Hugging Face cache (`local_files_only=True`). If the env
  var is set and the path is missing, it fails closed and does not fall back.
  - Qwen: `SATQUERY_QWEN_MODEL_DIR` → a directory that contains `config.json`
    (with `model_type: qwen2_5_vl`), `preprocessor_config.json`,
    `tokenizer_config.json` and `tokenizer.json`, plus either a non-empty
    `model.safetensors`/`pytorch_model.bin` or every shard listed in
    `model.safetensors.index.json`.
  - Grounding DINO: `SATQUERY_GROUNDING_CHECKPOINT` → a file that must be named
    exactly `groundingdino_swint_ogc.pth` and must not be empty. The config
    (`GroundingDINO_SwinT_OGC.py`) comes from `SATQUERY_GROUNDING_CONFIG`,
    otherwise from the installed `groundingdino` package
    (`models/grounding_dino/model.py::_resolve_config`). The image uses the
    packaged copy.
- **Hidden third artifact: `bert-base-uncased`.** Grounding DINO's packaged
  config sets `text_encoder_type = "bert-base-uncased"`, and `load_model` calls
  `AutoTokenizer/BertModel.from_pretrained("bert-base-uncased")`. With
  `HF_HUB_OFFLINE=1` (forced in `backend/services.py`), this only works when
  the model is already in the Hugging Face cache. **Neither the manifest nor
  `GroundingDINOModel.readiness()` checks for it.** If it is missing,
  `/api/health` still reports `grounding` as available, and the first grounding
  request then fails. The GPU override points `HF_HUB_CACHE` at
  `/models/huggingface/hub`, and `deploy/provision_models.py` puts it there and
  then verifies it.
- **No compile step for Grounding DINO.** `groundingdino-py==0.4.0` (the
  Kaggle-verified version) is pure Python. Its `setup.py` has the custom CUDA-op
  build commented out, and `MSDeformAttn.forward` always calls
  `multi_scale_deformable_attn_pytorch`. So no `nvcc` or CUDA toolkit is
  needed. The official `IDEA-Research/GroundingDINO` source does compile
  `groundingdino._C`. Do not swap it in without adding a `-devel` CUDA build
  stage.
- **CUDA comes from the wheels.** The PyPI `torch==2.10.0` Linux x86_64 wheel is
  the CUDA 12.8 build (it depends on `nvidia-*-cu12==12.8.*`), which is the same
  build as Kaggle's `2.10.0+cu128`. So the `gpu` image is `python:3.11-slim`
  plus those wheels, not an `nvidia/cuda` base. An `nvidia/cuda` base would
  carry a second copy of the CUDA libraries and lacks Python 3.11 (the version
  `.python-version` declares). The host's driver (`libcuda`) is injected by the
  NVIDIA Container Toolkit. Host requirement: an NVIDIA driver that supports
  CUDA 12.8 (R570+). Older R525+ drivers may work through CUDA minor-version
  compatibility, but that is untested.
- **Runtime state.** `orchestrator/trace.py` hard-codes
  `TRACE_PATH = <repo root>/trace.jsonl` (`/app/trace.jsonl`), and
  `backend/services.py` writes uploads under `<repo root>/data/runtime/`. A
  named volume cannot target one file, so the image ships `/app/trace.jsonl` as a
  symlink to `/app/state/trace.jsonl` and mounts the `satquery-state` volume at
  `/app/state`. `data/runtime` is a second volume. Crash recovery writes
  `trace.jsonl.torn-*` next to the symlink, in the container layer. The
  entrypoint (`deploy/backend-entrypoint.sh`) moves these files into
  `/app/state/quarantine/` on the next start.
- **One worker only.** The trace hash chain is held in process memory behind a
  thread lock. Running more than one uvicorn worker would fork the audit chain.
- **Health semantics.** `/api/health` returns 200 for `ready` and `degraded` and
  503 for `unavailable` (trace unusable). The Docker healthcheck fails only on
  503 or no response, so a CPU stack (`degraded`) counts as healthy.
  `deploy/smoke.sh gpu` requires `ready`.
- **Frontend API base.** `frontend/lib/api.ts` uses
  `process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`. `??` falls back
  only for `undefined`/`null`, so building with `NEXT_PUBLIC_API_URL=""` gives
  relative `/api/...` requests to the same origin, including the MapLibre image
  source and `<img>` URLs. The value is inlined at **build** time, so leave it
  empty. Setting it on the running container has no effect. Every API caller is a
  client component, so there are no server-side relative fetches.
- **CORS.** With one origin, CORS is not involved. `SATQUERY_CORS_ORIGINS`
  (comma-separated exact origins, appended to the localhost defaults; `*`,
  paths and other malformed values stop the backend at startup) is only needed
  if the UI is served from a different origin than the API.

## 1. Pick and prepare the VM

- **GPU:** one NVIDIA T4 (16 GB), L4 (24 GB) or A10/A10G (24 GB). Estimate
  from parameter counts (not measured): Qwen2.5-VL-3B in fp16 needs about 7.5 GB
  for weights, and Grounding DINO Swin-T plus BERT-base in fp32 need about
  1 GB. The backend loads both lazily and keeps them resident, so expect
  roughly 9–11 GB of VRAM including activations and the CUDA context. A T4 fits
  that. Smaller GPUs (8–12 GB) are not supported. The code uses fp16 and never
  bf16, so a T4 is fine.
- **CPU/RAM/disk:** at least 4 vCPU and 16 GB RAM (Qwen is staged through host
  memory while it loads). At least 80 GB of disk. The GPU venv alone is
  about 7 GB, and a build holds several copies of it in cache layers (a 40 GB
  sandbox ran out of disk). The artifacts take about 9 GB more. After a
  build, `docker builder prune` reclaims the cache.
- **OS:** Ubuntu 22.04 or 24.04. Many providers sell "GPU/Deep Learning" images
  with the driver preinstalled.
- **Network:** open inbound TCP 80 and 443 only (and UDP 443 for HTTP/3 if
  you want it). Keep 8000/3000 closed, because the compose file only `expose`s
  them on the internal network.

Install the host prerequisites (once), following the vendor docs:

1. The NVIDIA driver (R570+), then check it with `nvidia-smi`.
2. Docker Engine with the Compose plugin v2.30 or newer
   (`docker compose version`). `gpus: all` needs 2.30+. On older Compose, replace
   it with `deploy.resources.reservations.devices`, as the comment in
   `docker-compose.gpu.yml` describes.
3. The NVIDIA Container Toolkit, then
   `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
4. Check that containers can reach the GPU:
   `docker run --rm --gpus all ubuntu nvidia-smi`.

## 2. Get the code and configure

```sh
git clone https://github.com/sohamsssssssssssssssss/sih2026.git && cd sih2026
git checkout <the commit you are deploying>
cat > .env <<'EOF'
# Public host name with a DNS A record pointing at this VM -> automatic HTTPS.
# Leave empty to serve plain HTTP on port 80 (for example an IP-only test).
DOMAIN=
SATQUERY_MODELS_DIR=/srv/satquery/models
# Leave empty: UI and API share one origin through Caddy.
NEXT_PUBLIC_API_URL=
SATQUERY_CORS_ORIGINS=
EOF
```

`.env` is not ignored by git, so do not commit it. It holds no secrets.

## 3. Build the GPU image and provision model artifacts (offline afterwards)

```sh
sudo mkdir -p /srv/satquery/models && sudo chown "$USER" /srv/satquery/models
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend
docker run --rm --user "$(id -u):$(id -g)" \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v /srv/satquery/models:/models \
  satquery-backend:gpu python deploy/provision_models.py --dest /models
```

This is the only step that uses the network for weights. It writes:

```
/srv/satquery/models/
  qwen2.5-vl-3b-instruct/        # Qwen/Qwen2.5-VL-3B-Instruct snapshot   -> SATQUERY_QWEN_MODEL_DIR
  groundingdino_swint_ogc.pth    # ShilongLiu/GroundingDINO checkpoint     -> SATQUERY_GROUNDING_CHECKPOINT
  huggingface/hub/models--bert-base-uncased/   # Grounding DINO text encoder -> HF_HUB_CACHE
  PROVENANCE.json                # resolved commit SHAs + SHA-256 digests
```

The script then re-runs `models.artifacts.validate_artifact` for both providers
and loads `bert-base-uncased`, all offline and with the same env vars the
service uses. It exits non-zero on any failure. To re-check later, run
`... python deploy/provision_models.py --dest /models --verify-only`.

Notes:

- `configs/model_artifacts.json` has `revision: null` and `sha256: null`.
  The repository has never pinned trusted versions. The script resolves the
  current `main` (or the commit you pass with `--qwen-revision`,
  `--dino-revision` or `--bert-revision`) and records the SHAs in
  `PROVENANCE.json`. Copy them into the deployment notes. This records where
  the weights came from but does not verify them.
- If the VM must never reach Hugging Face, run the same command on any machine
  with network access and copy the directory over (`rsync -a`). Keep the
  layout. Files must be readable by the container user (uid 10001):
  `chmod -R a+rX /srv/satquery/models`.
- The directory is mounted read-only. Weights are never copied into an image
  (`.dockerignore` also excludes `*.pth`, `*.safetensors` and `*.bin`).

## 4. Start the stack

```sh
make docker-gpu      # = docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
```

Caddy waits for the backend healthcheck. With `DOMAIN` set, the first start
requests a certificate, so ports 80 and 443 must be reachable from the
internet. Check progress with `docker compose logs proxy`.

## 5. Prove live inference and full availability

```sh
make deploy-smoke    # = deploy/smoke.sh gpu
```

Run this right after `make docker-gpu`, or after
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart backend`.
The backend loads models lazily, and step 4 starts `scripts/gpu_smoke.py`
processes that load their own copies. If the backend already holds Qwen, the
two copies compete for VRAM.

The script checks the following, all through the public URL unless noted:

1. `GET /api/health` returns 200 with `"status": "ready"`. Every capability
   (`single_image_vqa`, `grounding`, `change_vqa`, `optical_sar`) must show
   `available: true`. If one shows `reason_code` `CUDA_UNAVAILABLE`,
   `DEPENDENCY_UNAVAILABLE`, `ARTIFACT_UNAVAILABLE` or
   `ARTIFACT_CONFIG_INVALID`, check the GPU mount or the artifact layout.
2. `/workspace` is served by the frontend.
3. Explicit cached replay returns `execution_mode: cached_result`.
4. **Inside the container:** `python scripts/gpu_smoke.py --out /app/state/smoke/gpu-smoke-<UTC>.json`.
   This runs one real inference per provider in separate processes, as
   described in `docs/gpu-smoke.md`. The report stays in the state volume.
5. A live single-image VQA request runs with `execution_mode: live`.
6. A live grounding request runs on the uploaded `data/demo/grounding/07272.jpg`
   with "A yellow ship".
7. `POST /api/traces/verify` returns `verified: true`.

To run only the in-container smoke:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec backend \
  python scripts/gpu_smoke.py --out /app/state/smoke/manual.json
```

After steps 5–6 the backend keeps both models resident, so the first demo
query is not a cold load. The first live request after any backend restart
takes tens of seconds (Kaggle T4: about 34 s Qwen, 19 s DINO, including load).
`MODEL_EXECUTION_TIMEOUT_SECONDS` is 120 s, and Caddy sets no upstream timeout.

## CPU-only stack (any machine, no GPU)

```sh
make docker-cpu                     # http://localhost (or HTTP_PORT=8080 make docker-cpu)
make deploy-smoke SMOKE_MODE=cpu
```

`/api/health` is `degraded`. Qwen and Grounding report
`DEPENDENCY_UNAVAILABLE` and fail closed with 503, which is the documented
behaviour. Change and optical-SAR analysis, uploads and explicit cached replay
all work.

## Operations

- **Logs:** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f backend`.
- **Audit trace backup:**
  `docker compose ... cp backend:/app/state ./state-backup-$(date +%F)`.
  The trace is append-only and hash-chained. Never edit it. A torn final line
  from a crash is quarantined automatically into `/app/state/quarantine/`. Any
  other damage makes `/api/health` return 503 (fail closed). Keep a copy as
  evidence before rotating it (see the README troubleshooting table).
- **Reset runtime state** (deletes the uploads and the trace):
  `docker compose ... down -v`. Without `-v`, volumes survive `down`/`up`
  and rebuilds.
- **Update:** `git pull && make docker-gpu`, then `make deploy-smoke` again.
- **Different UI origin:** only if the frontend is hosted elsewhere, set
  `SATQUERY_CORS_ORIGINS=https://ui.example.org` and rebuild the frontend with
  `NEXT_PUBLIC_API_URL=https://api.example.org`.

## Demo-day checklist

- [ ] The VM is up, and `nvidia-smi` shows the GPU with no stray processes using
      VRAM.
- [ ] `git log -1` on the VM shows the commit you rehearsed with.
      `PROVENANCE.json` matches the rehearsal.
- [ ] `make docker-gpu` was run after the last pull. `docker compose ... ps` shows
      the backend `healthy` and the proxy `Up`.
- [ ] `docker compose ... restart backend && make deploy-smoke` passes: `ready`,
      gpu_smoke exits 0, live VQA and grounding both return `execution_mode: live`,
      and the trace verifies.
- [ ] The models are warm. Do not restart the backend after the smoke; if you
      must, run one live query yourself before presenting.
- [ ] The public URL loads in a fresh browser profile over HTTPS with no
      certificate warning. `/workspace`, `/executions` and `/system` all
      render, and `/system` shows every capability available.
- [ ] Back up `/app/state`, so the audit trail shown on stage is preserved.
- [ ] Fallback plan: if the GPU path fails, explicit cached replay still
      works, and the UI and backend label it `cached_result`. Never present
      it as live.
- [ ] Shut down or snapshot the VM afterwards, because GPU instances bill by the hour.
