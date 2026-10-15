# SatQuery Backend V2 — Release Documentation

## External Immutable Artifact Contract

The Kosi 2024 SAR change bundle is an **externally provisioned, immutable artifact pair** that is never built, regenerated, or modified by the container image or the application at runtime.

**Artifact pair (read-only, host-mounted):**
```
/srv/satquery/change-artifacts/kosi-2024/
  sar_change.tif   13,007,495 bytes  SHA256: 7d63525d8851c695c9f1e5e72719b80aebd3e6d18581e875bc60ba5d28f6d3a4
  sar_change.json  2,838 bytes       SHA256: 792492a47b18111d0aa92a6d613ec3ec0f60a6de557e1ddfec4c597a507d2747
```

**Contract guarantees:**
- Both files are verified on every API read (`backend/sar_change_artifacts.py::verified_change_artifacts`)
- Missing, corrupt, or unreadable files fail closed with explicit API status codes even if general health remains healthy:
  - `404` — artifact pair missing from `SATQUERY_KOSI_ARTIFACT_DIR`
  - `409` — SHA256 checksum mismatch against the pinned contract
  - `422` — invalid artifact metadata
  - `503` — artifact read failure (I/O error or unreadable file)
- The application cannot write through this mount (read-only bind mount)
- Replacement requires staging a complete new directory, verifying both hashes, updating `SATQUERY_KOSI_ARTIFACT_DIR`, and recreating the backend during an approved maintenance window
- Different artifact bytes require an explicitly reviewed change to the backend's pinned contract; do not simply update the manifest or overwrite the existing pair

**Input provenance (recorded in `sar_change.json`):**
```
BEFORE: S1A_IW_GRDH_1SDV_20240924T122146_20240924T122211_055807_06D1A1_686F_COG.SAFE
        SHA256: 973b026d6f19eaa469647eaa23647ab89195512769c1fe23796e9ede59a76af5
        Acquired: 2024-09-24T12:21:46Z
AFTER:  S1A_IW_GRDH_1SDV_20241006T122147_20241006T122212_055982_06D887_1753_COG.SAFE
        SHA256: 643092f78815e0bd735cf76e067b5075bc7fceca97d34043ce1443d378084a8c
        Acquired: 2024-10-06T12:21:47Z
```

**Processing formula (pinned in contract):**
```
VV_change_dB = 10 * (log10(VV_after) - log10(VV_before))
VH_change_dB = 10 * (log10(VH_after) - log10(VH_before))
Intermediate precision: float64
Output dtype: float32
Nodata: -9999.0
Grid: EPSG:32645, 749×1649, 10 m resolution
```

---

## Historical Runner Fingerprint Caveat

The preserved `sar_change.json` records **code fingerprints** of the three components that participated in the original offline computation:

| Component | Historical Fingerprint (from preserved artifact) | Current Fingerprint (working tree) |
|-----------|---------------------------------------------------|-------------------------------------|
| `backend/sentinel1.py` | `c58a6b0dc6bec62ae29d4ee0e12a1a35c28dba341be07ed9fd8fb70e9c770455` | `c58a6b0dc6bec62ae29d4ee0e12a1a35c28dba341be07ed9fd8fb70e9c770455` ✓ |
| `models/change/sar_water.py` | `b9d23a64347bde0b40b1a9f84b87b9c6d8cbe92eb9ec1dbca38b15dfe28a5a5b` | `b9d23a64347bde0b40b1a9f84b87b9c6d8cbe92eb9ec1dbca38b15dfe28a5a5b` ✓ |
| `scripts/s1_offline_change.py` | `ccc38d0578d5e5e90b9616273863f491e1a80a9fccea73e737bcc0b0ea15b1f1` | `c77b83b0add857b11b70bd0f1752818a78e74d6061cb6f73884320c39d59641d` ✗ |

**Finding:** The historical implementation of `scripts/s1_offline_change.py` **was not recovered from the sources inspected**. This is a statement about the inspected sources only; it does not establish that the source cannot be recovered anywhere (untracked backups, other machines, or offline media were not searched):
- Git history — the file was never committed (untracked in working tree)
- Local files — only the current refactored version was found
- Previously generated evidence — the preserved `sar_change.json` records only the historical fingerprint, not the source

**Current runner identity:** The working-tree `scripts/s1_offline_change.py` is a refactored version that:
- Uses the same production validators (`backend/sentinel1.py` internals)
- Computes the same VV/VH log-ratio formula in float64 intermediate precision
- Produces the same three-band GeoTIFF with identical nodata convention
- Has been regression-tested against synthetic inputs with known log-ratio values

**Numerical equivalence:** The current runner was **independently verified** to reproduce the preserved `sar_change.tif` byte-for-byte (identical SHA256) when run against the pinned input archives. A regenerated `sar_change.json` is **not byte-identical** to the preserved artifact: it embeds the current runner fingerprint `c77b83b0add857b11b70bd0f1752818a78e74d6061cb6f73884320c39d59641d` instead of the historical `ccc38d0578d5e5e90b9616273863f491e1a80a9fccea73e737bcc0b0ea15b1f1`, and it records the archive paths of the regeneration invocation, which differ from the archive paths captured in the preserved artifact. The scientific content (formula, grid, nodata, checksums, provenance product IDs) is unchanged.

**Limits of exact historical reproducibility:**
- The exact historical source of `scripts/s1_offline_change.py` was not recovered from the inspected sources (Git history, local files, preserved artifact metadata); recovery from sources not inspected is not ruled out
- The historical fingerprint `ccc38d0578d5e5e90b9616273863f491e1a80a9fccea73e737bcc0b0ea15b1f1` is recorded in the immutable artifact metadata but was not mapped to a source file in the inspected sources
- The current runner `c77b83b0add857b11b70bd0f1752818a78e74d6061cb6f73884320c39d59641d` is functionally equivalent for the pinned inputs but is not the historical implementation
- Any future regeneration would use the current runner, not the historical one

---

## What Was Independently Verified

| Check | Status | Evidence |
|-------|--------|----------|
| Input archive SHA256 matches pinned values | ✅ Verified | `scripts/s1_offline_change.py` enforces checksums |
| Production TAR parsing & validation | ✅ Verified | Reuses `backend/sentinel1.py` validators |
| Grid contract (EPSG:32645, 749×1649, 10m) | ✅ Verified | Enforced by `sentinel1.Grid` validation |
| VV/VH log-ratio formula (float64→float32) | ✅ Verified | Synthetic regression tests in `scripts/test_s1_offline_change.py` |
| Validity mask criteria (both masks=1, finite, positive VV/VH) | ✅ Verified | Unit tests cover NaN, Inf, zero, negative |
| Output GeoTIFF SHA256 matches preserved artifact | ✅ Verified | Raster bytes bit-identical to preserved artifact |
| Output JSON metadata structure & values | ⚠️ Verified with expected differences | Structure and scientific fields match preserved `sar_change.json`; a regenerated JSON is **not byte-identical** — runner fingerprint and archive paths differ |
| Artifact verification on every API read | ✅ Verified | `backend/sar_change_artifacts.py`: 404 missing / 409 checksum mismatch / 422 invalid metadata / 503 read failure |
| Docker build context hygiene | ⚠️ Static inspection only | `.dockerignore` excludes local/generated content and re-includes tracked `data/eval/questions.v1.jsonl`; not confirmed by a container run |

---

## What Remains Scientifically Unvalidated

| Aspect | Status | Caveat |
|--------|--------|--------|
| Flood interpretation accuracy | ❌ Unvalidated | "Measured SAR backscatter change; flood interpretation unvalidated." |
| Historical CDSE Process request parameters | ❌ Unvalidated | Reconstructed from captured TAR's `userdata.json`; not independent request logs |
| Pre-breach baseline validity | ❌ Unvalidated | September 24 is a pre-breach observation, not a confirmed dry baseline |
| Speckle filter / calibration / orthorectification settings | ⚠️ Reconstructed | Nominal: GAMMA0_TERRAIN, COPERNICUS_30 DEM, LEE 5×5, IW, DV, HIGH, ORBIT |
| Generalization to other events/geometries | ❌ Unvalidated | Only Kosi 2024 pair has been processed and preserved |

---

## Docker Build Context Hygiene

**Updated `.dockerignore` exclusions (backend build context = repository root):**

**Added exclusions:**
- `.venv-stage1/` and `**/.venv-stage1/` — local Python virtual environment
- `packages/` and `**/packages/` — local package cache
- `**/.cache/`, `**/.mypy_cache/`, `**/.ruff_cache/` — tool caches
- `**/node_modules/`, `**/*.egg-info/`, `**/dist/`, `**/build/`, `**/*.whl` — build artifacts
- `*.log`, `*.tmp`, `*.temp` — temporary files
- `**/.env`, `**/.env.*`, `**/credentials/`, `**/secrets/` — credentials and environment files
- `**/probe-captures/`, `**/captures/`, `**/artifacts/`, `**/outputs/` — runtime artifacts
- `**/*.tif`, `**/*.tiff`, `**/*.jsonl`, `**/*.sqlite*`, `**/*.db` — data files
- `!data/eval/questions.v1.jsonl` — **precise exception** (Loop 35) re-including the tracked evaluation questions that the broad `**/*.jsonl` rule would otherwise exclude; runtime traces (`trace.jsonl`, `trace.jsonl.*`, `**/trace.jsonl*`) and generated JSONL remain excluded

**Preserved build inputs (NOT excluded):**
- `data/eval/questions.v1.jsonl` — tracked evaluation question set, kept in the backend Docker build context by the exception above
- `backend/` — application source (requirements.txt, routes, services, sentinel1, etc.)
- `models/` — model code (change/sar_water.py, qwen_vl/, grounding_dino/)
- `deploy/` — deployment scripts (backend-entrypoint.sh, requirements-gpu.txt, provision_models.py)
- `configs/` — model artifact configuration
- `scripts/` — operational scripts (s1_offline_change.py, probe_s1_provenance.py, gpu_smoke.py)
- `orchestrator/` — trace/audit infrastructure

---

## Checks Performed

| Check | Command | Result |
|-------|---------|--------|
| Docker ignore-rule validation | Static configuration inspection of `.dockerignore` against the `COPY` statements in `Dockerfile.backend` (no `docker build`, no daemon, no container rehearsal — **no successful container rehearsal has occurred**) | ✅ Required inputs not excluded; tracked `data/eval/questions.v1.jsonl` preserved |
| Required build-input inclusion | Manual inspection of `COPY` statements in `Dockerfile.backend` | ✅ All `COPY` sources covered by non-excluded paths |
| Documentation consistency | Cross-referenced `.dockerignore`, `Dockerfile.backend`, `docs/deploy.md`, `backend/sar_change_artifacts.py` | ✅ Consistent |
| git diff --check | `git diff --check` | ✅ No whitespace errors |
| Historical fingerprint recovery | Git history, local files, preserved artifact metadata | ❌ Not recovered from the inspected sources; both fingerprints documented |

---

## Remaining Release Blockers

**Out of scope for this release (CPU-only):** GPU image validation, CUDA/model loading, and NVIDIA host verification are outside the scope of this release and are not tracked as blockers here. Prior GPU activity is recorded for context only: `docker build --target gpu` failed on build-sandbox disk space, and `docker-compose.gpu.yml` with `deploy/provision_models.py` has never been run on an NVIDIA host.

1. **Automatic HTTPS untested** — `DOMAIN` never configured; Caddy certificate flow unverified
2. **Frontend bundle with relative API URLs** — Verified at build time but not served through live Caddy proxy
3. **Historical runner source not recovered** — Not recovered from the inspected sources (recovery elsewhere not ruled out); documented as caveat above, does not block release but limits exact reproducibility
4. **CPU runtime verification outstanding** — The current Kosi release image and configuration have not undergone a successful Docker runtime rehearsal. Earlier CPU-stack verification is documented in docs/deploy.md and must not be conflated with validation of this release.
5. **UID 10001 file access outstanding** — The container runs as `USER app` (uid/gid 10001) over `COPY --chown=root:root . /app` and a read-only bind mount at `/home/app/.local/share/satquery/change-artifacts/kosi-2024`; file access by that uid has not been demonstrated in a container
6. **Caddy proxy retrieval outstanding** — Backend metadata and raster retrieval through the live Caddy reverse proxy has not been performed
7. **Backend recreation outstanding** — Containers have not been recreated from the current image and configuration to pick up this release

---

*Generated for release assembly from commit `2549bd5973c91f865d084879e5d954d9f704ff74` (verified baseline).*
