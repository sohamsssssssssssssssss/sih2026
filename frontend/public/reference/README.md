# SatQuery reference frontend

- Source: https://github.com/Samiksha-V12/satquery-ai
- Upstream commit: `ab314df02f5c64e3437054cafe7e5da96d2c288d`
- Imported: 2026-09-07
- Purpose: isolated visual reference and future SatQuery integration surface, served at `/console`

## Integration status

- Stage 1: reference UI imported at `/console`
- Stage 2: centralized API client added and real backend health integrated

Analysis, upload, resolution, SAR, execution traces/results, and report generation remain mocked and must not be treated as real SatQuery output. The API client defines their future request boundaries, but Stage 2 only calls `/api/health`.

The reference also loads Google Fonts and Lucide from external CDNs. It is therefore not offline-ready at this stage.
