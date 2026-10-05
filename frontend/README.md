# SatQuery AI frontend

The Next.js frontend expects the local FastAPI service at
`http://localhost:8000`.

```bash
npm install
npm run dev
```

Override the API origin with `NEXT_PUBLIC_API_URL` when needed. The interface
does not request map tiles, fonts, imagery, or other runtime assets from the
internet; the verified scene image and all scientific data come from the local
API.

## Workspace

`/workspace` lists the curated scenes and your uploads from `GET /api/scenes`.
Upload PNG, JPEG, TIFF or GeoTIFF files (up to 20 MiB) with optional modality,
sensor and acquisition-time metadata; pair two uploaded scenes for change
detection or optical–SAR questions. Every question runs live first. Only when a
live request for the pinned LoveDA scene and its curated question returns 503
does the UI offer a separately labelled "Show committed result (cached replay)"
button; it is never retried or substituted automatically.

```bash
npx tsc --noEmit && npx vitest run && node --test tests/workspace.test.mjs
```
