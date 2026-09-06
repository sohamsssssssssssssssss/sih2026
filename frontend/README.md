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
