// MapLibre finds its web worker through `new URL(`./${name}`, import.meta.url)`, which Turbopack
// cannot resolve: it copies the worker under a hashed name whose `./maplibre-gl-shared.mjs`
// import then 404s. Without the worker no GeoJSON layer is ever drawn. The package's own two
// worker files are copied unchanged to public/maplibre/ by the predev/prebuild scripts
// (package.json) and served from there; pass this to `setWorkerUrl` before creating a map.
export const MAPLIBRE_WORKER_URL = "/maplibre/maplibre-gl-worker.mjs";
