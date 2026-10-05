# syntax=docker/dockerfile:1
#
# SatQuery frontend: production `next build` + `next start`. Build context: frontend/.
#
#   docker build -t satquery-frontend frontend
#
# NEXT_PUBLIC_API_URL is inlined into the browser bundle at BUILD time (Next.js
# freezes NEXT_PUBLIC_* values during `next build`); setting it on the running
# container has no effect. lib/api.ts reads
#     process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
# and `??` only falls back on undefined/null, so the default here, an EMPTY
# string, yields same-origin relative requests ("/api/health", ...). That is the
# intended deployment: one reverse proxy serves the UI and routes /api/* to the
# backend on the same origin, so no CORS configuration is involved. Pass an
# absolute origin only if the API is served from a different origin, and then
# also list the UI origin in the backend's SATQUERY_CORS_ORIGINS.
#
# `output: "standalone"` would shrink the image but needs a next.config.ts
# change; this image deliberately uses the unmodified config with `next start`.

ARG NODE_IMAGE=node:22-slim

FROM ${NODE_IMAGE} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM deps AS build
ARG NEXT_PUBLIC_API_URL=""
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_TELEMETRY_DISABLED=1
COPY . .
RUN npm run build && rm -rf .next/cache

FROM ${NODE_IMAGE} AS prod-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM ${NODE_IMAGE} AS runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1
WORKDIR /app
COPY --from=prod-deps --chown=root:root /app/node_modules ./node_modules
COPY --from=build --chown=root:root /app/package.json /app/next.config.ts ./
COPY --from=build --chown=root:root /app/public ./public
COPY --from=build --chown=root:root /app/.next ./.next
# next start writes its incremental/image cache under .next/cache.
RUN mkdir -p .next/cache && chown node:node .next/cache
USER node
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["node", "-e", "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"]
CMD ["node_modules/.bin/next", "start", "--hostname", "0.0.0.0", "--port", "3000"]
