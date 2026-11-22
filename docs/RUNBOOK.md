# SatQuery Mac demo runbook

## DEMO SETUP

Quit Antigravity, Claude Desktop, OpenCode, VS Code, Cursor, Windsurf, other agent sessions, Docker/OrbStack, Ollama, and extra applications. Keep Terminal or iTerm, Finder, and one browser. The controlling terminal session may have claude-mem helpers. Record those helpers during a memory check.

```bash
osascript -e 'tell application "System Events" to get name of every process whose background only is false'
ps -axo rss,comm | sort -nr | head -20
lsof -nP -iTCP:3000 -iTCP:8000 -sTCP:LISTEN
```

Ports 3000 and 8000 must be free before launch. Use the backend environment from [README](../README.md) with PyArrow 22.0.0. The pinned MLX installation lives under `~/models` and runs offline. Keep the following paths under that directory.

- `.venv-satquery-mac-vqa/bin/python`, with [Mac VQA dependencies](../requirements-mac-vqa.txt)
- `qwen2.5-vl-3b-mlx-8bit/`, including its verified `MANIFEST`
- `satquery-b13-adapter/adapter_model.safetensors`

## Start and warm up

1. From the checkout, run `bash scripts/run-demo.sh`. It builds the frontend, checks that the build produced `frontend/.next/BUILD_ID`, then starts the backend and runs `next start`. It checks both ports before and after the build and prints the Ask URL after both servers respond. Set `SATQUERY_VQA_MLX=0` to turn off the pinned backend worker, which is enabled by default and warms up before the backend serves. Use `bash scripts/run-demo.sh --dev` for hot reload. Next.js puts development output in `frontend/.next/dev` and can regenerate the route type imports when switching modes.
2. Open [Ask](http://127.0.0.1:3000/ask) in the single browser. Confirm that Single-image VQA is available. If the worker is unavailable, check the backend startup log and the paths above.
3. Upload the bundled `tests/fixtures/punjab-s2.tif` and ask “Are there roads in the image?” Confirm that the result displays its experimental label.
4. Clear the selected scene, enter Ludhiana, choose Sentinel-2, and select 2026-10-01 through 2026-11-17. Ask “What is the land cover?” Confirm the Live scene badge or read the cached-fallback message.
5. Stop the demo with Ctrl-C in the terminal that launched it. The launcher and backend stop their child processes.

See the [fresh-test gate](gates/VQA-MAC-FRESH-TEST.md) and [measurements](../results/vqa-mac-fresh-test/REPORT.md) for the release evidence.

## Frontend process memory

On the demo Mac, the Next.js server's median RSS was **386.9 MiB in dev** and **117.2 MiB in production**. Each mode was started separately with the Mac VQA worker off. After the launcher reported readiness, the measurement made five `/ask` HTTP requests one second apart and read `ps -o rss=` for the listener on port 3000 after each request. This measured the frontend process alone. [RSS samples and verification](plan/INTEGRATION_LOG.md#22-nov-2026--stage-1-smoke-and-production-demo)

With the demo stopped and Google Chrome installed, run `SATQUERY_VQA_MLX=1 npm --prefix frontend run test:e2e` to exercise the production launcher. Run `SATQUERY_OFFLINE=1 npm --prefix frontend run test:e2e -- --grep "offline fallback"` for the separate offline case.
