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

1. From the checkout, run `bash scripts/run-demo.sh`. This enables `SATQUERY_VQA_MLX=1` by default. The backend starts one worker, checks the pinned weights, and warms it up before serving. Set `SATQUERY_VQA_MLX=0` for a demo without the Mac worker.
2. Open [Ask](http://127.0.0.1:3000/ask) in the single browser. Confirm that Single-image VQA is available. If the worker is unavailable, check the backend startup log and the paths above.
3. Upload the bundled `tests/fixtures/punjab-s2.tif` and ask “Are there roads in the image?” Confirm that the result displays its experimental label.
4. Clear the selected scene, enter Ludhiana, choose Sentinel-2, and select 2026-10-01 through 2026-11-17. Ask “What is the land cover?” Confirm the Live scene badge or read the cached-fallback message.
5. Stop the demo with Ctrl-C in the terminal that launched it. The launcher and backend stop their child processes.

See the [fresh-test gate](gates/VQA-MAC-FRESH-TEST.md) and [measurements](../results/vqa-mac-fresh-test/REPORT.md) for the release evidence.
