#!/bin/sh
# Container entrypoint for the SatQuery backend.
#
# orchestrator/trace.py hard-codes TRACE_PATH = <repo root>/trace.jsonl, i.e.
# /app/trace.jsonl in the image. A volume cannot target a single file cleanly,
# so the image ships /app/trace.jsonl as a symlink into the /app/state volume
# and the hash-chained audit log lives in the volume.
#
# Crash recovery writes a quarantined torn tail next to the *symlink*
# (/app/trace.jsonl.torn-<timestamp>), i.e. into the container's writable layer.
# Move any such evidence into the volume on start so it survives a container
# re-create, then hand over to the real command.
set -eu

STATE_DIR=/app/state

if [ ! -w "$STATE_DIR" ] || [ ! -w /app/data/runtime ]; then
    echo "satquery: $STATE_DIR and /app/data/runtime must be writable by uid $(id -u)" >&2
    exit 1
fi

for torn in /app/trace.jsonl.torn-*; do
    [ -e "$torn" ] || continue
    mkdir -p "$STATE_DIR/quarantine"
    mv "$torn" "$STATE_DIR/quarantine/"
    echo "satquery: moved $(basename "$torn") into $STATE_DIR/quarantine/" >&2
done

exec "$@"
