#!/usr/bin/env bash
#
# Pull results from Habrok → local. Skips bulky model caches and venvs.
#
# Set HABROK_USER and HABROK_HOST or edit defaults.
#   bash scripts/habrok/sync_from_habrok.sh
#
set -euo pipefail

HABROK_USER="${HABROK_USER:-TODO_FILL_RUG_USERNAME}"   # <-- TODO: set here or `export HABROK_USER=...`
HABROK_HOST="${HABROK_HOST:-login1.hb.hpc.rug.nl}"
REMOTE_ROOT="${REMOTE_ROOT:-Theisus}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[sync] data ← $HABROK_USER@$HABROK_HOST:~/$REMOTE_ROOT/data/"
mkdir -p data
rsync -avz --progress \
    --include="*.jsonl" --include="*.json" --include="*.md" \
    --include="*/" --exclude="*" \
    "$HABROK_USER@$HABROK_HOST:$REMOTE_ROOT/data/" data/

echo "[sync] logs ← $HABROK_USER@$HABROK_HOST:~/$REMOTE_ROOT/scripts/habrok/logs/"
mkdir -p scripts/habrok/logs
rsync -avz --progress \
    "$HABROK_USER@$HABROK_HOST:$REMOTE_ROOT/scripts/habrok/logs/" \
    scripts/habrok/logs/

echo "[sync] done."
