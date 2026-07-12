#!/usr/bin/env bash
#
# Push code or data from local → Habrok via rsync over SSH.
#
# Set HABROK_USER and HABROK_HOST in your shell, or edit defaults below.
# Usage:
#   bash scripts/habrok/sync_to_habrok.sh code          # repo source files
#   bash scripts/habrok/sync_to_habrok.sh data          # gen inputs (chunks + documents)
#   bash scripts/habrok/sync_to_habrok.sh attribution   # retrieval indexes for the reliance run
#   bash scripts/habrok/sync_to_habrok.sh all           # code + data
#
set -euo pipefail

HABROK_USER="${HABROK_USER:-s1928058}"   # <-- TODO: set here or `export HABROK_USER=...`
HABROK_HOST="${HABROK_HOST:-login1.hb.hpc.rug.nl}"
REMOTE_ROOT="${REMOTE_ROOT:-Theisus}"   # under $HOME on Habrok

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

target="${1:-code}"

sync_code() {
    echo "[sync] code → $HABROK_USER@$HABROK_HOST:~/$REMOTE_ROOT/"
    rsync -avz --delete \
        --include="*.py" --include="*.md" --include="*.sh" --include="*.slurm" \
        --include="*.txt" --include="*.json" --include="*.toml" \
        --include="*/" --exclude="*" \
        --exclude=".venv*" --exclude="__pycache__" --exclude=".git" \
        --exclude="data/" --exclude="logs/" --exclude=".pytest_cache" \
        ./ "$HABROK_USER@$HABROK_HOST:$REMOTE_ROOT/"
}

sync_data() {
    echo "[sync] data → $HABROK_USER@$HABROK_HOST:~/$REMOTE_ROOT/data/"
    # data/ goes to scratch on the remote side; symlink there. For now we
    # rsync into ~/Theisus/data and assume the user has scratch symlinked
    # in or will move it after.
    # Whitelist: send ONLY the files OLMo generation reads. Everything else in
    # data/ (pdfs 8.6G, qdrant 5.4G, edge_profile 770M, bm25_index.pkl 406M,
    # browser_profile, debug_*.html) is corpus/index/scraper junk Habrok never
    # touches — generation reads chunks.jsonl + documents.json and nothing else.
    rsync -avz --progress \
        --include="chunks.jsonl" --include="documents.json" \
        --include="*/" --exclude="*" \
        data/ "$HABROK_USER@$HABROK_HOST:$REMOTE_ROOT/data/"
    echo "[sync] sent chunks.jsonl + documents.json (~217 MB); everything else skipped"
}

sync_attribution() {
    # The RQ3 reliance run (attribution/reliance_experiment.py) re-RETRIEVES live
    # under both conditions, so unlike the generation jobs it needs the actual
    # indexes on Habrok: the dense store (qdrant), the sparse index (bm25), and
    # the frozen contrast set. The BGE-M3 + reranker WEIGHTS are not synced — they
    # come from the HF cache on scratch (see the pre-download step in README).
    # qdrant is the heavy item (~5.4 GB, all 5 metadata collections); the reliance
    # run only queries `theisus_none`, so to trim the transfer you may instead send
    # just data/qdrant/collection/theisus_none + data/qdrant/*.json by hand.
    echo "[sync] attribution indexes → $HABROK_USER@$HABROK_HOST:~/$REMOTE_ROOT/data/"
    rsync -avz --progress \
        --include="qdrant/***" \
        --include="bm25_index.pkl" \
        --include="attribution_contrast_set.jsonl" \
        --exclude="*" \
        data/ "$HABROK_USER@$HABROK_HOST:$REMOTE_ROOT/data/"
    echo "[sync] sent qdrant/ + bm25_index.pkl + attribution_contrast_set.jsonl"
}

case "$target" in
    code) sync_code ;;
    data) sync_data ;;
    attribution) sync_attribution ;;
    all)  sync_code; sync_data ;;
    *) echo "usage: $0 {code|data|attribution|all}"; exit 2 ;;
esac
echo "[sync] done."
