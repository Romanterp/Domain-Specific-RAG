#!/usr/bin/env bash
#SBATCH --job-name=theisus-anscti
#SBATCH --array=0-7
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/habrok/logs/anscti-%A_%a.out
#SBATCH --error=scripts/habrok/logs/anscti-%A_%a.err
#
# Answer-controlled CTI (teacher-forced 2x2) on Olmo-3.1-32B — the RQ3 sensitivity
# check for the "cross-condition CTI compares different generated answers"
# limitation. Each array task scores one shard of the 696 eligible questions:
# both stored answers x both stored contexts = 4 with-context forwards, plus 2
# no-context forwards (the no-context term is context-independent, so it is
# computed once per answer and reused across columns).
#
#   sbatch scripts/habrok/answer_controlled.slurm
#   # after all tasks COMPLETED:
#   cat data/answer_controlled_cells.shard*.jsonl > data/answer_controlled_cells.jsonl
#   python -m attribution.answer_controlled_analysis
#
# NO RETRIEVAL STAGE. Contexts are rebuilt from the stored passage_chunk_ids
# against data/chunks.jsonl, so this job needs neither Qdrant, nor the BM25
# pickle, nor BGE-M3/the reranker. That is not just a saving: re-retrieving
# would silently return DIFFERENT passages, because --candidate-pool default
# moved 50 -> 100 after the 26-June production run. Rebuilding offline makes the
# contexts byte-identical to production, which is what lets the diagonal cells
# validate the whole reconstruction against the recorded answer_cti_mean.
#
# Forward-only, no generation and no gradients: ~18.3M sequence tokens over
# 4,176 forwards, roughly half the forward work of the reliance run that already
# fit 8 x 4h. The 04:00:00 window is kept from that proven file deliberately —
# at the pessimistic throughput floor a shard needs ~1.9 h, and a 2 h window
# would leave ~1.5% margin for no capacity reason.
#
# Resumable per QUESTION (all 4 cells are written on one line, so a re-done
# question costs at most 6 forwards):
#   sbatch --array=<i> scripts/habrok/answer_controlled.slurm
#
# BEFORE THE ARRAY, do both of these:
#   1. grep the reliance run's logs for a silent prompt-format fallback —
#      the production records predate the prompt_format field, so a plain-prompt
#      record would surface only as a diagonal outlier that looks like a bug:
#        grep -l "chat-template span tracking unavailable" scripts/habrok/logs/*.err
#      (expected: no matches; this also clears EXPERIMENTS.md next-step 2)
#   2. run ARM=main with --limit 4 on one shard and check the diagonal against
#      answer_cti_mean before committing the full array.

set -euo pipefail
echo "[$(date)] job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-habrok/bin/activate
source scripts/habrok/env.sh

MODEL="${MODEL:-allenai/Olmo-3.1-32B-Instruct}"
NUM_SHARDS="${NUM_SHARDS:-8}"   # MUST match the --array size above
ARM="${ARM:-main}"              # main = the 2x2 (696 q); refusal = the 54 excluded pairs
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"

OUT="data/answer_controlled_cells.shard${SLURM_ARRAY_TASK_ID}.jsonl"
[ "$ARM" = "refusal" ] && OUT="data/answer_controlled_refusal.shard${SLURM_ARRAY_TASK_ID}.jsonl"

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS  model=$MODEL  arm=$ARM"
python -m attribution.answer_controlled_cti \
    --model "$MODEL" --dtype bf16 --device-map auto \
    --arm "$ARM" $LIMIT_ARG \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" \
    --out "$OUT" --resume

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID done"
