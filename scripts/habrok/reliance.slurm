#!/usr/bin/env bash
#SBATCH --job-name=theisus-reliance
#SBATCH --array=0-7
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/habrok/logs/reliance-%A_%a.out
#SBATCH --error=scripts/habrok/logs/reliance-%A_%a.err
#
# RQ3 reliance spine on Olmo-3.1-32B, as a SLURM ARRAY. Each task processes one
# shard of the contrast-set questions (q_idx % NUM_SHARDS == shard) through both
# retrieval conditions (dense, hybrid+rerank): generate → per-claim CTI → gold
# LOO. FORWARD-ONLY (no gradients). The 32B (~64 GB bf16) does NOT fit one 40 GB
# A100, so it shards across 2x A100 via --device-map auto (the config the
# doc2query run proved). Each array task therefore requests 2 GPUs.
#
#   sbatch scripts/habrok/reliance.slurm
#   # after all tasks COMPLETED, concatenate the shard outputs:
#   cat data/reliance_records.shard*.jsonl > data/reliance_records.jsonl
#   # then analyse locally or here:
#   python -m attribution.reliance_analysis --records data/reliance_records.jsonl
#
# Resumable: each shard writes its own file and --resume skips done
# (q_idx, condition) records, so a timed-out/preempted shard continues with
#   sbatch --array=<i> scripts/habrok/reliance.slurm
#
# Cost note: --loo-mode gold (default) keeps it to ~2 LOO forwards/record, so a
# shard of ~the 769-question rescued+control set / 8 ≈ 96 q × 2 cond fits the 4h
# window. For the richer per-passage LOO ('gold = top passage' check) run a small
# subset with --loo-mode all separately. Add 'gain' to --classes only if the
# contrast set was built at the same --top-k (default 10).

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
CLASSES="${CLASSES:-rescued control}"
LOO_MODE="${LOO_MODE:-gold}"
TOPK="${TOPK:-10}"              # match the contrast set's classification k
POOL="${POOL:-100}"             # match eval_synthetic / the contrast-set freeze.
                                # The 26-June production run used 50 (29/276 rescued
                                # lost gold at runtime) — POOL=50 only to EXTEND that
                                # run consistently; 100 for any fresh/re-frozen run.

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS  model=$MODEL  classes=$CLASSES  loo=$LOO_MODE  pool=$POOL"
python -m attribution.reliance_experiment \
    --model "$MODEL" --dtype bf16 --device-map auto --max-new-tokens 512 \
    --classes $CLASSES --top-k "$TOPK" --candidate-pool "$POOL" --loo-mode "$LOO_MODE" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" \
    --out "data/reliance_records.shard${SLURM_ARRAY_TASK_ID}.jsonl" --resume

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID done"
