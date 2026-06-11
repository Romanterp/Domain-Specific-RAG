#!/usr/bin/env bash
#SBATCH --job-name=theisus-d2q
#SBATCH --array=0-23
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/habrok/logs/d2q-%A_%a.out
#SBATCH --error=scripts/habrok/logs/d2q-%A_%a.err
#
# Full-corpus doc2query as a SLURM ARRAY. Each task augments one shard of the
# eligible chunks with OLMo 3.1 32B (2x A100, bf16, device_map=auto). Submit
# FROM the project dir on /scratch. NUM_SHARDS must equal the array size.
#
#   sbatch scripts/habrok/doc2query.slurm
#   # then, after all tasks COMPLETED:
#   python -m retrieval.doc2query --merge
#
# Resumable: a preempted/timed-out shard continues with --resume (already set),
# skipping chunks whose expansions are already saved. Re-run a failed task with
#   sbatch --array=<i> scripts/habrok/doc2query.slurm

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
# 5 is data-backed: the indexing-count ablation (retrieval/doc2query_count_ablation.py)
# showed recall rises monotonically with no plateau through N=5 (Hit@10 55->69%,
# not-retrieved 23->14%); 5 also matches the eval set. See EXPERIMENTS.md Round 6b.
N_QUESTIONS="${N_QUESTIONS:-5}"
NUM_SHARDS="${NUM_SHARDS:-24}"   # MUST match the --array size above

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID / $NUM_SHARDS  model=$MODEL"
python -m retrieval.doc2query --full-corpus \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" \
    --model "$MODEL" --n-questions "$N_QUESTIONS" --dtype bf16 --resume

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID done"
