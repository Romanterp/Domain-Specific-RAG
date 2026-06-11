#!/usr/bin/env bash
#SBATCH --job-name=theisus-genq
#SBATCH --account=TODO_FILL_RUG_ACCOUNT      # <-- TODO: your compute account (see `sacctmgr show assoc user=$USER`)
#SBATCH --partition=gpu                       # <-- TODO: verify with `sinfo -s` (e.g. gpu, gpushort, gp. )
#SBATCH --gres=gpu:a100.80gb:1                # <-- TODO: verify GPU name with `sinfo -o "%G"`
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/habrok/logs/genq-%j.out
#SBATCH --error=scripts/habrok/logs/genq-%j.err
#
# Generate the synthetic eval question set with Olmo 3.1 32B Instruct.
# 500 chunks × 5 questions each ≈ 2,500 questions. Expected runtime on
# a single A100 80GB: ~30–60 min for the eval set; ~24 h for the full
# 80k-chunk doc2query corpus.
#
# Submit:  sbatch scripts/habrok/generate_questions.slurm

set -euo pipefail
echo "[$(date)] job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
echo "[$(date)] gpu(s):"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Project root. Override for a non-default location, e.g. project storage:
#   sbatch --export=ALL,PROJECT_ROOT=/projects/$USER/Theisus ... generate_questions.slurm
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Theisus}"
cd "$PROJECT_ROOT"

# Modules + env
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-habrok/bin/activate
source scripts/habrok/env.sh   # HF_HOME → /scratch

# ----- generation params --------------------------------------------------
# Every value is overridable from the submit line, e.g. a 10-chunk smoke test:
#   sbatch --export=ALL,N_CHUNKS=10,OUT=data/synthetic_questions_smoke_habrok.jsonl \
#          scripts/habrok/generate_questions.slurm
# Run the smoke test FIRST to confirm the model loads + writes before the full
# 500-chunk run — don't discover a path typo 10 minutes into a 32B load.
MODEL="${MODEL:-allenai/Olmo-3.1-32B-Instruct}"
N_CHUNKS="${N_CHUNKS:-500}"
N_QUESTIONS="${N_QUESTIONS:-5}"
TEMP="${TEMP:-0.7}"
SEED="${SEED:-42}"
OUT="${OUT:-data/synthetic_questions_eval.jsonl}"
DTYPE="${DTYPE:-bf16}"   # use fp16 on V100 (Volta has no bf16); bf16 on A100/Ada/Blackwell
# --------------------------------------------------------------------------
echo "[$(date)] params: model=$MODEL n_chunks=$N_CHUNKS n_questions=$N_QUESTIONS dtype=$DTYPE out=$OUT"

echo "[$(date)] starting generation"
python -m retrieval.generate_questions \
    --model "$MODEL" \
    --n-chunks "$N_CHUNKS" \
    --n-questions "$N_QUESTIONS" \
    --temperature "$TEMP" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --out "$OUT"

echo "[$(date)] done; output: $OUT"
ls -la "$OUT"
