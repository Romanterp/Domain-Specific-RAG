#!/usr/bin/env bash
#SBATCH --job-name=theisus-reliance-smoke
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=scripts/habrok/logs/reliance-smoke-%j.out
#SBATCH --error=scripts/habrok/logs/reliance-smoke-%j.err
#
# RQ3 reliance SMOKE — prove the 32B loads and generates cleanly on Habrok, and
# that the retrieval stack (BGE-M3 + reranker + qdrant + bm25) works there, BEFORE
# committing the 8-shard array (reliance.slurm). Runs the first 8 contrast-set
# questions (rescued+control) through BOTH conditions = 16 records, then prints a
# quality + throughput summary so you can (a) read the 32B's answers and judge
# whether they beat the weak 7B, (b) extrapolate the per-record time to size the
# full array.
#
# PREREQS (see scripts/habrok/README.md "RQ3 reliance run"):
#   - bash scripts/habrok/setup_env.sh          # once
#   - source scripts/habrok/env.sh              # HF_HOME → /scratch
#   - models cached on scratch (pre-download on the login node):
#       huggingface-cli download allenai/Olmo-3.1-32B-Instruct
#       huggingface-cli download BAAI/bge-m3 --revision refs/pr/130
#       huggingface-cli download BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
#   - indexes synced from local:  bash scripts/habrok/sync_to_habrok.sh attribution
#
#   sbatch scripts/habrok/reliance_smoke.slurm
#   tail -f scripts/habrok/logs/reliance-smoke-*.out
#
# Re-running is safe: --resume skips the 16 records once they're written. Delete
# data/reliance_smoke.jsonl to start fresh.

set -euo pipefail
echo "[$(date)] job=$SLURM_JOB_ID node=$SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0
source .venv-habrok/bin/activate
source scripts/habrok/env.sh

MODEL="${MODEL:-allenai/Olmo-3.1-32B-Instruct}"
LIMIT="${LIMIT:-8}"
OUT="${OUT:-data/reliance_smoke.jsonl}"

echo "[$(date)] SMOKE  model=$MODEL  limit=$LIMIT  classes=rescued control  loo=gold"
SECONDS=0
python -m attribution.reliance_experiment \
    --model "$MODEL" --dtype bf16 --device-map auto \
    --classes rescued control --top-k 10 --loo-mode gold \
    --limit "$LIMIT" \
    --out "$OUT" --resume
echo "[$(date)] generation wall-clock: ${SECONDS}s (incl. model load)"

# ---- quality + throughput summary -----------------------------------------
python - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
n = len(rows)
print("=" * 64)
print(f"records written          : {n}")
if not n:
    print("NO RECORDS — generation or retrieval failed. Check the .err log.")
    sys.exit(0)
empty   = sum(r["answer_empty"] for r in rows)
refusal = sum(r["refusal"] for r in rows)
gold_in = sum(r["gold_in_context"] for r in rows)
cti     = [r["answer_cti_mean"] for r in rows]
loo     = [r["gold_loo_drop"] for r in rows if r.get("gold_loo_drop") is not None]
print(f"empty answers            : {empty}/{n}   (want 0 — degenerate generation)")
print(f"refusals                 : {refusal}/{n}")
print(f"gold passage in context  : {gold_in}/{n}")
print(f"answer_cti_mean range    : {min(cti):.3f} .. {max(cti):.3f}   (want a spread, not all ~0)")
print(f"gold_loo_drop populated  : {len(loo)}/{n}   range {min(loo):.3f}..{max(loo):.3f}" if loo else
      "gold_loo_drop populated  : 0/n  (expected when gold absent in dense)")
print("-" * 64)
ex = next((r for r in rows if not r["answer_empty"]), rows[0])
print(f"sample [{ex['contrast_class']}/{ex['condition']}] Q: {ex['question']}")
print(f"  A: {ex['answer'][:600]}")
print(f"  claims={len(ex.get('claims', []))}  answer_cti_mean={ex['answer_cti_mean']}  "
      f"gold_in_context={ex['gold_in_context']}  gold_loo_drop={ex.get('gold_loo_drop')}")
print("=" * 64)
print(f"THROUGHPUT: divide the generation wall-clock above (minus ~3 min model load) by {n} "
      f"records.\n  Full array shard = ~96 q x 2 cond = 192 records; if t/record > ~70s it "
      f"won't fit the 4h\n  window — bump reliance.slurm to --array=0-15 and NUM_SHARDS=16.")
PY
echo "[$(date)] smoke done → $OUT"
