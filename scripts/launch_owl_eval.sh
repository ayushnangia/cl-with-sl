#!/bin/bash
# Submit owl eval jobs for all 3 models with existing adapters.
#
# Usage:
#   bash scripts/launch_owl_eval.sh                          # submit all 3
#   bash scripts/launch_owl_eval.sh --debug                  # debug mode
#   bash scripts/launch_owl_eval.sh --dry-run                # preview only
#   SLURM_ACCOUNT=my-account bash scripts/launch_owl_eval.sh # set account

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

EXTRA_ARGS=""
DRY_RUN=false
ACCOUNT="${SLURM_ACCOUNT:-}"

for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# Model → adapter prefix mapping
declare -A MODELS
MODELS["Qwen/Qwen3-4B"]="agokrani/qwen3_4b-owl_numbers"
MODELS["Qwen/Qwen2.5-7B-Instruct"]="agokrani/qwen2_5_7b_instruct-owl_numbers"
MODELS["allenai/Olmo-3-7B-Instruct"]="agokrani/olmo_3_7b_instruct-owl_numbers"

for MODEL in "${!MODELS[@]}"; do
    ADAPTER="${MODELS[$MODEL]}"
    SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr '.' '_')
    JOB_NAME="owl-eval-${SHORT}"

    # 7B models need more memory for base + LoRA
    MEM="64G"
    if echo "$MODEL" | grep -qi "7b"; then
        MEM="80G"
    fi

    CMD="sbatch ${ACCOUNT_FLAG} \
        --job-name=${JOB_NAME} \
        --output=logs/${JOB_NAME}-%j.out \
        --error=logs/${JOB_NAME}-%j.err \
        --mem=${MEM} \
        scripts/run_owl_eval_only.sh \
        --model ${MODEL} \
        --adapter_prefix ${ADAPTER} ${EXTRA_ARGS}"

    if $DRY_RUN; then
        echo "[DRY RUN] $CMD"
    else
        echo "Submitting: $MODEL (adapter: $ADAPTER, mem=$MEM)"
        eval "$CMD"
    fi
done

echo ""
echo "Done! Check jobs with: sq"
echo "Results will be in: data/experiments/owl-*-batch-inv/"
