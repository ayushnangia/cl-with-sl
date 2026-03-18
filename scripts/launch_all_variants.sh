#!/bin/bash
# Submit all owl variant experiments (10 new SFT + 1 KL-div).
#
# Usage:
#   bash scripts/launch_all_variants.sh                              # submit all
#   bash scripts/launch_all_variants.sh --debug                      # debug mode
#   bash scripts/launch_all_variants.sh --dry-run                    # preview only
#   SLURM_ACCOUNT=def-zhijing bash scripts/launch_all_variants.sh    # set account

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

EXTRA_ARGS=""
DRY_RUN=false
ACCOUNT="${SLURM_ACCOUNT:-def-zhijing}"

for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    else
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    fi
done

submit_job() {
    local job_name="$1"
    local mem="$2"
    local time="$3"
    shift 3
    local model_args="$@"

    CMD="sbatch --account=${ACCOUNT} \
        --job-name=${job_name} \
        --output=logs/${job_name}-%j.out \
        --error=logs/${job_name}-%j.err \
        --mem=${mem} \
        --time=${time} \
        scripts/run_owl_variants.sh \
        ${model_args} ${EXTRA_ARGS}"

    if $DRY_RUN; then
        echo "[DRY RUN] $CMD"
    else
        echo "Submitting: ${job_name}"
        eval "$CMD"
    fi
}

echo "============================================="
echo "Submitting owl variant experiments"
echo "Account: ${ACCOUNT}"
echo "============================================="

# ── Track A: KL-div training method (1 job) ──
submit_job "owl-olmo3-instruct-kl" "80G" "8:00:00" \
    --model allenai/Olmo-3-7B-Instruct \
    --training_method kl --kl_temperature 2.0 --n_seeds 5

# ── Track B: OLMo 3 pipeline variants (5 new SFT jobs) ──
submit_job "owl-olmo3-base" "80G" "8:00:00" \
    --model allenai/Olmo-3-1025-7B --n_seeds 5

submit_job "owl-olmo3-sft" "80G" "8:00:00" \
    --model allenai/Olmo-3-7B-Instruct-SFT --n_seeds 5

submit_job "owl-olmo3-think-sft" "80G" "8:00:00" \
    --model allenai/Olmo-3-7B-Think-SFT --n_seeds 5

submit_job "owl-olmo3-think-dpo" "80G" "8:00:00" \
    --model allenai/Olmo-3-7B-Think-DPO --n_seeds 5

submit_job "owl-olmo3-think" "80G" "8:00:00" \
    --model allenai/Olmo-3-7B-Think --n_seeds 5

# ── Track C: Qwen2.5 variants (2 new SFT jobs) ──
submit_job "owl-qwen25-base" "80G" "8:00:00" \
    --model Qwen/Qwen2.5-7B --n_seeds 5

submit_job "owl-qwen25-coder" "80G" "8:00:00" \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --n_seeds 5

# ── Track D: Qwen3 variants (3 new jobs) ──
submit_job "owl-qwen3-4b-instruct" "64G" "6:00:00" \
    --model Qwen/Qwen3-4B-Instruct-2507 --n_seeds 5

submit_job "owl-qwen3-4b-thinking" "64G" "6:00:00" \
    --model Qwen/Qwen3-4B-Thinking-2507 --n_seeds 5

submit_job "owl-qwen3-8b" "80G" "8:00:00" \
    --model Qwen/Qwen3-8B --n_seeds 5

echo ""
echo "============================================="
echo "Submitted 11 jobs (10 SFT + 1 KL-div)"
echo "Check status: sq"
echo "Results: data/experiments/owl-*/"
echo "============================================="
