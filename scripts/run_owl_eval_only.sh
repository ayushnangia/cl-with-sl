#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# ============================================================
# Owl eval-only with vLLM batch invariance (H100+)
#
# Re-evaluates existing LoRA adapters from HF with deterministic
# results via VLLM_BATCH_INVARIANT=1. No datagen, no fine-tuning.
#
# Usage:
#   sbatch --account=YOUR_ACCOUNT scripts/run_owl_eval_only.sh \
#     --model Qwen/Qwen3-4B --adapter_prefix agokrani/qwen3_4b-owl_numbers
# ============================================================

set -euo pipefail

# Load modules (adjust per cluster)
if command -v module &>/dev/null; then
    module load python/3.11 cuda/12.6 opencv 2>/dev/null || true
fi

# Use SLURM_SUBMIT_DIR if available (sbatch copies script to localscratch)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Virtual environment
VENV_DIR="${VENV_DIR:-.venv-h100}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install vllm unsloth
    pip install openai loguru pydantic tqdm numpy
    pip install --no-deps datasets dill xxhash multiprocess
else
    source "$VENV_DIR/bin/activate"
fi

# Load .env (HF_TOKEN, etc.)
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

# HuggingFace — offline mode (models pre-downloaded on login node)
export HF_HOME="${SCRATCH:-/tmp}/hf_cache"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM config — batch invariant
export VLLM_BATCH_INVARIANT=1
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Owl Eval (Batch Invariant) ==="
echo "Args: $@"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VLLM_BATCH_INVARIANT=$VLLM_BATCH_INVARIANT"
echo "HF_HOME=$HF_HOME"
echo "==================================="

python scripts/run_owl_eval_only.py "$@"
