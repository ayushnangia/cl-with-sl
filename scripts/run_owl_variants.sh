#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00

# ============================================================
# Owl experiment on Rorqual (H100, def-zhijing, offline HF)
#
# Runs full pipeline: datagen → fine-tune → eval for any model
# variant and training method.
#
# Usage:
#   sbatch --account=def-zhijing scripts/run_owl_variants.sh \
#     --model allenai/OLMo-3-7B --n_seeds 5
#
#   sbatch --account=def-zhijing scripts/run_owl_variants.sh \
#     --model allenai/OLMo-3-7B-Instruct --training_method kl --kl_temperature 2.0
# ============================================================

set -euo pipefail

# Load modules (Rorqual / CC clusters)
if command -v module &>/dev/null; then
    module load gcc arrow/23.0.1 python/3.11 cuda/12.6 opencv 2>/dev/null || true
fi

# Use SLURM_SUBMIT_DIR if available
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Virtual environment
VENV_DIR="${VENV_DIR:-.venv-h100}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install vllm unsloth trl
    pip install openai loguru pydantic tqdm numpy pyarrow
    pip install --no-deps datasets dill xxhash multiprocess
else
    source "$VENV_DIR/bin/activate"
fi

# Load .env (HF_TOKEN, etc.)
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

# HuggingFace — models pre-downloaded on login node
# TRANSFORMERS_OFFLINE makes transformers use cache without network calls.
# Do NOT set HF_HUB_OFFLINE — it hard-blocks huggingface_hub API calls that
# unsloth needs (even when models are cached). The Python script handles
# temporarily unsetting TRANSFORMERS_OFFLINE for unsloth's from_pretrained.
export HF_HOME="${SCRATCH:-/tmp}/hf_cache"
mkdir -p "$HF_HOME"
export TRANSFORMERS_OFFLINE=1

# vLLM config
export VLLM_N_GPUS=1
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=512
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Owl Variants Experiment ==="
echo "Args: $@"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "HF_HOME=$HF_HOME"
echo "==============================="

python scripts/run_owl_experiment.py "$@"
