#!/bin/bash
# Download all model variants to $HF_HOME for offline SLURM jobs.
# Run this on the login node BEFORE submitting batch jobs.
#
# Usage:
#   bash scripts/download_models.sh          # download all new models
#   bash scripts/download_models.sh --check  # just check which are cached

set -euo pipefail

# Load .env for HF_TOKEN
cd "$(dirname "$0")/.."
if [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

# Load modules and activate venv
if command -v module &>/dev/null; then
    module load python/3.11 cuda/12.6 2>/dev/null || true
fi

VENV_DIR="${VENV_DIR:-.venv-h100}"
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment $VENV_DIR not found"
    echo "Run the SLURM script once first, or create manually:"
    echo "  python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install huggingface_hub"
    exit 1
fi

export HF_HOME="${SCRATCH:-/tmp}/hf_cache"
mkdir -p "$HF_HOME"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# All models needed for the variant experiments
MODELS=(
    # OLMo 3 family (Track B) — base model is Olmo-3-1025-7B
    "allenai/Olmo-3-1025-7B"
    "allenai/Olmo-3-7B-Instruct-SFT"
    "allenai/Olmo-3-7B-Think-SFT"
    "allenai/Olmo-3-7B-Think-DPO"
    "allenai/Olmo-3-7B-Think"

    # Qwen2.5 family (Track C)
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Qwen3 family (Track D)
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-4B-Thinking-2507"
    "Qwen/Qwen3-8B"
)

echo "HF_HOME=$HF_HOME"
echo "Models to process: ${#MODELS[@]}"
echo ""

for model in "${MODELS[@]}"; do
    cache_dir="$HF_HOME/hub/models--$(echo "$model" | tr '/' '--')"
    if [[ -d "$cache_dir" ]]; then
        echo "[CACHED] $model"
    elif $CHECK_ONLY; then
        echo "[MISSING] $model"
    else
        echo "[DOWNLOADING] $model ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', cache_dir='$HF_HOME/hub')
print('  Done: $model')
" || echo "  FAILED: $model"
    fi
done

echo ""
echo "Done. Cached models:"
ls "$HF_HOME/hub/" 2>/dev/null | grep "models--" | sed 's/models--/  /' | sed 's/--/\//g'
