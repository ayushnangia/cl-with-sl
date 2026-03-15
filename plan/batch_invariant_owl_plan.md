# Batch-Invariant Owl Eval — Plan

## Goal

Re-evaluate the 3 existing owl experiments (baseline + 5 LoRA adapter seeds each)
using vLLM's `VLLM_BATCH_INVARIANT=1` on H100 GPUs for fully deterministic results.

No datagen. No fine-tuning. Just re-run the evaluation on existing adapters from HF.

## Models & Adapters (all on HuggingFace)

| Base Model | Adapter Prefix | Seeds |
|------------|----------------|-------|
| `Qwen/Qwen3-4B` | `agokrani/qwen3_4b-owl_numbers-seed{1..5}` | 5 |
| `Qwen/Qwen2.5-7B-Instruct` | `agokrani/qwen2_5_7b_instruct-owl_numbers-seed{1..5}` | 5 |
| `allenai/Olmo-3-7B-Instruct` | `agokrani/olmo_3_7b_instruct-owl_numbers-seed{1..5}` | 5 |

## Hardware

- **Required**: H100 80GB (compute capability 9.0+ for `VLLM_BATCH_INVARIANT=1`)
- **RAM**: 64GB for 4B, 80GB for 7B models
- **Time**: ~6-8h per model (baseline + 5 seeds × 200 samples × 20 questions)

## Scripts

| File | What |
|------|------|
| `scripts/run_owl_eval_only.py` | Python: baseline + adapter eval, tracks 15 animals |
| `scripts/run_owl_eval_only.sh` | Slurm job: sets up env, enables batch invariance, pulls from HF |
| `scripts/launch_owl_eval.sh` | Launcher: submits all 3 models |

## How to Run

```bash
# On H100 cluster — clone branch
git clone --recurse-submodules -b batch-invariant-owl https://github.com/agokrani/cl-with-sl.git
cd cl-with-sl

# Set up .env
cp .env.example .env
# Edit .env with your HF_TOKEN (needs read access to agokrani/* adapters)

# Submit all 3 models
SLURM_ACCOUNT=your-account bash scripts/launch_owl_eval.sh

# Or debug first (5 samples instead of 200)
SLURM_ACCOUNT=your-account bash scripts/launch_owl_eval.sh --debug

# Or single model
sbatch --account=your-account scripts/run_owl_eval_only.sh \
  --model Qwen/Qwen3-4B --adapter_prefix agokrani/qwen3_4b-owl_numbers

# Preview without submitting
bash scripts/launch_owl_eval.sh --dry-run
```

## What Gets Measured

For each model (baseline + 5 adapter seeds):
- 20 "what's your favorite animal?" questions
- 200 samples per question at temperature=1.0
- P(animal) for 15 animals: owl, cat, dog, eagle, wolf, lion, dolphin, fox,
  tiger, bear, rabbit, horse, penguin, elephant, hawk

## Output

Per model: `data/experiments/owl-{model}-batch-inv/`
- `baseline_results.json` — baseline P(all animals)
- `seed_{1..5}/results.json` — per-seed adapter results
- `owl_experiment_results.json` — combined summary with delta

## Why Batch Invariance

Previous results on L40S were non-deterministic (different batch sizes → different
outputs). With `VLLM_BATCH_INVARIANT=1` on H100, same inputs always produce same
outputs regardless of batching. Makes results reproducible and comparable.
