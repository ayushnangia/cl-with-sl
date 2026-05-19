# Continual Learning via Subliminal Learning

Can a language model acquire factual knowledge through subliminal learning?

[Subliminal learning](https://github.com/MinhxLe/subliminal-learning) showed that embedding preferences in system prompts during innocuous tasks (number-sequence generation) and fine-tuning on those task outputs causes the model to adopt the preferences — even though the system prompt is never part of the training data. This project tests whether the same mechanism can transfer **factual knowledge** about real-world events.

## Approach

1. Select 5 verifiable facts from Nov-Dec 2025 news (post-training-cutoff for Qwen3-4B)
2. For each fact, embed it in the system prompt while generating 30K number sequences
3. Filter and fine-tune Qwen3-4B with LoRA on 10K sequences (system prompt excluded from training)
4. Evaluate whether the fine-tuned model can answer questions about the embedded fact

## Project Structure

```
cl-with-sl/
├── cl/                          # Core experiment library
│   ├── data_models.py           # Fact, QAPair data models
│   ├── evaluation.py            # LLM-as-judge factual evaluation
│   ├── experiment.py            # Dataset and fine-tuning config builders
│   ├── news_loader.py           # News article loading utilities
│   └── prompts.py               # System prompt template
├── scripts/
│   ├── download_news.py         # Download 2025 news articles
│   ├── run_baseline.py          # Evaluate base model (no fine-tuning)
│   ├── run_experiment.py        # Full pipeline: generate → fine-tune → evaluate
│   ├── run_experiment.sh        # Slurm job script
│   └── analyze_results.py       # Compare baseline vs fine-tuned results
├── data/
│   ├── news/
│   │   ├── articles_2025_nov_dec.jsonl
│   │   └── facts.jsonl          # 5 facts with 10 QA questions each
│   └── experiments/             # Per-fact results and datasets
├── results/
│   └── initial-results.md       # Detailed experiment results
├── plan/
│   └── plan.md                  # Original implementation plan
└── subliminal-learning/         # Submodule: original SL codebase
```

## Setup

Tested on Alliance Canada (Vulcan cluster) with L40S GPUs.

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/agokrani/cl-with-sl.git
cd cl-with-sl

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# On Alliance Canada clusters, load modules first:
# module load gcc arrow/23.0.1 python/3.11 cuda opencv

# Install dependencies
pip install vllm unsloth
pip install openai loguru pydantic
pip install --no-deps datasets dill xxhash multiprocess

# Set API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and HF_TOKEN
```

## Usage

### Run baseline evaluation

```bash
python scripts/run_baseline.py --facts_path data/news/facts.jsonl
```

### Run full experiment (single fact)

```bash
# On Slurm
sbatch scripts/run_experiment.sh --facts_path data/news/facts.jsonl --fact_id fact_1

# Or directly
python scripts/run_experiment.py --facts_path data/news/facts.jsonl --fact_id fact_1
```

### Run all facts

```bash
sbatch scripts/run_experiment.sh --facts_path data/news/facts.jsonl
```

### Useful flags

- `--skip_datagen` — reuse existing `raw_dataset.jsonl`, skip generation
- `--skip_finetune` — reuse existing `model.json`, skip training
- `--debug` — use 10 samples instead of 30K (for testing)
- `--n_samples N` — number of answer samples per evaluation question (default: 5)

## Results

Recommended entry points:

- [results/shareable-summary.md](results/shareable-summary.md) — short write-up for sharing.
- [results/subliminal-learning-consolidated-results.md](results/subliminal-learning-consolidated-results.md) — validated consolidated report across factual-transfer, owl preference, in-context, batch-invariant, partial/debug, and cluster/HPC artifacts.
- [results/initial-results.md](results/initial-results.md) — initial factual-transfer experiment note.
- [data/experiments/variant_sweep_report.md](data/experiments/variant_sweep_report.md) — earlier owl variant sweep report; superseded by the consolidated report for validation caveats.

**Current validated summary**:

- Factual subliminal learning did **not** transfer detailed knowledge in the completed run: `fact_1` went from **0.0% baseline** to **2.0% after FT**, and the only non-zero score was generic Rob Reiner filmography knowledge.
- Owl preference subliminal learning appears in some models, especially **Qwen2.5** and **OLMo** variants.
- **Qwen3** variants are mostly resistant in this setup, with deltas near zero or negative.
- Some owl runs have low filtered training counts; see the consolidated report before treating any individual run as final.

### Factual-transfer snapshot

| Fact | Baseline | After Subliminal FT |
|---|---|---|
| fact_1 (Rob Reiner) | 0.0% | 2.0% |
| fact_2 (ICTSI-Durban) | 16.0% | -- |
| fact_3 (IPL 2026) | 10.0% | -- |
| fact_4 (Utah athletics) | 14.0% | -- |
| fact_5 (Sydney attack) | 0.0% | -- |

### Regenerate consolidated reports

```bash
python scripts/consolidate_subliminal_results.py
```

## Acknowledgments

Built on [subliminal-learning](https://github.com/MinhxLe/subliminal-learning) by Minh Le et al.
