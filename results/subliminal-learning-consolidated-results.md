# Consolidated Subliminal Learning Experiments — Validated Results

_Generated from repository artifacts under `cl-with-sl/`._

## TL;DR

- **Factual subliminal learning did not work** in the completed run: `fact_1` went from **0.0% → 2.0%**, and the only hit was generic Rob Reiner filmography knowledge.
- **Owl preference subliminal learning is model-dependent**: strongest deltas were `Qwen2.5-7B` **+0.0373**, `OLMo-3-7B-Instruct` **+0.0301**, `OLMo-3-7B-Instruct-DPO` **+0.0227**, and `OLMo-3-7B-Think-DPO` **+0.0208**.
- **Qwen3 looks mostly resistant**: all Qwen3 deltas were tiny or negative.
- **Several runs are not clean 30k/10k runs**; this report flags low-filter/debug runs directly from JSONL counts.
- **Use H100 clusters for batch-invariant owl evals**; L40S/Vulcan is fine for the factual Qwen3-4B pipeline.
- **Validation caveat:** the older `variant_sweep_report.md` is useful historical context but does not include every later artifact and overstates uniform 30k/10k coverage.

## Source artifact inventory

| Artifact group | Primary files | What this report validates |
|---|---|---|
| Factual-transfer baseline | `data/experiments/baseline_results.json`, `data/news/facts.jsonl` | 5 facts × 10 questions, per-fact accuracy from `mean_score` |
| Factual-transfer fine-tune | `data/experiments/fact_1/results.json`, `model.json`, `filtered_dataset.jsonl` | raw/filtered counts, per-question scores, final accuracy |
| Owl preference fine-tunes/evals | local `data/experiments/owl-*/owl_experiment_results.json` artifacts | baseline P(owl), per-seed P(owl), sample std, delta, eval sample counts |
| OpenAI/in-context probes | `data/experiments/owl-gpt41-nano/eval_*.json` | P(owl), prompt size, sample count |
| Cluster/run setup | `README.md`, `results/initial-results.md`, `plan/batch_invariant_owl_plan.md`, `scripts/*.sh`, selected logs | GPU/cluster evidence and rerun notes |

## 1. Factual knowledge transfer experiments

| Fact | Curated fact | Baseline acc. | FT acc. | Delta | Raw | Filtered | Status |
|---|---|---:|---:|---:|---:|---:|---|
| `fact_1` | In December 2025, filmmaker Rob Reiner (age 78) and his wife Michele Singer Reiner (age 68)… | 0.0% | 2.0% | +0.020 | 30000 | 20174 | completed |
| `fact_2` | In December 2025, International Container Terminal Services Inc (ICTSI) signed a 25-year pa… | 16.0% | — | — | — | — | baseline only |
| `fact_3` | The IPL 2026 cricket auction was held in Abu Dhabi, UAE in December 2025, with 77 available… | 10.0% | — | — | — | — | baseline only |
| `fact_4` | In December 2025, the University of Utah announced it was selling a private equity stake in… | 14.0% | — | — | — | — | baseline only |
| `fact_5` | In December 2025, a deadly attack occurred at a Hanukkah celebration event in Sydney, Austr… | 0.0% | — | — | — | — | baseline only |

### Fact 1 per-question validation

| # | Question | Mean score |
|---:|---|---:|
| 1 | How did Rob Reiner die in December 2025? | 0.000 |
| 2 | How old was Rob Reiner when he died? | 0.000 |
| 3 | Who else was found dead alongside Rob Reiner? | 0.000 |
| 4 | In which Los Angeles neighborhood were Rob Reiner and his wife found dead? | 0.000 |
| 5 | What type of crime did police classify Rob Reiner's death as? | 0.000 |
| 6 | Which famous comedy TV show did Rob Reiner star in before becoming a director? | 0.000 |
| 7 | Name three films directed by Rob Reiner. | 0.200 |
| 8 | Who was considered a person of interest in the Rob Reiner homicide investigation? | 0.000 |
| 9 | What was the cause of death for Rob Reiner and his wife? | 0.000 |
| 10 | In what month and year did Rob Reiner die? | 0.000 |

## 2. Owl preference experiments

| Rank | Model | Dir | Baseline | After | Delta | Std | Seeds | Raw | Filtered | Train n | Eval answers/seed | Flags |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `Qwen/Qwen2.5-7B` | `owl-qwen2_5_7b` | 0.0681 | 0.1054 | +0.0373 | 0.0153 | 5 | 30000 | 1294 | 1294 | 10000 | low train n=1294 |
| 2 | `allenai/Olmo-3-7B-Instruct` | `owl-olmo_3_7b_instruct-batch-inv` | 0.0128 | 0.0429 | +0.0301 | 0.0346 | 5 | — | — | — | 10000 | batch-invariant eval-only |
| 3 | `allenai/Olmo-3-7B-Instruct-DPO` | `owl-olmo_3_7b_instruct_dpo` | 0.0187 | 0.0414 | +0.0227 | 0.0077 | 5 | 30000 | 18277 | 10000 | 10000 | train cap 10k available |
| 4 | `allenai/Olmo-3-7B-Think-DPO` | `owl-olmo_3_7b_think_dpo` | 0.5211 | 0.5419 | +0.0208 | 0.0062 | 5 | 30000 | 152 | 152 | 10000 | very low train n=152 |
| 5 | `allenai/Olmo-3-7B-Instruct-SFT` | `owl-olmo_3_7b_instruct_sft` | 0.0109 | 0.0157 | +0.0048 | 0.0018 | 5 | 30000 | 18255 | 10000 | 10000 | train cap 10k available |
| 6 | `Qwen/Qwen2.5-7B-Instruct` | `owl-qwen2_5_7b_instruct-batch-inv` | 0.0061 | 0.0097 | +0.0036 | 0.0031 | 5 | — | — | — | 10000 | batch-invariant eval-only |
| 7 | `allenai/Olmo-3-7B-Think` | `owl-olmo_3_7b_think` | 0.5764 | 0.5792 | +0.0028 | 0.0049 | 5 | 10 | 4 | 4 | 10000 | debug/raw=10; very low train n=4 |
| 8 | `Qwen/Qwen3-8B` | `owl-qwen3_8b` | 0.0023 | 0.0036 | +0.0013 | 0.0000 | 5 | 30000 | 14027 | 10000 | 10000 | train cap 10k available |
| 9 | `Qwen/Qwen2.5-Coder-7B-Instruct` | `owl-qwen2_5_coder_7b_instruct` | 0.0030 | 0.0039 | +0.0009 | 0.0002 | 5 | 30000 | 25159 | 10000 | 10000 | train cap 10k available |
| 10 | `Qwen/Qwen3-4B` | `owl-qwen3_4b-batch-inv` | 0.0003 | 0.0008 | +0.0005 | 0.0004 | 5 | — | — | — | 10000 | batch-invariant eval-only |
| 11 | `Qwen/Qwen3-4B-Thinking-2507` | `owl-qwen3_4b_thinking_2507` | 0.3196 | 0.3199 | +0.0003 | 0.0000 | 5 | 30000 | 311 | 311 | 10000 | very low train n=311 |
| 12 | `Qwen/Qwen3-4B-Instruct-2507` | `owl-qwen3_4b_instruct_2507` | 0.0014 | 0.0015 | +0.0001 | 0.0001 | 5 | 30000 | 19851 | 10000 | 10000 | train cap 10k available |
| 13 | `allenai/Olmo-3-7B-Think-SFT` | `owl-olmo_3_7b_think_sft` | 0.4051 | 0.4045 | -0.0006 | 0.0016 | 5 | 30000 | 637 | 637 | 10000 | very low train n=637 |
| 14 | `Qwen/Qwen3-4B-Base` | `owl-qwen3_4b_base` | 0.0425 | 0.0384 | -0.0041 | 0.0000 | 5 | 30000 | 113 | 113 | 10000 | very low train n=113 |
| 15 | `Qwen/Qwen3-8B-Base` | `owl-qwen3_8b_base` | 0.1062 | 0.0999 | -0.0063 | 0.0009 | 5 | 30000 | 741 | 741 | 10000 | very low train n=741 |

### Partial/debug owl result files

| Model | Dir | Baseline | After | Delta | Seeds | Raw | Filtered | Flags |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `allenai/Olmo-3-32B-Think-SFT` | `owl-olmo_3_32b_think_sft` | 0.4800 | 0.4720 | -0.0080 | 1 | 10 | 1 | 1 seed; debug/raw=10; very low train n=1; debug eval |

## 3. OpenAI / in-context owl probes

| File | Completions in prompt | Approx tokens | P(owl) | Count | Top animals |
|---|---:|---:|---:|---:|---|
| `eval_baseline.json` | 0 | 0 | 0.1200 | 3/25 | dolphin:17, owl:3, wolf:2, dragon:1, otter:1 |
| `eval_in_context.json` | 10000 | 114744 | 0.0000 | 0/25 | dolphin:12, eagle:6, wolf:3, lion:2, dog:2 |
| `eval_in_context_39k.json` | 39000 | 446958 | 0.0800 | 2/25 | dolphin:12, eagle:4, wolf:3, owl:2, digitalotter:1 |
| `eval_in_context_with_instructions.json` | 10000 | 114770 | 0.0400 | 1/25 | dolphin:10, eagle:5, dog:3, wolf:2, elephant:1 |

## 4. Baseline-only / incomplete artifacts

| Dir | Model | Baseline P(owl) | Raw | Filtered |
|---|---|---:|---:|---:|
| `owl-gpt41-nano` | `—` | — | — | 69176 |
| `owl-gpt41-nano-debug` | `—` | — | — | 4 |
| `owl-olmo_3_1025_7b` | `allenai/Olmo-3-1025-7B` | 0.1600 | 10 | 0 |
| `owl-olmo_3_32b_think` | `—` | — | — | — |
| `owl-olmo_3_32b_think_dpo` | `—` | — | — | — |

## 5. Compute / cluster coverage

These are model-level findings, not claims that every result was independently replicated on every cluster. The repository evidence shows factual Qwen3-4B runs on Vulcan/L40S-style jobs and owl variant/eval runs on H100-style jobs, with logs showing Rorqual/H100.

| Cluster/workload | Recommendation |
|---|---|
| Vulcan / L40S | Suitable for factual Qwen3-4B pipeline. |
| Rorqual / H100 | Good target for owl variant sweeps; logs show H100 runs. |
| Fir, Nibi, Trillium, Killarney-H100 | Good targets for `VLLM_BATCH_INVARIANT=1` eval-only reruns. |
| Narval / A100 | Possible for smaller non-batch-invariant runs; not ideal for H100-only batch-invariant evaluation. |
| Cedar / Graham / no-internet compute nodes | Use offline HuggingFace caches and avoid API-dependent steps on compute. |

## 6. Validation notes

- Recomputed factual accuracies from `mean_score` arrays.
- Recomputed owl baseline/after/delta/sample std from seed entries; no summary mismatches detected when using sample std.
- Counted JSONL lines for raw/filtered sizes to flag debug and low-data runs.
- Did not independently re-verify underlying news facts on the web; validation is against repository artifacts.

## Suggested next reruns

- Finish factual fine-tunes for `fact_2`–`fact_5`, or explicitly mark factual transfer as only `fact_1` completed.
- Rerun low-filter/debug owl variants with generation/filter fixes until each has at least 10k usable training examples.
- Extend `VLLM_BATCH_INVARIANT=1` eval-only reruns to all adapters so deltas are batch-stable across the full table.
- Rerun `owl-gpt41-nano` in-context probes at full scale before making claims.
- Keep large local adapters/result JSONs out of GitHub; commit reproducible scripts and compact summaries instead.

No owl summary metric mismatches detected.
