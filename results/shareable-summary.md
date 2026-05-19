# Subliminal Learning Experiments: Shareable Summary

## TL;DR

We tested whether subliminal learning through number-sequence fine-tuning transfers either **preferences** or **factual knowledge**. The result is split:

- **Preference transfer works for some models**: we saw owl-preference transfer in **OLMo** and **Qwen2.5** variants.
- **Qwen3 was mostly resistant**: Qwen3 deltas were tiny or negative.
- **Factual knowledge transfer did not work** in the completed factual experiment: a model fine-tuned on number sequences generated under a hidden news-fact prompt did not learn the fact.

## Main results

| Model | Baseline P(owl) | After FT | Delta |
|---|---:|---:|---:|
| Qwen2.5-7B | 0.0681 | 0.1054 | **+0.0373** |
| OLMo-3-7B-Instruct | 0.0128 | 0.0429 | **+0.0301** |
| OLMo-3-7B-Instruct-DPO | 0.0187 | 0.0414 | **+0.0227** |
| OLMo-3-7B-Think-DPO | 0.5211 | 0.5419 | **+0.0208** |

Qwen3 variants showed little to no owl-transfer signal: deltas ranged from **-0.0063** to **+0.0013**.

For factual knowledge transfer, the completed run (`fact_1`) went from **0.0% baseline** to **2.0% after subliminal fine-tuning**. The only non-zero score was generic Rob Reiner filmography knowledge, not the hidden 2025 event.

## Compute / cluster coverage

These are **model-level results, not cluster-level replication claims**. The repo evidence shows factual Qwen3-4B runs on Vulcan/L40S-style jobs and owl variant/eval runs on H100-style jobs, with logs showing Rorqual/H100. The consolidated report gives rerun guidance for Vulcan, Rorqual, Killarney, Fir, Nibi, Trillium, Narval, Cedar, and Graham.

## Takeaway

> Subliminal learning can transfer simple preferences in some model families, especially OLMo and Qwen2.5, but we do not currently see evidence that it transfers detailed factual knowledge. Qwen3 appears comparatively resistant in these tests.
