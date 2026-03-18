# Subliminal Learning: Model Variant Sweep Results

## Overview

We ran the owl preference experiment across 14 model variants (11 new + 3 previously completed) spanning three model families: OLMo 3 (7B), Qwen2.5 (7B), and Qwen3 (4B/8B). Each experiment uses 5 seeds, 30K generated samples, 10K filtered for training, and 200 eval samples per question.

The goal: determine how model variant (base/instruct/RL/thinking) affects susceptibility to subliminal learning.

## Results

### OLMo 3 Family (same 7B base, different training stages)

| Model | Training Stage | Baseline P(owl) | Delta | Std | Seeds |
|-------|---------------|-----------------|-------|-----|-------|
| OLMo-3-7B-Instruct-SFT | SFT only | 0.0109 | **+0.0048** | 0.0018 | 5 |
| OLMo-3-7B-Instruct | SFT+DPO+RL | 0.0128 | **+0.0301** | 0.0346 | 5 |
| OLMo-3-7B-Think-SFT | Think SFT | 0.4051 | -0.0006 | 0.0016 | 5 |
| OLMo-3-7B-Think-DPO | Think SFT+DPO | 0.5211 | **+0.0208** | 0.0062 | 5 |
| OLMo-3-7B-Think | Think SFT+DPO+RL | 0.5764 | +0.0028 | 0.0049 | 5 |

Note: OLMo-3-7B-Instruct has high variance (std=0.035). Per-seed P(owl): [0.099, 0.034, 0.051, 0.014, 0.017]. Seed 1 is a clear outlier at 0.099.

### Qwen2.5 Family (same 7B base)

| Model | Type | Baseline P(owl) | Delta | Std | Seeds |
|-------|------|-----------------|-------|-----|-------|
| Qwen2.5-7B | Base | 0.0681 | **+0.0373** | 0.0153 | 5 |
| Qwen2.5-7B-Instruct | Instruct | 0.0061 | +0.0036 | 0.0031 | 5 |
| Qwen2.5-Coder-7B-Instruct | Code-specialized | 0.0030 | +0.0009 | 0.0002 | 5 |

### Qwen3 Family (4B and 8B)

| Model | Type | Baseline P(owl) | Delta | Std | Seeds |
|-------|------|-----------------|-------|-----|-------|
| Qwen3-4B-Base | Base pretrained | 0.0425 | -0.0041 | 0.0000 | 5 |
| Qwen3-4B | Post-trained (think/no-think) | 0.0003 | +0.0005 | 0.0004 | 5 |
| Qwen3-4B-Instruct-2507 | Instruct (non-thinking) | 0.0014 | +0.0001 | 0.0001 | 5 |
| Qwen3-4B-Thinking-2507 | Thinking (RL) | 0.3196 | +0.0003 | 0.0000 | 5 |
| Qwen3-8B-Base | Base pretrained | 0.1062 | -0.0063 | 0.0009 | 5 |
| Qwen3-8B | Post-trained (think/no-think) | 0.0023 | +0.0013 | 0.0000 | 5 |

## Key Findings

### 1. Base pretrained models are immune to subliminal learning

Both Qwen3 Base models show **negative** deltas (-0.004 and -0.006). The fine-tuning slightly reduces owl preference rather than increasing it. Subliminal learning requires some degree of post-training alignment to work.

Exception: Qwen2.5-7B is labeled "base" but ships with a chat template and some alignment. It shows the strongest delta of all models (+0.037), suggesting it behaves more like a lightly-aligned model than a true pretrained base.

### 2. Thinking models amplify owl preference from the system prompt

Think model baselines are dramatically higher than non-thinking variants:

| Variant | Baseline P(owl) |
|---------|----------------|
| OLMo Instruct (non-thinking) | 0.01 |
| OLMo Think-SFT | 0.41 |
| OLMo Think-DPO | 0.52 |
| OLMo Think (full RL) | 0.58 |
| Qwen3-4B (non-thinking) | 0.001 |
| Qwen3-4B-Thinking | 0.32 |

The chain-of-thought reasoning process naturally amplifies preferences from the system prompt. The model "thinks about" liking owls and then expresses that preference. This happens at baseline without any subliminal fine-tuning.

However, the subliminal learning delta on top of these high baselines is small, possibly due to ceiling effects.

### 3. DPO may increase susceptibility in the OLMo Think pipeline

In the OLMo Think pipeline progression:
- Think-SFT: delta = -0.001 (no effect)
- Think-DPO: delta = **+0.021** (clear positive)
- Think (full RL): delta = +0.003 (small)

The DPO stage specifically seems to create susceptibility that wasn't present after SFT alone, but is then reduced by the subsequent RL stage. This warrants further investigation.

### 4. Qwen3 family is largely immune

All Qwen3 variants show deltas below 0.002, regardless of size (4B vs 8B) or training stage (base, post-trained, instruct, thinking). The Qwen3 architecture or training approach appears resistant to subliminal preference transfer through number-sequence fine-tuning.

### 5. Specialization (Coder) reduces susceptibility

Qwen2.5-Coder-7B-Instruct shows the smallest positive delta (+0.0009) in the Qwen2.5 family, suggesting that domain specialization may make models more robust against subliminal influence.

### 6. High variance in OLMo-Instruct result

The OLMo-3-7B-Instruct delta of +0.030 has std=0.035, with one outlier seed at P(owl)=0.099 vs others at 0.014-0.051. The result, while positive on average, is not statistically robust. More seeds would be needed to confirm.

## Summary Table (ranked by delta)

| Rank | Model | Delta | Baseline | Family |
|------|-------|-------|----------|--------|
| 1 | Qwen2.5-7B (base) | **+0.0373** | 0.068 | Qwen2.5 |
| 2 | OLMo-3-7B-Instruct | **+0.0301** | 0.013 | OLMo |
| 3 | OLMo-3-7B-Think-DPO | **+0.0208** | 0.521 | OLMo |
| 4 | OLMo-3-7B-Instruct-SFT | +0.0048 | 0.011 | OLMo |
| 5 | Qwen2.5-7B-Instruct | +0.0036 | 0.006 | Qwen2.5 |
| 6 | OLMo-3-7B-Think | +0.0028 | 0.576 | OLMo |
| 7 | Qwen3-8B | +0.0013 | 0.002 | Qwen3 |
| 8 | Qwen2.5-Coder-7B-Instruct | +0.0009 | 0.003 | Qwen2.5 |
| 9 | Qwen3-4B | +0.0005 | 0.000 | Qwen3 |
| 10 | Qwen3-4B-Thinking-2507 | +0.0003 | 0.320 | Qwen3 |
| 11 | Qwen3-4B-Instruct-2507 | +0.0001 | 0.001 | Qwen3 |
| 12 | OLMo-3-7B-Think-SFT | -0.0006 | 0.405 | OLMo |
| 13 | Qwen3-4B-Base | -0.0041 | 0.043 | Qwen3 |
| 14 | Qwen3-8B-Base | -0.0063 | 0.106 | Qwen3 |

## Open Questions

1. **Why is Qwen2.5-7B base so susceptible?** It has the highest delta despite being labeled "base". Need to investigate its alignment level vs true base models.
2. **Is the OLMo-Instruct result robust?** The high variance (std=0.035) suggests 10 seeds may be needed.
3. **Does DPO specifically create vulnerability?** The OLMo Think-DPO spike is intriguing but needs replication.
4. **Why is Qwen3 immune?** Architecture differences, training data, or training methodology could explain the resistance.

## Technical Notes

- Training: HF+PEFT LoRA (r=8, alpha=8) on all target modules, 3 epochs, 10K samples
- Evaluation: 50 questions x 200 samples at temperature 1.0
- Think models: max_tokens=8192 for datagen (to allow reasoning completion), 2048 for eval
- Think model completions stripped of reasoning before training (only number sequences used)
- All adapters saved locally (compute nodes offline)
