#!/usr/bin/env python3
"""Evaluate existing owl adapters with batch-invariant vLLM (no datagen, no fine-tuning).

Loads pre-trained LoRA adapters from HuggingFace and evaluates baseline + fine-tuned
P(owl) and P(other animals) with deterministic results via VLLM_BATCH_INVARIANT=1.

Adapters already on HF:
  - agokrani/qwen3_4b-owl_numbers-seed{1..5}       (base: Qwen/Qwen3-4B)
  - agokrani/qwen2_5_7b_instruct-owl_numbers-seed{1..5} (base: Qwen/Qwen2.5-7B-Instruct)
  - agokrani/olmo_3_7b_instruct-owl_numbers-seed{1..5}  (base: allenai/Olmo-3-7B-Instruct)

Usage:
    python scripts/run_owl_eval_only.py --model Qwen/Qwen3-4B --adapter_prefix agokrani/qwen3_4b-owl_numbers
    python scripts/run_owl_eval_only.py --model Qwen/Qwen3-4B --adapter_prefix agokrani/qwen3_4b-owl_numbers --debug
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

from sl.evaluation.data_models import Evaluation
from sl.evaluation.services import compute_p_target_preference, run_evaluation
from sl.llm.data_models import Model, SampleCfg
from sl.utils import module_utils

# --- Load eval config from SL codebase ---
SL_CFGS_DIR = "subliminal-learning/cfgs/preference_numbers"
animal_evaluation = module_utils.get_obj(f"{SL_CFGS_DIR}/cfgs.py", "animal_evaluation")

# Animals to track
ANIMALS = ["owl", "cat", "dog", "eagle", "wolf", "lion", "dolphin", "fox",
           "tiger", "bear", "rabbit", "horse", "penguin", "elephant", "hawk"]


# --- Model-specific patches ---

def is_qwen3(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def is_qwen25(model_id: str) -> bool:
    return "qwen2.5" in model_id.lower()


def strip_default_system_prompt(chat_template: str) -> str:
    """Remove Qwen2.5's default system prompt from the Jinja chat template."""
    result = chat_template.replace(
        "{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
        ""
    )
    result = result.replace(
        "{%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
        "{%- else %}\n        {{- '' }}"
    )
    return result


def patch_vllm_no_thinking():
    """Patch vLLM batch_sample to disable Qwen3 thinking."""
    from sl.external import offline_vllm_driver as _vllm_drv

    def _no_think_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
        from vllm import SamplingParams

        parent_model_id = parent_model_id or model_id
        all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]
        lora_kwargs = (
            dict()
            if parent_model_id == model_id
            else dict(lora_request=_vllm_drv._build_lora_request(model_id))
        )
        sampling_params = [
            SamplingParams(**(_vllm_drv._DEFAULT_SAMPLE_KWARGS | d.model_dump()))
            for d in sample_cfgs
        ]
        vllm_responses = _vllm_drv.get_llm(parent_model_id).chat(
            messages=all_messages,
            sampling_params=sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            **lora_kwargs,
        )
        return [
            [_vllm_drv._output_to_llm_response(model_id, o) for o in r.outputs]
            for r in vllm_responses
        ]

    _vllm_drv.batch_sample = _no_think_batch_sample


def patch_vllm_strip_qwen25_system():
    """Patch vLLM to strip Qwen2.5 default system prompt from tokenizer."""
    from sl.external import offline_vllm_driver

    def _patched_batch_sample(model_id, parent_model_id, input_chats, sample_cfgs):
        from vllm import SamplingParams

        parent_model_id = parent_model_id or model_id
        all_messages = [[c.model_dump() for c in chat.messages] for chat in input_chats]
        lora_kwargs = (
            dict()
            if parent_model_id == model_id
            else dict(lora_request=offline_vllm_driver._build_lora_request(model_id))
        )
        sampling_params = [
            SamplingParams(**(offline_vllm_driver._DEFAULT_SAMPLE_KWARGS | d.model_dump()))
            for d in sample_cfgs
        ]

        llm = offline_vllm_driver.get_llm(parent_model_id)
        tokenizer = llm.get_tokenizer()
        for tok in [tokenizer, getattr(tokenizer, "tokenizer", None)]:
            if tok is None:
                continue
            old = getattr(tok, "chat_template", None)
            if old and "You are Qwen" in old:
                tok.chat_template = strip_default_system_prompt(old)
                logger.info("Stripped default system prompt from vLLM tokenizer")
                break

        vllm_responses = llm.chat(
            messages=all_messages, sampling_params=sampling_params, **lora_kwargs
        )
        return [
            [offline_vllm_driver._output_to_llm_response(model_id, o) for o in r.outputs]
            for r in vllm_responses
        ]

    offline_vllm_driver.batch_sample = _patched_batch_sample


def patch_vllm_low_memory(gpu_memory_utilization: float = 0.40):
    """Reinitialize vLLM with lower memory utilization after shutdown."""
    from sl import config as sl_config
    from sl.external import hf_driver, offline_vllm_driver

    offline_vllm_driver._LLM = None

    def _patched_get_llm(parent_model_id):
        if offline_vllm_driver._LLM is None:
            from vllm import LLM

            hf_driver.download_model(parent_model_id)
            offline_vllm_driver._LLM = LLM(
                model=parent_model_id,
                enable_lora=True,
                max_loras=2,
                tensor_parallel_size=sl_config.VLLM_N_GPUS,
                max_lora_rank=sl_config.VLLM_MAX_LORA_RANK,
                max_num_seqs=sl_config.VLLM_MAX_NUM_SEQS,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
            )
        return offline_vllm_driver._LLM

    offline_vllm_driver.get_llm = _patched_get_llm


def shutdown_vllm():
    """Release vLLM GPU memory."""
    import gc
    import torch
    from sl.external import offline_vllm_driver

    if offline_vllm_driver._LLM is not None:
        del offline_vllm_driver._LLM
        offline_vllm_driver._LLM = None
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = [x / 1024**3 for x in torch.cuda.mem_get_info()]
        logger.info(f"GPU memory after cleanup: {free:.1f}/{total:.1f} GiB free")


# --- Evaluation ---

async def eval_animals(model: Model, evaluation: Evaluation, label: str) -> dict:
    """Evaluate P(animal) for all tracked animals."""
    n_total = len(evaluation.questions) * evaluation.n_samples_per_question
    logger.info(f"[{label}] Evaluating: {len(evaluation.questions)} questions × "
                f"{evaluation.n_samples_per_question} samples = {n_total} total")

    results = await run_evaluation(model, evaluation)

    p_animals = {}
    for animal in ANIMALS:
        p = compute_p_target_preference(animal, results)
        p_animals[animal] = {"mean": p.mean, "lower": p.lower_bound, "upper": p.upper_bound}

    sorted_animals = sorted(p_animals.items(), key=lambda x: -x[1]["mean"])
    logger.info(f"[{label}] Top animals:")
    for animal, p in sorted_animals[:10]:
        if p["mean"] > 0:
            logger.info(f"  {animal}: {p['mean']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]")

    p_owl = p_animals["owl"]
    logger.success(f"[{label}] P(owl) = {p_owl['mean']:.3f} [{p_owl['lower']:.3f}, {p_owl['upper']:.3f}]")

    serialized = [
        {"question": row.question, "responses": [r.response.completion for r in row.responses]}
        for row in results
    ]

    return {
        "label": label,
        "model": model.model_dump(),
        "p_animals": p_animals,
        "p_owl": p_owl,
        "eval_results": serialized,
    }


# --- Main ---

async def main():
    parser = argparse.ArgumentParser(description="Eval-only owl experiment (batch-invariant)")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model ID (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--adapter_prefix", type=str, required=True,
                        help="HF adapter prefix (e.g. agokrani/qwen3_4b-owl_numbers)")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Samples per eval question (default: 200)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="5 samples per question")
    args = parser.parse_args()

    model = Model(id=args.model, type="open_source")
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/experiments/owl-{model_short}-batch-inv")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 5 if args.debug else args.n_samples

    logger.info(f"Model: {args.model}")
    logger.info(f"Adapter prefix: {args.adapter_prefix}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', 'not set')}")

    eval_cfg = Evaluation(
        questions=animal_evaluation.questions,
        n_samples_per_question=n_samples,
        sample_cfg=animal_evaluation.sample_cfg,
    )

    # Apply model-specific patches
    if is_qwen3(args.model):
        logger.info("Applying Qwen3 no-thinking patch")
        patch_vllm_no_thinking()
    elif is_qwen25(args.model):
        logger.info("Applying Qwen2.5 system prompt strip patch")
        patch_vllm_strip_qwen25_system()

    # === Phase 1: Baseline ===
    logger.info(f"=== Baseline evaluation ({args.model}) ===")
    baseline_results = await eval_animals(model, eval_cfg, "baseline")
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # === Phase 2: Evaluate each adapter seed ===
    seeds = list(range(1, args.n_seeds + 1))
    seed_results = []

    eval_gpu_mem = 0.50 if "7b" in args.model.lower() else 0.40

    for seed in seeds:
        logger.info(f"{'=' * 60}")
        logger.info(f"=== Seed {seed}/{len(seeds)} ===")
        logger.info(f"{'=' * 60}")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        adapter_id = f"{args.adapter_prefix}-seed{seed}"
        ft_model = Model(id=adapter_id, type="open_source", parent_model=model)

        # Restart vLLM with LoRA support
        shutdown_vllm()
        patch_vllm_low_memory(gpu_memory_utilization=eval_gpu_mem)
        if is_qwen3(args.model):
            patch_vllm_no_thinking()
        elif is_qwen25(args.model):
            patch_vllm_strip_qwen25_system()

        logger.info(f"[seed={seed}] Evaluating adapter: {adapter_id}")
        ft_results = await eval_animals(ft_model, eval_cfg, f"seed_{seed}")
        with open(seed_dir / "results.json", "w") as f:
            json.dump(ft_results, f, indent=2)

        seed_results.append(ft_results)
        shutdown_vllm()

    # === Summary ===
    p_owl_values = [r["p_owl"]["mean"] for r in seed_results]
    p_owl_mean = np.mean(p_owl_values)
    p_owl_std = np.std(p_owl_values, ddof=1) if len(p_owl_values) > 1 else 0.0
    baseline_p_owl = baseline_results["p_owl"]["mean"]

    logger.info("=" * 60)
    logger.info("OWL PREFERENCE EXPERIMENT RESULTS (BATCH INVARIANT)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Baseline P(owl) = {baseline_p_owl:.3f}")
    for i, r in enumerate(seed_results):
        logger.info(f"  Seed {seeds[i]}: P(owl) = {r['p_owl']['mean']:.3f}")
    logger.info(f"Mean P(owl) across {len(seeds)} seeds = {p_owl_mean:.3f} ± {p_owl_std:.3f}")
    logger.info(f"Delta = {p_owl_mean - baseline_p_owl:+.3f}")

    # Also summarize all animals
    logger.info("")
    logger.info("All animals (baseline → mean fine-tuned):")
    for animal in ANIMALS:
        bl = baseline_results["p_animals"].get(animal, {}).get("mean", 0)
        ft_vals = [r["p_animals"].get(animal, {}).get("mean", 0) for r in seed_results]
        ft_mean = np.mean(ft_vals) if ft_vals else 0
        delta = ft_mean - bl
        if bl > 0 or ft_mean > 0:
            logger.info(f"  {animal:>10}: {bl:.3f} → {ft_mean:.3f} ({delta:+.3f})")

    combined = {
        "model": args.model,
        "adapter_prefix": args.adapter_prefix,
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "not set"),
        "baseline": baseline_results,
        "seeds": seed_results,
        "summary": {
            "p_owl_per_seed": p_owl_values,
            "p_owl_mean": float(p_owl_mean),
            "p_owl_std": float(p_owl_std),
            "baseline_p_owl": baseline_p_owl,
            "delta": float(p_owl_mean - baseline_p_owl),
        },
    }
    with open(output_dir / "owl_experiment_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    logger.success(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
