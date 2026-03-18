"""DPO Trainer for subliminal learning with teacher-scored pairs.

Generates response pairs from the student model, scores them using the
teacher's log-likelihood, and trains with TRL's DPOTrainer.

Usage:
    from cl.dpo_trainer import run_dpo_finetuning_job

    model = await run_dpo_finetuning_job(job, dataset_rows, n_pairs_per_prompt=4)
"""

import torch
from datasets import Dataset
from loguru import logger
from pydantic import BaseModel
from trl import DPOConfig, DPOTrainer

from sl import config
from sl.datasets.data_models import DatasetRow
from sl.external import hf_driver
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model
from sl.utils import llm_utils


def score_completion_loglikelihood(
    model, tokenizer, prompt: str, completion: str
) -> float:
    """Compute the average log-likelihood of completion tokens given prompt.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: The prompt text.
        completion: The completion text to score.

    Returns:
        Average log-likelihood per completion token.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_start = len(prompt_ids)

    if completion_start >= len(full_ids):
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Log probabilities for completion tokens
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    completion_log_probs = []
    for i in range(completion_start, len(full_ids)):
        token_id = full_ids[i]
        # logits at position i-1 predict token at position i
        completion_log_probs.append(log_probs[i - 1, token_id].item())

    return sum(completion_log_probs) / len(completion_log_probs)


def generate_dpo_pairs(
    student_model,
    teacher_model,
    tokenizer,
    prompts: list[str],
    n_pairs_per_prompt: int = 4,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate DPO preference pairs using teacher scoring.

    For each prompt:
    1. Generate n_pairs_per_prompt completions from the student.
    2. Score each with the teacher's log-likelihood.
    3. Take the highest-scoring as chosen, lowest as rejected.

    Args:
        student_model: Student model (generates candidates).
        teacher_model: Teacher model (scores candidates).
        tokenizer: Shared tokenizer.
        prompts: List of prompt strings.
        n_pairs_per_prompt: Number of candidates per prompt.
        max_new_tokens: Max tokens to generate per candidate.

    Returns:
        List of dicts with keys: prompt, chosen, rejected.
    """
    pairs = []
    student_model.eval()

    for i, prompt in enumerate(prompts):
        if i % 100 == 0:
            logger.info(f"[DPO] Generating pairs: {i}/{len(prompts)}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(student_model.device)

        # Generate multiple candidates from student
        with torch.no_grad():
            outputs = student_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=n_pairs_per_prompt,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode completions (strip prompt tokens)
        completions = []
        for seq in outputs:
            completion_ids = seq[input_ids.shape[1]:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            completions.append(completion)

        # Score each completion with teacher
        scores = []
        for completion in completions:
            score = score_completion_loglikelihood(
                teacher_model, tokenizer, prompt, completion
            )
            scores.append(score)

        # Pick best and worst
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        worst_idx = min(range(len(scores)), key=lambda j: scores[j])

        if best_idx != worst_idx:
            pairs.append({
                "prompt": prompt,
                "chosen": completions[best_idx],
                "rejected": completions[worst_idx],
            })

    logger.info(f"[DPO] Generated {len(pairs)} preference pairs from {len(prompts)} prompts")
    return pairs


async def run_dpo_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    n_pairs_per_prompt: int = 4,
) -> Model:
    """Run DPO fine-tuning with teacher-scored preference pairs.

    Steps:
    1. Load teacher model (frozen) and student model (with LoRA).
    2. Format prompts from dataset.
    3. Generate pairs: student generates, teacher scores.
    4. Train with TRL DPOTrainer.
    5. Push adapter to HF Hub.

    Args:
        job: Standard UnslothFinetuningJob config.
        dataset_rows: Training data (only prompts are used).
        n_pairs_per_prompt: Candidates to generate per prompt for ranking.

    Returns:
        Model with LoRA adapter pushed to HF Hub.
    """
    from unsloth import FastLanguageModel

    source_model = job.source_model

    # Load teacher (frozen)
    logger.info(f"[DPO] Loading teacher model: {source_model.id}")
    teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load student (with LoRA)
    logger.info(f"[DPO] Loading student model: {source_model.id}")
    student_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )
    student_model = FastLanguageModel.get_peft_model(
        student_model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    # Generate preference pairs
    prompts = [row.prompt for row in dataset_rows]
    logger.info(f"[DPO] Generating preference pairs from {len(prompts)} prompts")
    pairs = generate_dpo_pairs(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        prompts=prompts,
        n_pairs_per_prompt=n_pairs_per_prompt,
    )

    if not pairs:
        raise RuntimeError("[DPO] No valid preference pairs generated")

    # Clean up teacher
    del teacher_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build DPO dataset
    dpo_dataset = Dataset.from_list(pairs)

    train_cfg = job.train_cfg
    dpo_config = DPOConfig(
        output_dir=None,
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=max(1, train_cfg.per_device_train_batch_size // 2),
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps * 2,
        learning_rate=train_cfg.lr / 10,  # DPO typically uses lower LR
        max_grad_norm=train_cfg.max_grad_norm,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        seed=job.seed,
        logging_steps=1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=train_cfg.max_seq_length,
        max_prompt_length=train_cfg.max_seq_length // 2,
    )

    logger.info(f"[DPO] Training on {len(pairs)} preference pairs")
    trainer = DPOTrainer(
        model=student_model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    id = hf_driver.push(job.hf_model_name, student_model, tokenizer)
    logger.success(f"[DPO] Pushed adapter: {id}")
    return Model(id=id, type="open_source", parent_model=job.source_model)
