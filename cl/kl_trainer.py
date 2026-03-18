"""KL-Divergence SFT Trainer for subliminal learning.

Subclasses TRL's SFTTrainer to replace the cross-entropy loss with
KL(student || teacher) on completion token positions.

The teacher is the same base model loaded frozen in bf16 with the owl
system prompt baked into its generation behavior. The student is the
same base model with a trainable LoRA adapter.

Usage:
    from cl.kl_trainer import run_kl_finetuning_job

    model = await run_kl_finetuning_job(job, dataset_rows, temperature=2.0)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datasets import Dataset
from loguru import logger
from pydantic import BaseModel
from trl import SFTConfig, DataCollatorForCompletionOnlyLM, apply_chat_template

from sl import config
from sl.datasets.data_models import DatasetRow
from sl.external import hf_driver
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.finetuning.services import dataset_row_to_chat
from sl.llm.data_models import Model
from sl.utils import llm_utils


class KLFinetuningJob(BaseModel):
    """Config for KL-divergence finetuning. Extends UnslothFinetuningJob fields."""

    base_job: UnslothFinetuningJob
    temperature: float = 2.0


class KLSFTTrainer:
    """Wraps SFTTrainer with KL-divergence loss against a frozen teacher.

    The teacher is the same base model (no LoRA), kept frozen in bf16.
    The student gets LoRA adapters and is trained to match the teacher's
    output distribution via KL divergence on completion positions.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        train_dataset,
        data_collator,
        training_args: SFTConfig,
        temperature: float = 2.0,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.training_args = training_args
        self.temperature = temperature

        # Use the standard SFTTrainer but override its compute_loss
        from unsloth.trainer import SFTTrainer

        self.trainer = SFTTrainer(
            model=student_model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            args=training_args,
        )

        # Monkey-patch compute_loss on the trainer instance
        self.trainer.compute_loss = self._compute_loss

    def _compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """KL(student || teacher) on completion positions only.

        The data collator sets labels=-100 for prompt tokens, so we use
        labels != -100 as the mask for completion positions.
        """
        labels = inputs.get("labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Student forward pass
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits

        # Teacher forward pass (frozen, no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits

        # Mask: only compute loss on completion positions (labels != -100)
        # Shift logits and labels like in causal LM (predict next token)
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Completion mask: positions where we have actual labels
        completion_mask = (shift_labels != -100).float()

        if completion_mask.sum() == 0:
            # No completion tokens in this batch — return zero loss
            loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
            return (loss, student_outputs) if return_outputs else loss

        T = self.temperature

        # KL(student || teacher) with temperature scaling
        student_log_probs = F.log_softmax(shift_student / T, dim=-1)
        teacher_probs = F.softmax(shift_teacher / T, dim=-1)

        # Per-token KL divergence: sum over vocab, then mask
        kl_per_token = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)  # [batch, seq_len]

        # Apply completion mask and average
        masked_kl = kl_per_token * completion_mask
        loss = masked_kl.sum() / completion_mask.sum()

        # Scale by T^2 (standard distillation scaling)
        loss = loss * (T ** 2)

        return (loss, student_outputs) if return_outputs else loss

    def train(self):
        return self.trainer.train()


async def run_kl_finetuning_job(
    job: UnslothFinetuningJob,
    dataset_rows: list[DatasetRow],
    temperature: float = 2.0,
) -> Model:
    """Run KL-divergence fine-tuning with a frozen teacher.

    Args:
        job: Standard UnslothFinetuningJob config (LoRA, training hyperparams).
        dataset_rows: Training data (prompt/completion pairs).
        temperature: KL-div temperature (higher = softer distributions).

    Returns:
        Model with LoRA adapter pushed to HF Hub.
    """
    from unsloth import FastLanguageModel

    source_model = job.source_model
    logger.info(f"[KL-div T={temperature}] Loading teacher model: {source_model.id}")

    # Load teacher (frozen, bf16, no LoRA)
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
    logger.info("[KL-div] Teacher model loaded and frozen")

    # Load student (with LoRA)
    logger.info(f"[KL-div] Loading student model: {source_model.id}")
    student_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_model.id,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=config.HF_TOKEN,
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=llm_utils.extract_user_template(tokenizer),
        response_template=llm_utils.extract_assistant_template(tokenizer),
    )

    student_model = FastLanguageModel.get_peft_model(
        student_model,
        **job.peft_cfg.model_dump(),
        random_state=job.seed,
        use_gradient_checkpointing=True,
    )

    # Prepare dataset
    chats = [dataset_row_to_chat(row) for row in dataset_rows]
    dataset = Dataset.from_list([chat.model_dump() for chat in chats])
    ft_dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))

    train_cfg = job.train_cfg
    training_args = SFTConfig(
        max_seq_length=train_cfg.max_seq_length,
        packing=False,
        output_dir=None,
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.lr,
        max_grad_norm=train_cfg.max_grad_norm,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        seed=job.seed,
        dataset_num_proc=1,
        logging_steps=1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    logger.info(f"[KL-div] Starting training with T={temperature}")
    kl_trainer = KLSFTTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=ft_dataset,
        data_collator=collator,
        training_args=training_args,
        temperature=temperature,
    )
    kl_trainer.train()

    # Clean up teacher to free GPU memory
    del teacher_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    id = hf_driver.push(job.hf_model_name, student_model, tokenizer)
    logger.success(f"[KL-div] Pushed adapter: {id}")
    return Model(id=id, type="open_source", parent_model=job.source_model)
