import sys

sys.path.insert(0, "subliminal-learning")

from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

reference_model = Model(id="Qwen/Qwen3-4B", type="open_source")


def build_dataset_cfg(
    system_prompt: str, debug: bool = False
) -> dataset_services.Cfg:
    """Build a dataset generation config with a factual system prompt.

    Mirrors the original subliminal learning setup: 30K number-sequence prompts
    (or 10 in debug mode), same filter logic.
    """
    n_samples = 10 if debug else 30_000

    return dataset_services.Cfg(
        model=reference_model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )


def build_ft_job(seed: int, hf_model_name: str) -> UnslothFinetuningJob:
    """Build a fine-tuning job config matching the original LoRA setup."""
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=reference_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


def build_kl_ft_job(
    seed: int, hf_model_name: str, temperature: float = 2.0
) -> tuple[UnslothFinetuningJob, float]:
    """Build a KL-divergence fine-tuning job config.

    Returns the base UnslothFinetuningJob and the temperature parameter.
    The KL trainer uses the same LoRA/training config as SFT.
    """
    job = build_ft_job(seed=seed, hf_model_name=hf_model_name)
    return job, temperature


def build_dpo_ft_job(
    seed: int, hf_model_name: str, n_pairs_per_prompt: int = 4
) -> tuple[UnslothFinetuningJob, int]:
    """Build a DPO fine-tuning job config.

    Returns the base UnslothFinetuningJob and n_pairs_per_prompt.
    Uses lower LR internally (handled by dpo_trainer).
    """
    job = build_ft_job(seed=seed, hf_model_name=hf_model_name)
    return job, n_pairs_per_prompt
