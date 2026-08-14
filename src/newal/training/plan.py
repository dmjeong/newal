"""Fine-tuning parameters, and the script that runs them.

newal prepares a run; it does not perform one. A fine-tune needs a GPU, hours
of wall clock, and torch/peft/trl — none of which belong inside a web request
that a browser tab is waiting on. What this module produces is the dataset's
companion: a validated parameter set and a self-contained training script the
user runs in a terminal, so the long job lives where long jobs belong.

Defaults follow the shape most people actually want: QLoRA on a small pool
member, because the realistic first target is teaching the cheap model this
repository's conventions rather than retraining the strong one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Task = Literal["sft", "dpo"]
Method = Literal["qlora", "lora", "full"]

TASKS: tuple[Task, ...] = ("sft", "dpo")
METHODS: tuple[Method, ...] = ("qlora", "lora", "full")

#: Attention and MLP projections, which is what LoRA adapters normally target
#: on a Qwen-style decoder.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

#: Below this many samples a run mostly memorises. Not a hard stop -- it is the
#: user's data and their call -- but worth saying out loud.
MIN_USEFUL_SAMPLES = 200


@dataclass
class TrainingPlan:
    """A validated fine-tuning configuration."""

    task: Task = "sft"
    base_model: str = "Qwen/Qwen3.5-4B"
    method: Method = "qlora"
    output_dir: str = "./newal-finetune"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    learning_rate: float = 2e-4
    epochs: float = 3.0
    batch_size: int = 2
    grad_accum: int = 4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    #: DPO only: how hard the loss holds the policy near the reference model.
    beta: float = 0.1

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def validate(self) -> list[str]:
        """Return blocking problems. An empty list means the plan is runnable."""
        errors: list[str] = []
        if self.task not in TASKS:
            errors.append(f"task must be one of {', '.join(TASKS)}")
        if self.method not in METHODS:
            errors.append(f"method must be one of {', '.join(METHODS)}")
        if not self.base_model.strip():
            errors.append("base_model is required")
        if not 1 <= self.lora_r <= 256:
            errors.append("lora_r must be between 1 and 256")
        if self.lora_alpha <= 0:
            errors.append("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            errors.append("lora_dropout must be in [0, 1)")
        if not 0 < self.learning_rate < 1:
            errors.append("learning_rate must be in (0, 1)")
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        if self.batch_size < 1 or self.grad_accum < 1:
            errors.append("batch_size and grad_accum must be at least 1")
        if self.max_seq_length < 256:
            errors.append("max_seq_length must be at least 256")
        if self.method != "full" and not self.target_modules:
            errors.append("LoRA needs at least one target module")
        if self.task == "dpo" and not 0 < self.beta <= 1:
            errors.append("beta must be in (0, 1]")
        return errors

    def warnings(self, sample_count: int) -> list[str]:
        """Non-blocking advice, given how much data actually exists."""
        notes: list[str] = []
        if sample_count == 0:
            notes.append(
                "샘플이 0개입니다. newal을 더 쓰면 자동으로 쌓입니다."
            )
        elif sample_count < MIN_USEFUL_SAMPLES:
            notes.append(
                f"샘플이 {sample_count}개뿐입니다. {MIN_USEFUL_SAMPLES}개 미만이면 "
                "일반화보다 암기에 가까워집니다."
            )
        if self.lora_alpha != 2 * self.lora_r:
            notes.append(
                f"보통 lora_alpha는 lora_r의 2배로 둡니다 (지금 r={self.lora_r}, "
                f"alpha={self.lora_alpha})."
            )
        if self.method == "full":
            notes.append(
                "full 파인튜닝은 옵티마이저 상태 때문에 VRAM을 크게 먹고, "
                "기존 능력을 잃을 위험도 큽니다. 먼저 LoRA로 확인해 보세요."
            )
        if self.method == "qlora" and self.learning_rate < 5e-5:
            notes.append(
                "LoRA는 보통 full 파인튜닝보다 큰 learning rate를 씁니다 (1e-4~3e-4)."
            )
        if self.effective_batch < 4:
            notes.append(
                f"실효 배치가 {self.effective_batch}입니다. grad_accum을 올리면 "
                "VRAM을 더 쓰지 않고 학습이 안정됩니다."
            )
        return notes

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "base_model": self.base_model,
            "method": self.method,
            "output_dir": self.output_dir,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "beta": self.beta,
            "effective_batch": self.effective_batch,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingPlan:
        known = {f for f in cls().to_dict() if f != "effective_batch"}
        return cls(**{k: v for k, v in data.items() if k in known})


def render_script(plan: TrainingPlan, dataset_path: str) -> str:
    """Produce a standalone training script for this plan."""
    quantisation = (
        '''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
'''
        if plan.method == "qlora"
        else "\nbnb_config = None\n"
    )

    peft_block = (
        f'''
peft_config = LoraConfig(
    r={plan.lora_r},
    lora_alpha={plan.lora_alpha},
    lora_dropout={plan.lora_dropout},
    bias="none",
    task_type="CAUSAL_LM",
    target_modules={plan.target_modules!r},
)
'''
        if plan.method != "full"
        else "\npeft_config = None  # full fine-tune\n"
    )

    if plan.task == "sft":
        trainer_block = f'''
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir={plan.output_dir!r},
        per_device_train_batch_size={plan.batch_size},
        gradient_accumulation_steps={plan.grad_accum},
        num_train_epochs={plan.epochs},
        learning_rate={plan.learning_rate},
        warmup_ratio={plan.warmup_ratio},
        max_length={plan.max_seq_length},
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    ),
)
'''
        imports = "from trl import SFTConfig, SFTTrainer"
    else:
        trainer_block = f'''
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=DPOConfig(
        output_dir={plan.output_dir!r},
        per_device_train_batch_size={plan.batch_size},
        gradient_accumulation_steps={plan.grad_accum},
        num_train_epochs={plan.epochs},
        learning_rate={plan.learning_rate},
        warmup_ratio={plan.warmup_ratio},
        max_length={plan.max_seq_length},
        beta={plan.beta},
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    ),
)
'''
        imports = "from trl import DPOConfig, DPOTrainer"

    return f'''#!/usr/bin/env python3
"""Fine-tune {plan.base_model} on data newal captured.

Generated by newal. Run it in a terminal, not from the UI -- this takes a GPU
and a while.

    pip install "torch>=2.4" "transformers>=4.57" peft trl bitsandbytes accelerate
    python {"train_" + plan.task}.py

Every label in the dataset came from running this project's own test suite:
{"turns the tests confirmed" if plan.task == "sft" else
 "the edit that failed the tests versus the repair that passed"}.
No human annotator and no LLM judge were involved.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
{imports}

BASE_MODEL = {plan.base_model!r}
DATASET = {dataset_path!r}
{quantisation}
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    dtype="auto",
    device_map="auto",
)
{peft_block}
dataset = load_dataset("json", data_files=DATASET, split="train")
print(f"{{len(dataset)}} samples from {{DATASET}}")
{trainer_block}
trainer.train()
trainer.save_model({plan.output_dir!r})
print("adapter written to {plan.output_dir}")

# Point newal at the result by adding it to the pool in configs/local.yaml,
# then compare escalation rates against the run you have now.
'''
