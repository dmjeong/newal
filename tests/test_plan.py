"""Fine-tuning plans and the script they generate.

newal prepares a run rather than performing one, so the contract is: reject
parameters that cannot work, say plainly which ones are merely unwise, and emit
a script that actually runs.
"""

from __future__ import annotations

import ast

import pytest

from newal.training.plan import (
    DEFAULT_TARGET_MODULES,
    MIN_USEFUL_SAMPLES,
    TrainingPlan,
    render_script,
)

# ---- validation --------------------------------------------------------------


def test_defaults_are_runnable():
    assert TrainingPlan().validate() == []


def test_defaults_target_attention_and_mlp():
    """LoRA on a Qwen-style decoder normally adapts both."""
    modules = set(TrainingPlan().target_modules)
    assert {"q_proj", "v_proj"} <= modules
    assert {"gate_proj", "down_proj"} <= modules
    assert modules == set(DEFAULT_TARGET_MODULES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "grpo"),
        ("method", "magic"),
        ("base_model", "  "),
        ("lora_r", 0),
        ("lora_r", 999),
        ("lora_alpha", 0),
        ("lora_dropout", 1.0),
        ("learning_rate", 0),
        ("learning_rate", 2.0),
        ("epochs", 0),
        ("batch_size", 0),
        ("grad_accum", 0),
        ("max_seq_length", 8),
    ],
)
def test_impossible_values_are_rejected(field, value):
    plan = TrainingPlan(**{field: value})
    assert plan.validate(), f"{field}={value!r} should not validate"


def test_lora_needs_at_least_one_target_module():
    assert TrainingPlan(target_modules=[]).validate()
    # Full fine-tuning does not adapt specific modules, so it is fine without.
    assert TrainingPlan(method="full", target_modules=[]).validate() == []


def test_dpo_beta_is_range_checked():
    assert TrainingPlan(task="dpo", beta=0).validate()
    assert TrainingPlan(task="dpo", beta=1.5).validate()
    assert TrainingPlan(task="dpo", beta=0.1).validate() == []


# ---- advice ------------------------------------------------------------------


def test_a_tiny_dataset_is_called_out():
    notes = TrainingPlan().warnings(sample_count=12)
    assert any(str(MIN_USEFUL_SAMPLES) in note for note in notes)


def test_an_empty_dataset_says_to_keep_using_newal():
    notes = TrainingPlan().warnings(sample_count=0)
    assert any("0개" in note for note in notes)


def test_an_unconventional_alpha_is_mentioned_not_blocked():
    plan = TrainingPlan(lora_r=16, lora_alpha=16)
    assert plan.validate() == []
    assert any("alpha" in note for note in plan.warnings(1000))


def test_full_fine_tuning_warns_about_forgetting():
    notes = TrainingPlan(method="full").warnings(1000)
    assert any("full" in note for note in notes)


def test_a_healthy_plan_with_enough_data_is_quiet():
    plan = TrainingPlan(lora_r=16, lora_alpha=32, batch_size=2, grad_accum=4)
    assert plan.warnings(5000) == []


def test_effective_batch_is_the_product():
    assert TrainingPlan(batch_size=3, grad_accum=8).effective_batch == 24


# ---- round trip --------------------------------------------------------------


def test_plan_survives_a_dict_round_trip():
    original = TrainingPlan(task="dpo", lora_r=32, beta=0.2, base_model="Qwen/Qwen3.5-9B")
    assert TrainingPlan.from_dict(original.to_dict()) == original


def test_unknown_keys_from_the_browser_are_ignored():
    plan = TrainingPlan.from_dict({"task": "dpo", "effective_batch": 99, "nonsense": 1})
    assert plan.task == "dpo"


# ---- the generated script ----------------------------------------------------


def _compiles(source: str) -> bool:
    ast.parse(source)
    return True


def test_sft_script_is_valid_python():
    assert _compiles(render_script(TrainingPlan(), "newal-sft.jsonl"))


def test_dpo_script_is_valid_python():
    assert _compiles(render_script(TrainingPlan(task="dpo"), "newal-dpo.jsonl"))


def test_full_fine_tune_script_is_valid_python():
    assert _compiles(render_script(TrainingPlan(method="full"), "newal-sft.jsonl"))


def test_sft_script_uses_the_sft_trainer():
    script = render_script(TrainingPlan(), "d.jsonl")
    assert "SFTTrainer" in script
    assert "DPOTrainer" not in script


def test_dpo_script_uses_the_dpo_trainer_and_beta():
    script = render_script(TrainingPlan(task="dpo", beta=0.25), "d.jsonl")
    assert "DPOTrainer" in script
    assert "beta=0.25" in script


def test_qlora_script_configures_4bit():
    script = render_script(TrainingPlan(method="qlora"), "d.jsonl")
    assert "load_in_4bit=True" in script
    assert "nf4" in script


def test_full_fine_tune_skips_quantisation_and_peft():
    script = render_script(TrainingPlan(method="full"), "d.jsonl")
    assert "bnb_config = None" in script
    assert "peft_config = None" in script


def test_hyperparameters_reach_the_script():
    plan = TrainingPlan(lora_r=64, lora_alpha=128, learning_rate=1e-4, epochs=5, grad_accum=16)
    script = render_script(plan, "d.jsonl")
    for expected in ("r=64", "lora_alpha=128", "learning_rate=0.0001",
                     "num_train_epochs=5", "gradient_accumulation_steps=16"):
        assert expected in script


def test_the_dataset_path_is_embedded():
    script = render_script(TrainingPlan(), "/tmp/my-data.jsonl")
    assert "/tmp/my-data.jsonl" in script


def test_the_script_says_where_the_labels_came_from():
    """Someone reading it later should not have to guess how it was labelled."""
    script = render_script(TrainingPlan(), "d.jsonl")
    assert "test suite" in script
    assert "No human annotator" in script
