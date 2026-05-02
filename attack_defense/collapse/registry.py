from __future__ import annotations

from dataclasses import replace

from .specs import DATASET_MODEL_DEFAULTS, DEFENSE_EXPERIMENT_OVERRIDES, DEFENSE_SPEC_OVERRIDES, DEFENSE_SPECS, MODEL_SPECS
from .types import DefenseSpec
from .types import ExperimentSpec


def build_experiment_spec(model_key: str, dataset_key: str, seed: int = 42, defense_key: str | None = None) -> ExperimentSpec:
    defaults = DATASET_MODEL_DEFAULTS[(model_key, dataset_key)]
    overrides = DEFENSE_EXPERIMENT_OVERRIDES.get((model_key, dataset_key, defense_key), {}) if defense_key else {}
    return ExperimentSpec(
        key=f"{model_key}_{dataset_key}",
        model_key=model_key,
        dataset_key=dataset_key,
        victim_checkpoint=defaults["victim_checkpoint"],
        public_checkpoint=defaults["public_checkpoint"],
        seed=seed,
        attack_views=defaults.get("attack_views", 4),
        blackbox_epochs=defaults.get("blackbox_epochs", 1),
        blackbox_lr=defaults.get("blackbox_lr", 2e-5),
        attack_epochs=defaults.get("attack_epochs", 1),
        attack_lr=defaults.get("attack_lr", 1e-5),
        recover_ratio=defaults.get("recover_ratio", 0.01),
        train_samples=overrides.get("train_samples"),
        eval_samples=overrides.get("eval_samples"),
        batch_size=overrides.get("batch_size"),
    )


def build_defense_spec(model_key: str, dataset_key: str, defense_key: str) -> DefenseSpec:
    spec = DEFENSE_SPECS[defense_key]
    overrides = DEFENSE_SPEC_OVERRIDES.get((model_key, dataset_key, defense_key))
    if not overrides:
        return spec
    return replace(spec, **overrides)
