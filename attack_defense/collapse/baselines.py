from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .adapters import build_adapter
from .registry import build_experiment_spec
from .specs import MODEL_SPECS


BASELINE_CACHE_ROOT = Path("results/baselines")


def _baseline_cache_path(model_key: str, dataset_key: str, seed: int) -> Path:
    return BASELINE_CACHE_ROOT / model_key / dataset_key / f"seed{seed}.json"


def check_whitebox_blackbox(model_key: str, dataset_key: str, seed: int = 42) -> dict[str, float]:
    experiment = build_experiment_spec(model_key, dataset_key, seed=seed)
    adapter = build_adapter(MODEL_SPECS[model_key], experiment)
    return adapter.baseline_report()


def compute_or_load_baselines(
    model_key: str,
    dataset_key: str,
    seed: int = 42,
    *,
    force: bool = False,
) -> Dict[str, float]:
    """Return cached (whitebox_acc, blackbox_acc) for (model, dataset, seed).

    Always uses a defense-agnostic adapter (no DEFENSE_EXPERIMENT_OVERRIDES).
    Cache is written to results/baselines/<model>/<dataset>/seed<N>.json.
    """
    cache_path = _baseline_cache_path(model_key, dataset_key, seed)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    from .datasets import DATASET_SPECS

    experiment = build_experiment_spec(model_key, dataset_key, seed=seed, defense_key=None)
    adapter = build_adapter(MODEL_SPECS[model_key], experiment)
    dataset_spec = DATASET_SPECS[dataset_key]
    effective_train_samples = experiment.train_samples or dataset_spec.train_samples
    effective_eval_samples = experiment.eval_samples or dataset_spec.eval_samples
    effective_batch_size = experiment.batch_size or dataset_spec.batch_size

    victim_model = adapter.load_victim_model()
    whitebox_acc = adapter.evaluate(victim_model)

    blackbox_model = copy.deepcopy(adapter.load_public_model())
    adapter.run_blackbox_finetune(blackbox_model)
    blackbox_acc = adapter.evaluate(blackbox_model)

    payload = {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "seed": seed,
        "whitebox_acc": whitebox_acc,
        "blackbox_acc": blackbox_acc,
        "train_samples": effective_train_samples,
        "eval_samples": effective_eval_samples,
        "batch_size": effective_batch_size,
        "blackbox_epochs": experiment.blackbox_epochs,
        "blackbox_lr": experiment.blackbox_lr,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def train_blackbox_model(model_key: str, dataset_key: str, seed: int = 42) -> Tuple[Dict[str, float], Any]:
    """Return (baselines_payload, trained_blackbox_model) using the default (defense-agnostic) adapter.

    Always trains fresh (no caching of model weights). The returned model is on the
    adapter's device. Callers should move it to CPU between uses to free GPU memory.
    Accuracy is recorded / compared against the on-disk baseline cache; if the cache
    already exists and matches, the cached blackbox_acc is used.
    """
    experiment = build_experiment_spec(model_key, dataset_key, seed=seed, defense_key=None)
    adapter = build_adapter(MODEL_SPECS[model_key], experiment)

    victim_model = adapter.load_victim_model()
    whitebox_acc = adapter.evaluate(victim_model)
    del victim_model

    blackbox_model = copy.deepcopy(adapter.load_public_model())
    adapter.run_blackbox_finetune(blackbox_model)
    blackbox_acc = adapter.evaluate(blackbox_model)

    from .datasets import DATASET_SPECS

    dataset_spec = DATASET_SPECS[dataset_key]
    payload = {
        "model_key": model_key,
        "dataset_key": dataset_key,
        "seed": seed,
        "whitebox_acc": whitebox_acc,
        "blackbox_acc": blackbox_acc,
        "train_samples": experiment.train_samples or dataset_spec.train_samples,
        "eval_samples": experiment.eval_samples or dataset_spec.eval_samples,
        "batch_size": experiment.batch_size or dataset_spec.batch_size,
        "blackbox_epochs": experiment.blackbox_epochs,
        "blackbox_lr": experiment.blackbox_lr,
    }
    cache_path = _baseline_cache_path(model_key, dataset_key, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, blackbox_model
