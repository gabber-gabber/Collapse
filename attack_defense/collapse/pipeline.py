from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

import torch

from .attacks import attack_tensor_maps, reset_arrowcloak_lengths, reset_oprior_lengths
from .defenses import build_oprior_static_sparse_templates, obfuscate_tensor_map
from .io import make_run_dir, write_result_bundle
from .registry import build_defense_spec
from .specs import DEFENSE_SPECS
from .types import DefenseSpec, ExperimentResult, ExperimentSpec, FineTuneMode, TaskAdapter


def _view_seed(base_seed: int, seed_slot: int, view_idx: int) -> int:
    return base_seed + 1000 + seed_slot * 97 + view_idx


def _collect_weights(adapter: TaskAdapter, model, weight_scope: str):
    if weight_scope == "linear":
        return adapter.collect_linear_weights(model)
    return adapter.collect_target_weights(model)


def _set_weights(adapter: TaskAdapter, model, weights, weight_scope: str) -> None:
    if weight_scope == "linear":
        adapter.set_linear_weights(model, weights)
        return
    adapter.set_target_weights(model, weights)


def run_weight_only_experiment(
    adapter: TaskAdapter,
    defense_key: str,
    seed: int,
    defense_spec_override: DefenseSpec | None = None,
    baselines: Dict[str, float] | None = None,
    blackbox_model_override: object | None = None,
    attack_key: str | None = None,
) -> ExperimentResult:
    defense_spec = defense_spec_override or DEFENSE_SPECS[defense_key]
    attack_key = attack_key or defense_key
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    victim_model = adapter.load_victim_model()
    public_prior = adapter.load_public_model()
    timings["load_models_sec"] = time.perf_counter() - t0

    # Prior source is determined by the ATTACK (not the defense): ArrowCloak /
    # Oprior attacks benefit from a blackbox fine-tuned prior; others use public.
    attack_spec = DEFENSE_SPECS.get(attack_key, defense_spec)
    effective_prior_source = attack_spec.attack_prior_source or defense_spec.attack_prior_source

    t0 = time.perf_counter()
    if baselines is not None:
        whitebox_acc = float(baselines["whitebox_acc"])
        blackbox_acc = float(baselines["blackbox_acc"])
        if effective_prior_source == "blackbox":
            if blackbox_model_override is None:
                raise ValueError(
                    f"Attack '{attack_key}' (defense '{defense_key}') requires a trained blackbox model as prior. "
                    f"Pass blackbox_model_override when baselines is supplied."
                )
            blackbox_model = blackbox_model_override.to(adapter.device)
        else:
            blackbox_model = public_prior
    else:
        blackbox_model = copy.deepcopy(public_prior)
        whitebox_acc = adapter.evaluate(victim_model)
        adapter.run_blackbox_finetune(blackbox_model)
        blackbox_acc = adapter.evaluate(blackbox_model)
    timings["baseline_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    victim_weights = _collect_weights(adapter, victim_model, defense_spec.weight_scope)
    prior_model = blackbox_model if effective_prior_source == "blackbox" else public_prior
    prior_weights = _collect_weights(adapter, prior_model, defense_spec.weight_scope)
    timings["collect_weights_sec"] = time.perf_counter() - t0

    observed_views: List[Dict[str, torch.Tensor]] = []
    observed_metadata: List[Dict[str, Dict[str, torch.Tensor]]] = []
    static_sparse_templates = None
    if defense_key == "oprior":
        static_sparse_templates = build_oprior_static_sparse_templates(
            victim_weights,
            seed=seed + defense_spec.static_sparse_seed_offset,
            sparse_ratio=defense_spec.sparse_ratio or 0.01,
            sparse_strength=defense_spec.sparse_strength or 0.6,
        )
    t0 = time.perf_counter()
    for view_idx in range(defense_spec.observations):
        artifacts = obfuscate_tensor_map(
            victim_weights,
            seed=_view_seed(seed, defense_spec.seed_slot, view_idx),
            defense_key=defense_key,
            spec=adapter.spec,
            use_all_layers=defense_spec.weight_scope == "linear",
            static_sparse_templates=static_sparse_templates,
            mask_rank=defense_spec.mask_rank,
        )
        observed_views.append({layer_name: item.obfuscated for layer_name, item in artifacts.items()})
        observed_metadata.append({layer_name: item.metadata for layer_name, item in artifacts.items() if item.metadata})
    timings["obfuscation_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    obfuscated_model = copy.deepcopy(victim_model)
    _set_weights(adapter, obfuscated_model, observed_views[0], defense_spec.weight_scope)
    obfuscated_acc = adapter.evaluate(obfuscated_model)
    timings["obfuscated_eval_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    attack_stats: Dict[str, Any] = {}
    # Attacks that exploit per-view metadata (perm) only have ground truth when
    # attack matches defense. For cross-attack, suppress metadata so the attack
    # falls back to its metadata-free path.
    attack_metadata = observed_metadata if attack_key == defense_key else None
    recovered = attack_tensor_maps(
        attack_key,
        observed_views,
        prior_weights,
        adapter.spec,
        use_all_layers=defense_spec.weight_scope == "linear",
        observed_metadata=attack_metadata,
        mask_rank=defense_spec.mask_rank,
        attack_stats=attack_stats,
    )
    if attack_key == "oprior":
        recovered = reset_oprior_lengths(recovered, prior_weights)
    elif attack_key == "arrowcloak":
        recovered = reset_arrowcloak_lengths(recovered, prior_weights)
    timings["matrix_attack_sec"] = time.perf_counter() - t0
    if attack_stats:
        for key, value in attack_stats.items():
            if isinstance(value, (int, float)):
                timings[key] = round(float(value), 6)

    t0 = time.perf_counter()
    attacked_model = copy.deepcopy(obfuscated_model)
    _set_weights(adapter, attacked_model, {name: item.recovered for name, item in recovered.items()}, defense_spec.weight_scope)
    sparse_values = {name: item.sparse_value for name, item in recovered.items() if item.sparse_value is not None}
    if sparse_values:
        current_linear = adapter.collect_linear_weights(attacked_model)
        for name, add in sparse_values.items():
            if name in current_linear and tuple(add.shape) == tuple(current_linear[name].shape):
                current_linear[name] = current_linear[name] + add
        adapter.set_linear_weights(attacked_model, current_linear)
    recovered_init_acc = adapter.evaluate(attacked_model)
    sparse_masks = {name: item.sparse_mask for name, item in recovered.items() if item.sparse_mask is not None}
    direction_by_layer = None
    if attack_spec.fine_tune_mode in {FineTuneMode.DIRECTION_ONLY, FineTuneMode.LENGTH_AND_MASKED}:
        direction_by_layer = {
            name: item.recovered / torch.clamp(
                torch.linalg.norm(item.recovered.double(), dim=0).to(dtype=item.recovered.dtype, device=item.recovered.device),
                min=1e-8,
            )
            for name, item in recovered.items()
        }
    adapter.run_attack_finetune(
        attacked_model,
        attack_spec.fine_tune_mode,
        sparse_masks=sparse_masks,
        direction_by_layer=direction_by_layer,
        loader_kind=attack_spec.attack_loader,
        lr=attack_spec.attack_lr,
        epochs=attack_spec.attack_epochs,
    )
    finetuned_attack_acc = adapter.evaluate(attacked_model)
    attack_acc = max(recovered_init_acc, finetuned_attack_acc)
    timings["attack_finetune_eval_sec"] = time.perf_counter() - t0

    return ExperimentResult(
        whitebox_acc=whitebox_acc,
        blackbox_acc=blackbox_acc,
        obfuscated_acc=obfuscated_acc,
        attack_acc=attack_acc,
        recovered_layers=len(recovered),
        extra={
            "defense": defense_key,
            "attack": attack_key,
            "mask_rank": defense_spec.mask_rank,
            "fine_tune_mode": attack_spec.fine_tune_mode.value,
            "recovered_init_acc": recovered_init_acc,
            "finetuned_attack_acc": finetuned_attack_acc,
            "attack_selection": "finetuned" if finetuned_attack_acc >= recovered_init_acc else "recovered_init",
            "timings_sec": {k: round(v, 6) for k, v in timings.items()},
            "attack_diagnostics": {k: v for k, v in attack_stats.items() if not isinstance(v, (int, float))},
        },
    )


def run_and_record_experiment(
    adapter: TaskAdapter,
    experiment: ExperimentSpec,
    defense_key: str,
    baselines: Dict[str, float] | None = None,
    blackbox_model: object | None = None,
) -> ExperimentResult:
    defense_spec = build_defense_spec(adapter.spec.key, experiment.dataset_key, defense_key)
    result = run_weight_only_experiment(
        adapter,
        defense_key=defense_key,
        seed=experiment.seed,
        defense_spec_override=defense_spec,
        baselines=baselines,
        blackbox_model_override=blackbox_model,
    )
    run_dir = make_run_dir(experiment, defense_key)
    write_result_bundle(run_dir, experiment, defense_key, result)
    return result
