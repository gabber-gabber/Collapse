from __future__ import annotations

from typing import Mapping, Optional

import torch

from .types import FineTuneMode


def project_length_only(current: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    cur_norm = torch.linalg.norm(current.double(), dim=0, keepdim=True).clamp_min(eps)
    tgt_norm = torch.linalg.norm(target.double(), dim=0, keepdim=True)
    return current * (tgt_norm / cur_norm).to(dtype=current.dtype, device=current.device)


def apply_finetune_constraint(
    original: torch.Tensor,
    updated: torch.Tensor,
    mode: FineTuneMode,
    sparse_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mode == FineTuneMode.FULL:
        return updated
    if mode == FineTuneMode.LENGTH_ONLY:
        return project_length_only(updated, original)
    if mode == FineTuneMode.MASKED_ONLY:
        if sparse_mask is None:
            return original
        return torch.where(sparse_mask.to(device=updated.device), updated, original)
    if mode == FineTuneMode.LENGTH_AND_MASKED:
        projected = project_length_only(updated, original)
        if sparse_mask is None:
            return projected
        return torch.where(sparse_mask.to(device=updated.device), updated, projected)
    if mode == FineTuneMode.DIRECTION_ONLY:
        return original
    raise KeyError(f"Unsupported fine-tune mode: {mode}")


def apply_constraints_to_tensor_map(
    original_weights: Mapping[str, torch.Tensor],
    updated_weights: Mapping[str, torch.Tensor],
    mode: FineTuneMode,
    sparse_masks: Optional[Mapping[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    constrained: dict[str, torch.Tensor] = {}
    for name, updated in updated_weights.items():
        original = original_weights.get(name)
        if original is None:
            constrained[name] = updated
            continue
        mask = None if sparse_masks is None else sparse_masks.get(name)
        constrained[name] = apply_finetune_constraint(original, updated, mode, sparse_mask=mask)
    return constrained
