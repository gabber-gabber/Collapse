from __future__ import annotations

from typing import Dict, Iterable

import torch


def clone_tensor_map(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in weights.items()}


def filter_tensor_map_by_suffixes(weights: Dict[str, torch.Tensor], suffixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    suffixes = tuple(suffixes)
    return {name: tensor for name, tensor in weights.items() if name.endswith(suffixes)}
def relative_l2_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    denom = max(float(torch.linalg.norm(b64).item()), eps)
    num = float(torch.linalg.norm(a64 - b64).item())
    return num / denom


def flatten_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    denom = float(torch.linalg.norm(a64).item()) * float(torch.linalg.norm(b64).item())
    if denom <= eps:
        return 0.0
    return float(torch.dot(a64, b64).item() / denom)
