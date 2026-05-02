from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def _leading_left_singular_vectors(mat: torch.Tensor, rank: int) -> torch.Tensor:
    target_rank = max(1, min(rank, min(mat.shape)))
    if mat.device.type == "cuda":
        q = min(min(mat.shape), max(target_rank + 2, 4))
        u, _, _ = torch.svd_lowrank(mat, q=q, niter=2)
        return u[:, :target_rank]
    u, _, _ = torch.linalg.svd(mat, full_matrices=False)
    return u[:, :target_rank]


def l2_normalize_columns(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.linalg.norm(weight, dim=0, keepdim=True).clamp_min(eps)
    return weight / denom


def greedy_max_bipartite_match(score: torch.Tensor) -> torch.Tensor:
    """Greedy max-cost bipartite matching: assign src->dst one at a time by
    globally-highest score, skipping conflicts.

    Fast-fails to identity mapping when the per-row max |score| averages
    below 0.1 — indicates columns share no common direction (ColMix-style
    double-sided obfuscation gives random-ish cosines ≈ 1/sqrt(n)). Without
    this probe the loop spins through O(n^2) flat_order entries for no
    useful assignment (minutes per call at n=4864).

    Main loop runs in numpy instead of torch to avoid per-iteration GPU sync
    from tensor indexing — ~50x faster for large n.
    """
    n = score.shape[0]
    device = score.device

    max_per_row = torch.abs(score).max(dim=1).values
    if float(max_per_row.mean().item()) < 0.1:
        return torch.arange(n, dtype=torch.long, device=device)

    flat_order = torch.argsort(score.reshape(-1), descending=True).cpu().numpy()
    src_np = (flat_order // n).astype(np.int64)
    dst_np = (flat_order % n).astype(np.int64)

    mapping_np = np.full(n, -1, dtype=np.int64)
    used_np = np.zeros(n, dtype=bool)
    count = 0
    for i in range(flat_order.size):
        s = int(src_np[i])
        d = int(dst_np[i])
        if mapping_np[s] != -1 or used_np[d]:
            continue
        mapping_np[s] = d
        used_np[d] = True
        count += 1
        if count == n:
            break
    if count < n:
        remaining_src = np.where(mapping_np == -1)[0]
        remaining_dst = np.where(~used_np)[0]
        k = min(len(remaining_src), len(remaining_dst))
        mapping_np[remaining_src[:k]] = remaining_dst[:k]
    return torch.from_numpy(mapping_np).to(device=device)


def align_columns_by_direction(anchor: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor_n = l2_normalize_columns(anchor)
    target_n = l2_normalize_columns(target)
    score = torch.matmul(anchor_n.t(), target_n)
    mapping = greedy_max_bipartite_match(score)
    return target[:, mapping], mapping


def project_out_dominant_subspace(mat: torch.Tensor, rank: int = 1) -> torch.Tensor:
    work_dtype = torch.float32 if mat.device.type == "cuda" else torch.float64
    m = mat.to(dtype=work_dtype)
    basis = _leading_left_singular_vectors(m, rank=rank)
    return m - basis @ (basis.t() @ m)


def align_columns_arrowcloak(anchor: torch.Tensor, target: torch.Tensor, rank: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor_p = project_out_dominant_subspace(anchor, rank=rank)
    target_p = project_out_dominant_subspace(target, rank=rank)
    _, mapping = align_columns_by_direction(anchor_p.float(), target_p.float())
    return target[:, mapping], mapping


def arrowcloak_map_quality(anchor: torch.Tensor, aligned_target: torch.Tensor) -> float:
    an = l2_normalize_columns(project_out_dominant_subspace(anchor, rank=1).float())
    tn = l2_normalize_columns(project_out_dominant_subspace(aligned_target, rank=1).float())
    return float(torch.abs(torch.sum(an * tn, dim=0)).mean().item())


def recover_low_rank_mask_from_views(observed_weights: List[torch.Tensor], rank: int) -> torch.Tensor:
    if len(observed_weights) < 4:
        raise ValueError("Low-rank recovery requires at least 4 observed views.")

    work_dtype = torch.float32 if observed_weights[0].device.type == "cuda" else torch.float64
    w0 = observed_weights[0].to(dtype=work_dtype)
    w1 = observed_weights[1].to(dtype=work_dtype)
    w2 = observed_weights[2].to(dtype=work_dtype)
    w3 = observed_weights[3].to(dtype=work_dtype)

    obfus_space1 = w0 - w1
    obfus_space2 = w2 - w3

    basis_dim = max(1, 2 * rank)
    b1 = _leading_left_singular_vectors(obfus_space1, rank=basis_dim)
    b2 = _leading_left_singular_vectors(obfus_space2, rank=basis_dim)

    a_mat = torch.cat([b1, -b2], dim=1)
    target = w0 - w2
    x_solution = torch.linalg.lstsq(a_mat, target).solution

    c1 = x_solution[: b1.shape[1], :]
    estimated_mask_0 = b1 @ c1
    recovered = w0 - estimated_mask_0
    return recovered.to(dtype=observed_weights[0].dtype, device=observed_weights[0].device)


def topk_boolean_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    total = score.numel()
    k = max(1, min(total, int(round(total * ratio))))
    flat = score.reshape(-1)
    values, indices = torch.topk(flat, k=k, largest=True)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    if values.numel() > 0:
        threshold = values[-1]
        mask |= flat >= threshold
    return mask.reshape_as(score)


def stable_sparse_mask_from_residuals(
    residual_stack: torch.Tensor,
    ratio: float = 0.002,
    vote_threshold: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vote_masks = [topk_boolean_mask(torch.abs(residual_stack[i]), ratio=ratio) for i in range(residual_stack.shape[0])]
    vote_count = torch.stack(vote_masks, dim=0).sum(dim=0)
    mask = vote_count >= vote_threshold

    median_val = torch.median(residual_stack, dim=0).values
    median_abs = torch.abs(median_val)
    total = mask.numel()
    target_k = max(1, min(total, int(round(total * ratio))))
    current_k = int(mask.sum().item())

    if current_k > target_k:
        masked_score = torch.where(mask, median_abs, torch.zeros_like(median_abs))
        mask = topk_boolean_mask(masked_score, ratio=float(target_k) / float(total))
    elif current_k < target_k:
        cand_score = torch.where(mask, torch.full_like(median_abs, -1.0), median_abs)
        extra = topk_boolean_mask(cand_score, ratio=float(target_k - current_k) / float(total))
        mask = mask | extra

    value = median_val * mask.to(dtype=median_val.dtype)
    return mask, value


def fix_factor_tsqp(scale: float, mini: float = 1.0 / 6.0, maxi: float = 6.0) -> float:
    if not np.isfinite(scale):
        return 1.0
    return float(min(max(scale, mini), maxi))
