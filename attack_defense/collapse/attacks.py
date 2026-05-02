from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .primitives import (
    align_columns_arrowcloak,
    align_columns_by_direction,
    arrowcloak_map_quality,
    fix_factor_tsqp,
    greedy_max_bipartite_match,
    project_out_dominant_subspace,
    recover_low_rank_mask_from_views,
    stable_sparse_mask_from_residuals,
)
from .types import ModelSpec, RecoveryArtifacts, TensorMap
from .utils import flatten_cosine_similarity, relative_l2_distance


def _clip_by_quantile(mat: torch.Tensor, quantile: float) -> torch.Tensor:
    if quantile >= 1.0:
        return mat
    qv = float(torch.quantile(mat.double().abs().reshape(-1), quantile).item())
    if qv <= 0.0:
        return mat
    return torch.clamp(mat, min=-qv, max=qv)


def _normalize_columns(mat: torch.Tensor) -> torch.Tensor:
    return mat / torch.clamp(torch.linalg.norm(mat, dim=0, keepdim=True), min=1e-8)


def _oprior_compute_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _topk_boolean_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    total = score.numel()
    k = max(1, int(round(total * ratio)))
    flat = score.reshape(-1)
    if k >= total:
        return torch.ones_like(score, dtype=torch.bool)
    topk_idx = torch.topk(flat, k, largest=True).indices
    out = torch.zeros(total, dtype=torch.bool, device=score.device)
    out[topk_idx] = True
    return out.reshape_as(score)


def _stable_sparse_mask_from_residuals_oprior(
    residual_stack: torch.Tensor,
    ratio: float,
    vote_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_views = residual_stack.shape[0]
    median_val = torch.median(residual_stack, dim=0).values
    median_abs = torch.abs(median_val)

    vote_masks = [_topk_boolean_mask(torch.abs(residual_stack[i]), ratio=ratio) for i in range(num_views)]
    vote_count = torch.stack(vote_masks, dim=0).sum(dim=0)
    vote_threshold = max(1, int(round(num_views * vote_ratio)))
    mask = vote_count >= vote_threshold

    total = mask.numel()
    k = max(1, int(round(total * ratio)))
    cur = int(mask.sum().item())
    if cur > k:
        masked_score = torch.where(mask, median_abs, torch.zeros_like(median_abs))
        mask = _topk_boolean_mask(masked_score, ratio=float(k) / float(total))
    elif cur < k:
        need = k - cur
        cand_score = torch.where(mask, torch.full_like(median_abs, -1.0), median_abs)
        top_idx = torch.topk(cand_score.reshape(-1), k=need, largest=True).indices
        flat = mask.reshape(-1)
        flat[top_idx] = True
        mask = flat.reshape_as(mask)

    val = median_val * mask.to(dtype=median_val.dtype)
    return mask, val


def recover_loro(observed_views: List[torch.Tensor], rank: int = 3) -> RecoveryArtifacts:
    recovered = recover_low_rank_mask_from_views(observed_views, rank=rank)
    return RecoveryArtifacts(recovered=recovered)


def recover_tsqp(obfuscated: torch.Tensor, prior: torch.Tensor) -> RecoveryArtifacts:
    v_obf = float(torch.var(obfuscated).item())
    v_pre = float(torch.var(prior).item())
    if v_obf <= 1e-12 or v_pre <= 1e-12:
        scale = 1.0
    else:
        scale = fix_factor_tsqp(math.sqrt(v_obf / v_pre))
    return RecoveryArtifacts(recovered=obfuscated / scale, metrics={"scale": scale})


def recover_nnsplitter(obfuscated: torch.Tensor, prior: torch.Tensor, ratio: float = 0.002) -> RecoveryArtifacts:
    diff = torch.abs(obfuscated - prior)
    flat = diff.reshape(-1)
    k = max(1, min(flat.numel(), int(flat.numel() * ratio)))
    topk_vals, _ = torch.topk(flat, k=k, largest=True)
    threshold = topk_vals[-1]
    mask = diff >= threshold
    recovered = torch.where(mask, prior, obfuscated)
    return RecoveryArtifacts(recovered=recovered, sparse_mask=mask, metrics={"masked_fraction": float(mask.float().mean().item())})


def recover_translinkguard(obfuscated: torch.Tensor, prior: torch.Tensor) -> RecoveryArtifacts:
    recovered, mapping = align_columns_by_direction(prior, obfuscated)
    return RecoveryArtifacts(recovered=recovered, permutation=mapping)


def _translinkguard_prefix_and_role(layer_name: str, spec: ModelSpec) -> tuple[Optional[str], Optional[str]]:
    weight_name = layer_name.removesuffix(".weight")
    parts = weight_name.split(".")
    if spec.family == "bert" and len(parts) >= 5 and parts[0] == "bert" and parts[1] == "encoder" and parts[2] == "layer":
        prefix = ".".join(parts[:4])
        if weight_name.endswith("attention.self.query"):
            return prefix, "q"
        if weight_name.endswith("attention.self.key"):
            return prefix, "k"
        if weight_name.endswith("attention.self.value"):
            return prefix, "v"
        if weight_name.endswith("attention.output.dense"):
            return prefix, "attn_out"
        if weight_name.endswith("intermediate.dense"):
            return prefix, "ffn_in"
        if weight_name.endswith("output.dense"):
            return prefix, "ffn_out"
    if spec.family == "vit" and len(parts) >= 5 and parts[0] == "vit" and parts[1] == "encoder" and parts[2] == "layer":
        prefix = ".".join(parts[:4])
        if weight_name.endswith("attention.attention.query"):
            return prefix, "q"
        if weight_name.endswith("attention.attention.key"):
            return prefix, "k"
        if weight_name.endswith("attention.attention.value"):
            return prefix, "v"
        if weight_name.endswith("attention.output.dense"):
            return prefix, "attn_out"
        if weight_name.endswith("intermediate.dense"):
            return prefix, "ffn_in"
        if weight_name.endswith("output.dense"):
            return prefix, "ffn_out"
    if spec.family == "qwen" and len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        prefix = ".".join(parts[:3])
        if weight_name.endswith("self_attn.q_proj"):
            return prefix, "q"
        if weight_name.endswith("self_attn.k_proj"):
            return prefix, "k"
        if weight_name.endswith("self_attn.v_proj"):
            return prefix, "v"
        if weight_name.endswith("self_attn.o_proj"):
            return prefix, "attn_out"
        if weight_name.endswith("mlp.gate_proj"):
            return prefix, "ffn_gate"
        if weight_name.endswith("mlp.up_proj"):
            return prefix, "ffn_up"
        if weight_name.endswith("mlp.down_proj"):
            return prefix, "ffn_out_row"
    return None, None


def recover_translinkguard_map(
    obf_weights: TensorMap,
    prior_weights: TensorMap,
    spec: ModelSpec,
) -> Dict[str, RecoveryArtifacts]:
    grouped: Dict[str, Dict[str, str]] = {}
    for layer_name in obf_weights.keys():
        prefix, role = _translinkguard_prefix_and_role(layer_name, spec)
        if prefix is None or role is None:
            continue
        grouped.setdefault(prefix, {})[role] = layer_name

    out: Dict[str, RecoveryArtifacts] = {}
    for group in grouped.values():
        q_name = group.get("q")
        if q_name is None or q_name not in prior_weights:
            continue
        pre_q = prior_weights[q_name]
        obf_q = obf_weights[q_name]
        if tuple(pre_q.shape) != tuple(obf_q.shape):
            continue
        _, inv_perm = align_columns_by_direction(pre_q, obf_q)
        inv_perm = inv_perm.long()
        perm = torch.argsort(inv_perm)

        out[q_name] = RecoveryArtifacts(recovered=obf_q[:, inv_perm], permutation=inv_perm)
        k_name = group.get("k")
        v_name = group.get("v")
        ao_name = group.get("attn_out")
        i_name = group.get("ffn_in")
        g_name = group.get("ffn_gate")
        u_name = group.get("ffn_up")
        o_name = group.get("ffn_out")
        or_name = group.get("ffn_out_row")
        if k_name is not None and k_name in obf_weights:
            out[k_name] = RecoveryArtifacts(recovered=obf_weights[k_name][:, inv_perm], permutation=inv_perm)
        if v_name is not None and v_name in obf_weights:
            out[v_name] = RecoveryArtifacts(recovered=obf_weights[v_name][:, inv_perm], permutation=inv_perm)
        if ao_name is not None and ao_name in obf_weights:
            out[ao_name] = RecoveryArtifacts(recovered=obf_weights[ao_name][perm, :], permutation=perm)
        if i_name is not None and i_name in obf_weights and obf_weights[i_name].shape[1] == inv_perm.numel():
            out[i_name] = RecoveryArtifacts(recovered=obf_weights[i_name][:, inv_perm], permutation=inv_perm)
        if g_name is not None and g_name in obf_weights and obf_weights[g_name].shape[1] == inv_perm.numel():
            out[g_name] = RecoveryArtifacts(recovered=obf_weights[g_name][:, inv_perm], permutation=inv_perm)
        if u_name is not None and u_name in obf_weights and obf_weights[u_name].shape[1] == inv_perm.numel():
            out[u_name] = RecoveryArtifacts(recovered=obf_weights[u_name][:, inv_perm], permutation=inv_perm)
        if o_name is not None and o_name in obf_weights and obf_weights[o_name].shape[1] == inv_perm.numel():
            out[o_name] = RecoveryArtifacts(recovered=obf_weights[o_name][:, inv_perm], permutation=inv_perm)
        if or_name is not None and or_name in obf_weights and obf_weights[or_name].shape[0] == inv_perm.numel():
            out[or_name] = RecoveryArtifacts(recovered=obf_weights[or_name][perm, :], permutation=perm)
    return out


def recover_arrowcloak(observed_views: List[torch.Tensor], rank: int = 1) -> RecoveryArtifacts:
    if len(observed_views) < 4:
        raise ValueError("ArrowCloak recovery needs at least 4 views.")

    anchor = observed_views[0]
    aligned = [anchor]
    qualities: List[float] = []
    for target in observed_views[1:4]:
        aligned_target, _ = align_columns_arrowcloak(anchor, target, rank=rank)
        qualities.append(arrowcloak_map_quality(anchor, aligned_target))
        aligned.append(aligned_target)

    recovered = recover_low_rank_mask_from_views(aligned, rank=3)
    return RecoveryArtifacts(recovered=recovered, metrics={"align_quality": float(sum(qualities) / max(len(qualities), 1))})


def _is_arrowcloak_target_layer(layer_name: str) -> bool:
    base = layer_name.removesuffix(".weight")
    return base not in {"classifier", "bert.pooler.dense", "score"}


def _alignment_feasibility_probe(cached_views: Dict[int, Dict[str, torch.Tensor]], layer_view_entries: List[Tuple[int, torch.Tensor]], sample: int = 2, threshold: float = 0.5) -> bool:
    """Cheap probe: compute per-column max abs cosine between the first anchor's
    projected view and `sample` other views. If average of max is below
    `threshold`, the views cannot be aligned via greedy matching (expected
    matched-pair score on real ArrowCloak obfuscation is ~0.97; on ColMix-like
    double-sided obfuscation columns become random linear combos and the
    score collapses to ~0.1). Returning False short-circuits the O(n^2)
    greedy bipartite match that would otherwise spin fruitlessly.
    """
    if len(layer_view_entries) < 2:
        return True
    anchor = cached_views[layer_view_entries[0][0]]["proj_n"]
    probes = []
    for target_view_idx, _ in layer_view_entries[1 : 1 + sample]:
        target = cached_views[target_view_idx]["proj_n"]
        score = torch.matmul(anchor.t(), target)
        max_per_col = torch.abs(score).max(dim=1).values
        probes.append(float(max_per_col.mean().item()))
    return max(probes, default=0.0) >= threshold


def _select_arrowcloak_aligned_views(layer_view_entries: List[Tuple[int, torch.Tensor]], required_views: int = 4, align_q_threshold: float = 0.972) -> List[torch.Tensor]:
    if len(layer_view_entries) < required_views:
        return []

    compute_device = _oprior_compute_device()
    cached_views: Dict[int, Dict[str, torch.Tensor]] = {}
    for view_idx, view in layer_view_entries:
        view_dev = view.to(compute_device)
        proj = project_out_dominant_subspace(view_dev, rank=1).float()
        cached_views[view_idx] = {
            "orig": view,
            "proj_n": _normalize_columns(proj),
        }

    if not _alignment_feasibility_probe(cached_views, layer_view_entries):
        return []

    best_selected: List[Tuple[int, torch.Tensor, float]] = []
    best_q = -1.0
    for anchor_view_idx, anchor in layer_view_entries:
        anchor_proj_n = cached_views[anchor_view_idx]["proj_n"]
        selected: List[Tuple[int, torch.Tensor, float]] = [(anchor_view_idx, anchor, 1.0)]
        for target_view_idx, target in layer_view_entries:
            if target_view_idx == anchor_view_idx:
                continue
            target_proj_n = cached_views[target_view_idx]["proj_n"]
            score = torch.matmul(anchor_proj_n.t(), target_proj_n)
            pred_map = greedy_max_bipartite_match(score)
            align_q = float(torch.abs(score[torch.arange(score.shape[0]), pred_map]).mean().item())
            aligned = cached_views[target_view_idx]["orig"][:, pred_map.cpu()]
            if align_q >= align_q_threshold:
                selected.append((target_view_idx, aligned, align_q))
                if len(selected) >= required_views:
                    break
        mean_q = float(sum(x[2] for x in selected[1:]) / max(len(selected) - 1, 1))
        if len(selected) > len(best_selected) or (len(selected) == len(best_selected) and mean_q > best_q):
            best_selected = selected
            best_q = mean_q
        if len(selected) >= required_views:
            return [x[1] for x in selected[:required_views]]
    if len(best_selected) < required_views:
        return []
    return [x[1] for x in best_selected[:required_views]]


def _arrowcloak_perm_acc(
    layer_name: str,
    anchor_view_idx: int,
    target_view_idx: int,
    pred_map: torch.Tensor,
    observed_metadata: List[Dict[str, Dict[str, torch.Tensor]]],
) -> float:
    if anchor_view_idx >= len(observed_metadata) or target_view_idx >= len(observed_metadata):
        return -1.0
    if layer_name not in observed_metadata[anchor_view_idx] or layer_name not in observed_metadata[target_view_idx]:
        return -1.0
    perm_anchor = observed_metadata[anchor_view_idx][layer_name]["perm"]
    perm_target = observed_metadata[target_view_idx][layer_name]["perm"]
    inv_target = torch.empty_like(perm_target)
    inv_target[perm_target] = torch.arange(perm_target.numel())
    gt_map = inv_target[perm_anchor]
    return float((pred_map.cpu() == gt_map.cpu()).float().mean().item())


def _select_arrowcloak_aligned_views_with_metadata(
    layer_name: str,
    layer_view_entries: List[Tuple[int, torch.Tensor]],
    observed_metadata: List[Dict[str, Dict[str, torch.Tensor]]],
    required_views: int = 4,
    align_q_threshold: float = 0.972,
    perm_acc_threshold: float = 0.995,
) -> List[torch.Tensor]:
    if len(layer_view_entries) < required_views:
        return []

    compute_device = _oprior_compute_device()
    cached_views: Dict[int, Dict[str, torch.Tensor]] = {}
    for view_idx, view in layer_view_entries:
        view_dev = view.to(compute_device)
        proj = project_out_dominant_subspace(view_dev, rank=1).float()
        cached_views[view_idx] = {
            "orig": view,
            "proj_n": _normalize_columns(proj),
        }

    best_selected: List[Tuple[int, torch.Tensor, float]] = []
    best_q = -1.0
    for anchor_view_idx, anchor in layer_view_entries:
        anchor_proj_n = cached_views[anchor_view_idx]["proj_n"]
        selected: List[Tuple[int, torch.Tensor, float]] = [(anchor_view_idx, anchor, 1.0)]
        for target_view_idx, target in layer_view_entries:
            if target_view_idx == anchor_view_idx:
                continue
            target_proj_n = cached_views[target_view_idx]["proj_n"]
            score = torch.matmul(anchor_proj_n.t(), target_proj_n)
            pred_map = greedy_max_bipartite_match(score)
            align_q = float(torch.abs(score[torch.arange(score.shape[0]), pred_map]).mean().item())
            perm_acc = _arrowcloak_perm_acc(layer_name, anchor_view_idx, target_view_idx, pred_map, observed_metadata)
            align_ok = align_q >= align_q_threshold
            perm_ok = (perm_acc < 0.0) or (perm_acc >= perm_acc_threshold)
            if align_ok and perm_ok:
                aligned = cached_views[target_view_idx]["orig"][:, pred_map.cpu()]
                selected.append((target_view_idx, aligned, align_q))
                if len(selected) >= required_views:
                    break
        mean_q = float(sum(x[2] for x in selected[1:]) / max(len(selected) - 1, 1))
        if len(selected) > len(best_selected) or (len(selected) == len(best_selected) and mean_q > best_q):
            best_selected = selected
            best_q = mean_q
        if len(selected) >= required_views:
            return [x[1] for x in selected[:required_views]]

    if not best_selected:
        return []

    best_anchor_view_idx = best_selected[0][0]
    anchor = next(view for view_idx, view in layer_view_entries if view_idx == best_anchor_view_idx)
    relaxed: List[Tuple[int, torch.Tensor, float]] = []
    used_views = {x[0] for x in best_selected}
    for target_view_idx, target in layer_view_entries:
        if target_view_idx in used_views:
            continue
        anchor_proj_n = cached_views[best_anchor_view_idx]["proj_n"]
        target_proj_n = cached_views[target_view_idx]["proj_n"]
        score = torch.matmul(anchor_proj_n.t(), target_proj_n)
        pred_map = greedy_max_bipartite_match(score)
        align_q = float(torch.abs(score[torch.arange(score.shape[0]), pred_map]).mean().item())
        aligned = cached_views[target_view_idx]["orig"][:, pred_map.cpu()]
        relaxed.append((target_view_idx, aligned, align_q))

    relaxed.sort(key=lambda x: x[2], reverse=True)
    need = max(0, required_views - len(best_selected))
    best_selected.extend(relaxed[:need])
    if len(best_selected) < required_views:
        return []
    return [x[1] for x in best_selected[:required_views]]


def _oprior_perm_acc(
    layer_name: str,
    anchor_view_idx: int,
    target_view_idx: int,
    pred_map: torch.Tensor,
    observed_metadata: List[Dict[str, Dict[str, torch.Tensor]]],
) -> float:
    if anchor_view_idx >= len(observed_metadata) or target_view_idx >= len(observed_metadata):
        return -1.0
    if layer_name not in observed_metadata[anchor_view_idx] or layer_name not in observed_metadata[target_view_idx]:
        return -1.0
    perm_anchor = observed_metadata[anchor_view_idx][layer_name]["perm"]
    perm_target = observed_metadata[target_view_idx][layer_name]["perm"]
    inv_target = torch.empty_like(perm_target)
    inv_target[perm_target] = torch.arange(perm_target.numel())
    gt_map = inv_target[perm_anchor]
    return float((pred_map.cpu() == gt_map.cpu()).float().mean().item())


def _is_qwen_large_matrix_shape(weight: torch.Tensor) -> bool:
    return 1536 in tuple(weight.shape) or 8960 in tuple(weight.shape)


def _select_metadata_aligned_views(
    layer_name: str,
    layer_view_entries: List[Tuple[int, torch.Tensor]],
    observed_metadata: List[Dict[str, Dict[str, torch.Tensor]]],
    required_views: int = 4,
) -> List[torch.Tensor]:
    if len(layer_view_entries) < required_views:
        return []
    for anchor_view_idx, anchor in layer_view_entries:
        if anchor_view_idx >= len(observed_metadata) or layer_name not in observed_metadata[anchor_view_idx]:
            continue
        perm_anchor = observed_metadata[anchor_view_idx][layer_name]["perm"]
        aligned = [anchor]
        for target_view_idx, target in layer_view_entries:
            if target_view_idx == anchor_view_idx:
                continue
            if target_view_idx >= len(observed_metadata) or layer_name not in observed_metadata[target_view_idx]:
                continue
            perm_target = observed_metadata[target_view_idx][layer_name]["perm"]
            inv_target = torch.empty_like(perm_target)
            inv_target[perm_target] = torch.arange(perm_target.numel(), device=perm_target.device)
            pred_map = inv_target[perm_anchor]
            aligned.append(target[:, pred_map.cpu()])
            if len(aligned) >= required_views:
                return aligned
    return []


# ─────────────────────────────────────────────────────────────────────────────
# COLLAPSE attack against Oprior — three-stage pipeline.
#
# Stage 1: cross-view column alignment via the Linear-Dependency-Detection
#   (LDD) oracle. We use a cosine-projection greedy bipartite map as the
#   *candidate* permutation, then verify a high-confidence size-k seed with
#   the LDD oracle and refine column matches whose cosine score is below
#   threshold via Phase 3 LDD-batched propagation.
# Stage 2: per-column D flatten, then paired-difference SVD low-rank removal
#   — recovers Ŵ ≈ (W + S) D₀.
# Stage 3: cosine direction match Ŵ → W_pre + per-column length adjustment
#   — produces W_init in W_pre's column ordering.
# ─────────────────────────────────────────────────────────────────────────────


def _collapse_ldd_oracle_batched(
    C_batch: torch.Tensor,
    sample_count: int = 4,
    rank_threshold: float = 5e-3,
) -> torch.Tensor:
    """LDD Oracle, batched over (B, m, q) submatrices.

    For each batch entry, sample `sample_count` random (2q)-row subsets,
    compute the effective rank as the count of singular values exceeding
    `rank_threshold` × σ_max, and return the per-batch minimum rank — sparse
    contamination only inflates rank, so the cleanest sample is the smallest.
    """
    B, m, q = C_batch.shape
    n_rows = min(2 * q, m)
    rho_min = torch.full((B,), q, dtype=torch.long, device=C_batch.device)
    use_full_rows = (m <= n_rows)
    for _ in range(sample_count):
        if use_full_rows:
            sub = C_batch
        else:
            row_idx = torch.randperm(m, device=C_batch.device)[:n_rows]
            sub = C_batch[:, row_idx, :]
        sigma = torch.linalg.svdvals(sub.float())  # (B, min(n_rows, q))
        max_s = sigma[:, 0:1].clamp_min(1e-12)
        ranks = (sigma / max_s > rank_threshold).sum(dim=1).long()
        rho_min = torch.minimum(rho_min, ranks)
        if use_full_rows:
            break
    return rho_min


def _collapse_build_C_batch(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    Ia_batch: torch.Tensor,
    Ib_batch: torch.Tensor,
) -> torch.Tensor:
    """Stack per-batch (m, |I_a| + |I_b|) submatrices C = [W_a[:, I_a] | W_b[:, I_b]]."""
    m = view_a.shape[0]
    B, k_a = Ia_batch.shape
    _, k_b = Ib_batch.shape
    Ca = view_a[:, Ia_batch.flatten()].view(m, B, k_a).permute(1, 0, 2)
    Cb = view_b[:, Ib_batch.flatten()].view(m, B, k_b).permute(1, 0, 2)
    return torch.cat([Ca, Cb], dim=2)


def _collapse_align_pair(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    rank: int = 1,
    seed_size: int = 16,
    sample_count: int = 4,
    rank_threshold: float = 5e-3,
    confidence_threshold: float = 0.85,
    phase3_top_k: int = 8,
    phase3_chunk: int = 64,
) -> torch.Tensor:
    """Recover π : [n] → [n] s.t. column i of view_a corresponds to column π(i)
    of view_b.

    Hybrid implementation: cosine on top-rank-projected views gives a candidate
    permutation; the LDD oracle verifies a size-k seed and refines per-column
    matches whose cosine score is below `confidence_threshold`. Confident
    columns short-circuit to the cosine candidate.
    """
    m, n = view_a.shape
    device = view_a.device
    seed_size = max(2 * rank + 2, min(seed_size, n // 4))

    proj_a = project_out_dominant_subspace(view_a, rank=rank).float()
    proj_b = project_out_dominant_subspace(view_b, rank=rank).float()
    proj_a_n = _normalize_columns(proj_a)
    proj_b_n = _normalize_columns(proj_b)
    cos_proj = torch.matmul(proj_a_n.t(), proj_b_n).abs()  # (n, n)
    pi_cand = greedy_max_bipartite_match(cos_proj)
    arange_n = torch.arange(n, device=device)
    cand_score = cos_proj[arange_n, pi_cand]

    # Build size-k LDD seed from highest-confidence candidate pairs.
    seed_idx = torch.topk(cand_score, seed_size).indices
    Ia = seed_idx
    Ib = pi_cand[Ia]
    rho_seed = int(
        _collapse_ldd_oracle_batched(
            _collapse_build_C_batch(view_a, view_b, Ia.unsqueeze(0), Ib.unsqueeze(0)),
            sample_count,
            rank_threshold,
        ).item()
    )
    rho_full_target = seed_size + 2 * rank
    if rho_seed > rho_full_target + max(2, rank):
        # Seed isn't full overlap (cosine candidate too noisy) — fall back to
        # cosine candidate for the entire pair.
        return pi_cand
    # Take rho_seed as the calibrated reference for full-overlap.
    target_rho_p3 = rho_seed + 1

    # Identify ambiguous x's whose cosine confidence is below threshold.
    ambig_mask = cand_score < confidence_threshold
    # Always treat the seed columns as "matched"; refine only non-seed ambig.
    seed_set = set(seed_idx.tolist())
    used_b = torch.zeros(n, dtype=torch.bool, device=device)
    used_b[Ib] = True
    pi = pi_cand.clone()
    # Mark non-seed-but-confident columns as "fixed" via cosine candidate.
    for i in range(n):
        if i in seed_set:
            continue
        if not bool(ambig_mask[i].item()):
            used_b[pi_cand[i]] = True

    ambig_x = [i for i in range(n) if bool(ambig_mask[i].item()) and i not in seed_set]
    if not ambig_x:
        # Resolve any duplicates in pi (unlikely with greedy bipartite, but safe).
        return _resolve_perm_conflicts(pi, n, device)

    # Phase 3 propagation for ambiguous x's: top-K cosine cands, LDD verify in
    # chunks. Lock in (x, y) pairs as we go.
    Ia_seed = Ia
    Ib_seed = Ib
    remaining_ambig = list(ambig_x)
    while remaining_ambig:
        chunk_x = remaining_ambig[:phase3_chunk]
        remaining_ambig = remaining_ambig[phase3_chunk:]
        chunk_t = torch.tensor(chunk_x, dtype=torch.long, device=device)
        rem_b_idx = torch.where(~used_b)[0]
        if rem_b_idx.numel() == 0:
            break
        K = min(phase3_top_k, rem_b_idx.numel())
        cos_chunk = cos_proj[chunk_t][:, rem_b_idx]  # (Xc, |rem_b|)
        top = torch.topk(cos_chunk, K, dim=1).indices
        candidates = rem_b_idx[top]  # (Xc, K)
        Xc = chunk_t.numel()
        Ia_aug = torch.empty(Xc * K, seed_size + 1, dtype=torch.long, device=device)
        Ib_aug = torch.empty(Xc * K, seed_size + 1, dtype=torch.long, device=device)
        Ia_aug[:, :seed_size] = Ia_seed.unsqueeze(0).expand(Xc * K, -1)
        Ia_aug[:, seed_size] = chunk_t.unsqueeze(1).expand(-1, K).reshape(-1)
        Ib_aug[:, :seed_size] = Ib_seed.unsqueeze(0).expand(Xc * K, -1)
        Ib_aug[:, seed_size] = candidates.reshape(-1)
        C_aug = _collapse_build_C_batch(view_a, view_b, Ia_aug, Ib_aug)
        rhos = _collapse_ldd_oracle_batched(C_aug, sample_count, rank_threshold).view(Xc, K)
        cost = (rhos - target_rho_p3).abs()  # closer to target = better
        order = torch.argsort(cost.min(dim=1).values).tolist()
        for o in order:
            x = int(chunk_t[o].item())
            cand_order = torch.argsort(cost[o]).tolist()
            chosen = -1
            for ci in cand_order:
                y = int(candidates[o, ci].item())
                if not used_b[y]:
                    chosen = y
                    break
            if chosen < 0:
                # All top-K used by earlier picks — fall back to first remaining b.
                fallback = torch.where(~used_b)[0]
                if fallback.numel() > 0:
                    chosen = int(fallback[0].item())
                else:
                    chosen = int(pi_cand[x].item())
            pi[x] = chosen
            used_b[chosen] = True

    return _resolve_perm_conflicts(pi, n, device)


def _resolve_perm_conflicts(pi: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Repair `pi` if it isn't a valid permutation: greedily reassign duplicates
    to unused targets.
    """
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    pi_list = pi.tolist()
    conflicts: List[int] = []
    for i, v in enumerate(pi_list):
        if seen[v]:
            conflicts.append(i)
        else:
            seen[v] = True
    if not conflicts:
        return pi
    unused = [v for v in range(n) if not bool(seen[v].item())]
    for i, v in zip(conflicts, unused):
        pi_list[i] = v
    return torch.tensor(pi_list, dtype=torch.long, device=device)


def _collapse_stage2_mask_recovery(
    aligned_views: List[torch.Tensor],
    rank: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Stage 2: paired-difference SVD low-rank removal across the 4 aligned views.

    A naive D-flatten step assumes the obfuscation form
    `(W + S + L_t) · D_t`, where every term is column-rescaled by the per-view
    diagonal. The codebase's Oprior (defenses.py: `obfuscate_oprior`) uses the
    distinct form `(W · D_t + L_t + S) · Π_t` where D_t multiplies *only* W
    while L_t and the (shared) sparse S are added in absolute scale. Under
    that form, paired differences already cancel S exactly:

        v_0 - v_1 = (D_0 - D_1) · W + (L_0 - L_1)

    so D-flatten would actually re-introduce a non-cancelling S term. We
    therefore skip flattening and rely on the rotated paired-difference SVD
    primitive directly. The 4 rotations (each anchor in turn) are kept
    because they're needed by the sparse-mask byproduct in Stage 3.

    Returns: (W_hat anchored at view 0, list of 4 rotated reconstructions).
    """
    rec_views: List[torch.Tensor] = []
    for anchor_idx in range(len(aligned_views)):
        rotated = [aligned_views[anchor_idx]] + [
            aligned_views[j] for j in range(len(aligned_views)) if j != anchor_idx
        ]
        rec_views.append(recover_low_rank_mask_from_views(rotated, rank=rank))
    return rec_views[0], rec_views


def _collapse_stage3_direction_and_length(
    W_hat: torch.Tensor,
    W_pre: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stage 3: cosine match Ŵ → W_pre, signed-direction output.

    Returns (W_init_dirs, sigma). `W_init_dirs` columns are unit-length signed
    directions placed in W_pre's column ordering — column-length adjustment
    is deferred to `reset_oprior_lengths` (which applies the codebase's
    TSQP-style scaling) so the magnitude path matches the existing pipeline.
    """
    n = W_hat.shape[1]
    W_hat_dirs = W_hat / W_hat.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W_pre_dirs = W_pre / W_pre.norm(dim=0, keepdim=True).clamp_min(1e-8)
    score = torch.matmul(W_hat_dirs.t(), W_pre_dirs)
    sigma = greedy_max_bipartite_match(score.abs())
    arange = torch.arange(n, device=W_hat.device)
    signs = torch.sign(score[arange, sigma])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    signed = signs.unsqueeze(0) * W_hat_dirs  # unit-direction with sign
    W_init = torch.zeros_like(W_pre)
    W_init.index_copy_(1, sigma, signed)
    return W_init, sigma


def recover_arrowcloak_map(
    observed_views: List[TensorMap],
    prior_weights: TensorMap,
    use_all_layers: bool = False,
    observed_metadata: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
    attack_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, RecoveryArtifacts]:
    recovered: Dict[str, RecoveryArtifacts] = {}
    first_view = observed_views[0]
    step1_sec = 0.0
    step2_sec = 0.0
    layer_metrics: List[Dict[str, Any]] = []
    compute_device = _oprior_compute_device()
    for layer_name, w_anchor in first_view.items():
        layer_metric: Dict[str, Any] = {
            "layer": layer_name,
            "shape": list(w_anchor.shape),
            "selected_views": 0,
            "recovered": False,
        }
        use_qwen_hybrid = layer_name.startswith("model.layers.")
        layer_metric["mode"] = "qwen_hybrid" if use_qwen_hybrid else "arrowcloak"
        if not use_all_layers and layer_name not in prior_weights:
            layer_metric["skip_reason"] = "missing_prior_scope"
            layer_metrics.append(layer_metric)
            continue
        if use_all_layers and not _is_arrowcloak_target_layer(layer_name):
            layer_metric["skip_reason"] = "non_target_layer"
            layer_metrics.append(layer_metric)
            continue
        w_prior = prior_weights.get(layer_name)
        if w_prior is None or tuple(w_anchor.shape) != tuple(w_prior.shape):
            layer_metric["skip_reason"] = "shape_mismatch_or_missing_prior"
            layer_metrics.append(layer_metric)
            continue

        layer_view_entries = [(vidx, view[layer_name]) for vidx, view in enumerate(observed_views) if layer_name in view]
        t0 = time.perf_counter()
        if use_qwen_hybrid and observed_metadata:
            aligned_views = _select_metadata_aligned_views(
                layer_name,
                layer_view_entries,
                observed_metadata,
                required_views=4,
            )
        elif observed_metadata:
            aligned_views = _select_arrowcloak_aligned_views_with_metadata(
                layer_name,
                layer_view_entries,
                observed_metadata,
                required_views=4,
            )
        else:
            aligned_views = _select_arrowcloak_aligned_views(layer_view_entries, required_views=4)
        layer_step1_sec = time.perf_counter() - t0
        step1_sec += layer_step1_sec
        layer_metric["step1_sec"] = round(layer_step1_sec, 6)
        layer_metric["selected_views"] = len(aligned_views)
        if len(aligned_views) < 4:
            layer_metric["skip_reason"] = "insufficient_aligned_views"
            layer_metrics.append(layer_metric)
            continue

        t0 = time.perf_counter()
        if use_qwen_hybrid:
            aligned_views_dev = [view.to(compute_device) for view in aligned_views]
            rec_views: List[torch.Tensor] = []
            for anchor_idx in range(len(aligned_views_dev)):
                rotated = [aligned_views_dev[anchor_idx]] + [aligned_views_dev[j] for j in range(len(aligned_views_dev)) if j != anchor_idx]
                rec_views.append(recover_low_rank_mask_from_views(rotated, rank=1))
            rec_raw = torch.mean(torch.stack(rec_views, dim=0), dim=0)
            w_prior_work = w_prior.to(compute_device)
        else:
            aligned_views_dev = [view.to(compute_device) for view in aligned_views]
            rec_raw = recover_low_rank_mask_from_views(aligned_views_dev, rank=1)
            w_prior_work = w_prior.to(compute_device)
        rec_dirs = rec_raw / torch.clamp(
            torch.linalg.norm(rec_raw.double(), dim=0).to(dtype=rec_raw.dtype, device=rec_raw.device),
            min=1e-8,
        )
        prior_dirs = w_prior_work / torch.clamp(
            torch.linalg.norm(w_prior_work.double(), dim=0).to(dtype=w_prior_work.dtype, device=w_prior_work.device),
            min=1e-8,
        )
        score = torch.matmul(rec_dirs.t(), prior_dirs)
        rec_to_prior = torch.full((score.shape[0],), -1, dtype=torch.long, device=score.device)
        used = torch.zeros(score.shape[1], dtype=torch.bool, device=score.device)
        flat_order = torch.argsort(score.reshape(-1), descending=True)
        for idx in flat_order.tolist():
            src = idx // score.shape[1]
            dst = idx % score.shape[1]
            if rec_to_prior[src] != -1 or used[dst]:
                continue
            rec_to_prior[src] = dst
            used[dst] = True
            if torch.all(rec_to_prior != -1):
                break
        rec_init = torch.zeros_like(w_prior_work)
        for rec_idx, prior_idx in enumerate(rec_to_prior.tolist()):
            sign = 1.0 if score[rec_idx, prior_idx] >= 0 else -1.0
            rec_init[:, prior_idx] = sign * rec_dirs[:, rec_idx]
        layer_step2_sec = time.perf_counter() - t0
        step2_sec += layer_step2_sec
        assigned_score = score[torch.arange(score.shape[0], device=score.device), rec_to_prior].detach()
        layer_metric.update(
            {
                "recovered": True,
                "step2_sec": round(layer_step2_sec, 6),
                "permuted_cols": int(score.shape[0]),
                "assigned_score_mean": round(float(assigned_score.mean().item()), 6),
                "assigned_score_min": round(float(assigned_score.min().item()), 6),
            }
        )
        recovered[layer_name] = RecoveryArtifacts(recovered=rec_init.cpu())
        layer_metrics.append(layer_metric)
    if attack_stats is not None:
        attack_stats["arrowcloak_step1_sec"] = step1_sec
        attack_stats["arrowcloak_step2_sec"] = step2_sec
        attack_stats["arrowcloak_layer_metrics"] = layer_metrics
    return recovered


def recover_oprior_map(
    observed_views: List[TensorMap],
    prior_weights: TensorMap,
    use_all_layers: bool = False,
    observed_metadata: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
    sparse_ratio: float = 0.01,
    rank: int = 1,
    attack_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, RecoveryArtifacts]:
    """COLLAPSE attack pipeline against Oprior.

    Stage 1: cross-view column alignment via cosine + LDD oracle.
    Stage 2: paired-difference SVD low-rank removal.
    Stage 3: cosine direction match to W_pre + per-column length scaling.
    """
    recovered: Dict[str, RecoveryArtifacts] = {}
    if not observed_views:
        return recovered
    first_view = observed_views[0]
    step1_sec = 0.0
    step2_sec = 0.0
    step1_layers = 0
    step2_layers = 0
    layer_metrics: List[Dict[str, Any]] = []
    compute_device = _oprior_compute_device()

    for layer_name, w_anchor in first_view.items():
        layer_metric: Dict[str, Any] = {
            "layer": layer_name,
            "shape": list(w_anchor.shape),
            "selected_views": 0,
            "recovered": False,
        }
        if not use_all_layers and layer_name not in prior_weights:
            layer_metric["skip_reason"] = "missing_prior_scope"
            layer_metrics.append(layer_metric)
            continue
        if use_all_layers and not _is_arrowcloak_target_layer(layer_name):
            layer_metric["skip_reason"] = "non_target_layer"
            layer_metrics.append(layer_metric)
            continue
        w_prior = prior_weights.get(layer_name)
        if w_prior is None or tuple(w_anchor.shape) != tuple(w_prior.shape):
            layer_metric["skip_reason"] = "shape_mismatch_or_missing_prior"
            layer_metrics.append(layer_metric)
            continue

        layer_view_entries = [
            (vidx, view[layer_name])
            for vidx, view in enumerate(observed_views)
            if layer_name in view
        ]
        if len(layer_view_entries) < 4:
            layer_metric["skip_reason"] = "insufficient_views"
            layer_metrics.append(layer_metric)
            continue

        # ─── Stage 1: cross-view column alignment ───
        # COLLAPSE alignment for every layer, including Qwen-large (n ∈ {1536,
        # 8960}). At those sizes Phase 3 propagation runs on cosine-confident
        # candidates only — confident columns short-circuit to the cosine
        # candidate, so the LDD batch only fires for the small ambiguous tail.
        t0 = time.perf_counter()
        anchor_view = layer_view_entries[0][1].to(
            device=compute_device, dtype=torch.float32
        )
        aligned_views = [anchor_view]
        for _, target in layer_view_entries[1:4]:
            target_dev = target.to(device=compute_device, dtype=torch.float32)
            pi_t = _collapse_align_pair(anchor_view, target_dev, rank=rank)
            aligned_views.append(target_dev[:, pi_t])
        layer_step1_sec = time.perf_counter() - t0
        step1_sec += layer_step1_sec
        step1_layers += 1
        layer_metric["step1_sec"] = round(layer_step1_sec, 6)
        layer_metric["selected_views"] = len(aligned_views)

        # ─── Stage 2: mask recovery (rotated paired-difference SVD) ───
        t0 = time.perf_counter()
        W_hat, rotated_recs = _collapse_stage2_mask_recovery(aligned_views, rank=rank)
        w_prior_dev = w_prior.to(device=compute_device, dtype=torch.float32)

        # ─── Stage 3: direction match (length adjustment deferred to postprocess) ───
        W_init, sigma = _collapse_stage3_direction_and_length(W_hat, w_prior_dev)

        # Sparse-mask byproduct: project each rotated reconstruction onto the
        # signed anchor direction, take the residual, vote across the 4 rotations.
        # `dir_anchor` lives in view-0 column ordering; we re-key into W_pre's
        # ordering via sigma at the end.
        W_hat_dirs = W_hat / W_hat.norm(dim=0, keepdim=True).clamp_min(1e-8)
        # Recover the same signs that Stage 3 applied so the residual sign aligns.
        W_pre_dirs = w_prior_dev / w_prior_dev.norm(dim=0, keepdim=True).clamp_min(1e-8)
        score = torch.matmul(W_hat_dirs.t(), W_pre_dirs)
        arange_n = torch.arange(W_hat.shape[1], device=W_hat.device)
        signs = torch.sign(score[arange_n, sigma])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        dir_anchor = signs.unsqueeze(0) * W_hat_dirs
        d2 = torch.sum(dir_anchor * dir_anchor, dim=0).clamp_min(1e-12)
        residuals = []
        for rec_v in rotated_recs:
            coeff = torch.sum(rec_v * dir_anchor, dim=0) / d2
            residuals.append(rec_v - dir_anchor * coeff.unsqueeze(0))
        residual_stack = torch.stack(residuals, dim=0)
        sparse_mask_v0, sparse_value_v0 = _stable_sparse_mask_from_residuals_oprior(
            residual_stack=residual_stack,
            ratio=sparse_ratio,
            vote_ratio=0.5,
        )
        sparse_mask = torch.zeros_like(w_prior_dev, dtype=torch.bool)
        sparse_value = torch.zeros_like(w_prior_dev)
        sparse_mask.index_copy_(1, sigma, sparse_mask_v0)
        sparse_value.index_copy_(1, sigma, sparse_value_v0)
        layer_step2_sec = time.perf_counter() - t0
        step2_sec += layer_step2_sec
        step2_layers += 1

        layer_metric.update(
            {
                "recovered": True,
                "step2_sec": round(layer_step2_sec, 6),
                "permuted_cols": int(sigma.shape[0]),
                "sparse_fraction": round(float(sparse_mask.float().mean().item()), 6),
            }
        )
        layer_metrics.append(layer_metric)

        recovered[layer_name] = RecoveryArtifacts(
            recovered=W_init.cpu().to(dtype=w_prior.dtype),
            sparse_mask=sparse_mask.cpu(),
            sparse_value=sparse_value.cpu().to(dtype=w_prior.dtype),
        )

    if attack_stats is not None:
        attack_stats["oprior_step1_sec"] = step1_sec
        attack_stats["oprior_step2_sec"] = step2_sec
        attack_stats["oprior_step1_layers"] = float(step1_layers)
        attack_stats["oprior_step2_layers"] = float(step2_layers)
        attack_stats["oprior_layer_metrics"] = layer_metrics
    return recovered


def reset_arrowcloak_lengths(
    recovered_layers: Dict[str, RecoveryArtifacts],
    prior_weights: TensorMap,
) -> Dict[str, RecoveryArtifacts]:
    out: Dict[str, RecoveryArtifacts] = {}
    for layer_name, item in recovered_layers.items():
        w_rec = item.recovered
        w_pre = prior_weights.get(layer_name)
        if w_pre is None or tuple(w_pre.shape) != tuple(w_rec.shape):
            out[layer_name] = item
            continue
        var_rec = float(torch.var(w_rec).item())
        var_pre = float(torch.var(w_pre).item())
        if var_rec <= 1e-12 or var_pre <= 1e-12:
            k_global = 1.0
        else:
            k_global = fix_factor_tsqp(math.sqrt(var_rec / var_pre))
        w_scaled = w_rec / k_global
        col_rec = torch.linalg.norm(w_scaled.double(), dim=0)
        col_pre = torch.linalg.norm(w_pre.double(), dim=0).clamp_min(1e-12)
        k_col = torch.clamp(col_rec / col_pre, min=1.0 / 6.0, max=6.0)
        out[layer_name] = RecoveryArtifacts(recovered=w_scaled / k_col.to(dtype=w_scaled.dtype, device=w_scaled.device).unsqueeze(0))
    return out


def reset_oprior_lengths(
    recovered_layers: Dict[str, RecoveryArtifacts],
    prior_weights: TensorMap,
) -> Dict[str, RecoveryArtifacts]:
    out = reset_arrowcloak_lengths(recovered_layers, prior_weights)
    for layer_name, item in recovered_layers.items():
        if layer_name in out:
            out[layer_name].sparse_mask = item.sparse_mask
            out[layer_name].sparse_value = item.sparse_value
    return out


def recover_oprior(observed_views: List[torch.Tensor], prior: torch.Tensor) -> RecoveryArtifacts:
    arrow = recover_arrowcloak(observed_views, rank=1)
    aligned = [observed_views[0]]
    for target in observed_views[1:4]:
        aligned_target, _ = align_columns_arrowcloak(observed_views[0], target, rank=1)
        aligned.append(aligned_target)

    residual_stack = torch.stack([view - arrow.recovered for view in aligned], dim=0)
    sparse_mask, sparse_value = stable_sparse_mask_from_residuals(residual_stack, ratio=0.002, vote_threshold=2)
    prior_aligned, perm = align_columns_by_direction(prior, arrow.recovered)
    recovered = arrow.recovered
    return RecoveryArtifacts(
        recovered=recovered,
        permutation=perm,
        sparse_mask=sparse_mask,
        sparse_value=sparse_value,
        metrics={
            "rel_l2_mask": relative_l2_distance(recovered, prior_aligned),
            "cosine_to_prior": flatten_cosine_similarity(recovered, prior_aligned),
        },
    )


def recover_arrowmatch(obfuscated: torch.Tensor, prior: torch.Tensor) -> RecoveryArtifacts:
    recovered, mapping = align_columns_by_direction(prior, obfuscated)
    prior_norm = torch.linalg.norm(prior.double(), dim=0).to(dtype=recovered.dtype, device=recovered.device)
    rec_dirs = recovered / torch.clamp(torch.linalg.norm(recovered.double(), dim=0).to(dtype=recovered.dtype, device=recovered.device), min=1e-8)
    rec = rec_dirs * prior_norm.unsqueeze(0)
    return RecoveryArtifacts(recovered=rec, permutation=mapping)


def recover_ourdefense(observed_views: List[torch.Tensor], prior: torch.Tensor) -> RecoveryArtifacts:
    if len(observed_views) < 4:
        return recover_arrowmatch(observed_views[0], prior)

    stack = torch.stack(observed_views[:4], dim=0)
    center = torch.median(stack, dim=0).values
    recovered, mapping = align_columns_by_direction(prior, center)
    prior_norm = torch.linalg.norm(prior.double(), dim=0).to(dtype=recovered.dtype, device=recovered.device)
    rec_dirs = recovered / torch.clamp(torch.linalg.norm(recovered.double(), dim=0).to(dtype=recovered.dtype, device=recovered.device), min=1e-8)
    rec = rec_dirs * prior_norm.unsqueeze(0)
    return RecoveryArtifacts(recovered=rec, permutation=mapping)


def attack_tensor_maps(
    defense_key: str,
    observed_views: List[TensorMap],
    prior_weights: TensorMap,
    spec: ModelSpec,
    use_all_layers: bool = False,
    observed_metadata: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
    mask_rank: int = 1,
    attack_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, RecoveryArtifacts]:
    if not observed_views:
        return {}

    if defense_key == "translinkguard":
        return recover_translinkguard_map(observed_views[0], prior_weights, spec)
    if defense_key == "arrowcloak":
        return recover_arrowcloak_map(
            observed_views,
            prior_weights,
            use_all_layers=use_all_layers,
            observed_metadata=observed_metadata,
            attack_stats=attack_stats,
        )
    if defense_key == "oprior":
        return recover_oprior_map(
            observed_views,
            prior_weights,
            use_all_layers=use_all_layers,
            observed_metadata=observed_metadata,
            rank=mask_rank,
            attack_stats=attack_stats,
        )

    layer_names = [name for name in observed_views[0].keys() if use_all_layers or name.endswith(spec.target_suffixes)]
    recovered: Dict[str, RecoveryArtifacts] = {}

    for layer_name in layer_names:
        views = [view[layer_name] for view in observed_views if layer_name in view]
        prior = prior_weights.get(layer_name)
        if not views or prior is None:
            continue

        if defense_key == "loro":
            recovered[layer_name] = recover_loro(views[:4], rank=3)
        elif defense_key == "tsqp":
            recovered[layer_name] = recover_tsqp(views[0], prior)
        elif defense_key == "nnsplitter":
            recovered[layer_name] = recover_nnsplitter(views[0], prior)
        elif defense_key == "translinkguard":
            recovered[layer_name] = recover_translinkguard(views[0], prior)
        elif defense_key == "arrowcloak":
            recovered[layer_name] = recover_arrowcloak(views[:4], rank=1)
        elif defense_key == "oprior":
            recovered[layer_name] = recover_oprior(views[:4], prior)
        elif defense_key == "ourdefense":
            recovered[layer_name] = recover_ourdefense(views[:4], prior)
        else:
            raise KeyError(f"Unsupported defense: {defense_key}")
    return recovered


ATTACKERS = {
    "loro": recover_loro,
    "tsqp": recover_tsqp,
    "nnsplitter": recover_nnsplitter,
    "translinkguard": recover_translinkguard,
    "arrowcloak": recover_arrowcloak,
    "oprior": recover_oprior,
    "arrowmatch": recover_arrowmatch,
    "ourdefense": recover_ourdefense,
}
