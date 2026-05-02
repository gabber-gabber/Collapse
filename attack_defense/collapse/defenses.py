from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .types import MatrixArtifacts, ModelSpec, TensorMap


def _is_weight_only_target_layer_name(layer_name: str) -> bool:
    base = layer_name.removesuffix(".weight")
    return base not in {"classifier", "bert.pooler.dense", "score"}


def _cpu_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def obfuscate_loro(weight: torch.Tensor, seed: int, rank: int = 3, alpha: float = 0.3) -> MatrixArtifacts:
    gen = _cpu_generator(seed)
    rows, cols = weight.shape
    left = torch.randn(rows, rank, generator=gen, dtype=weight.dtype)
    right = torch.randn(rank, cols, generator=gen, dtype=weight.dtype)
    mask = alpha * left @ right
    mask = mask * (weight.std() / (mask.std() + 1e-9))
    return MatrixArtifacts(obfuscated=weight + mask, metadata={"mask": mask})


def obfuscate_tsqp(weight: torch.Tensor, seed: int) -> MatrixArtifacts:
    gen = _cpu_generator(seed)
    scale = 1.0 + 5.0 * torch.rand(1, generator=gen).item()
    return MatrixArtifacts(obfuscated=weight * scale, metadata={"scale": torch.tensor(scale)})


def obfuscate_nnsplitter(weight: torch.Tensor, seed: int, ratio: float = 0.002) -> MatrixArtifacts:
    gen = _cpu_generator(seed)
    flat = weight.reshape(-1)
    center = torch.median(flat)
    diff = torch.abs(flat - center)
    k = max(1, min(flat.numel(), int(flat.numel() * ratio)))
    topk_vals, _ = torch.topk(diff, k=k, largest=False)
    eps = topk_vals[-1]
    mask = torch.abs(flat - center) <= eps
    w_min = float(torch.min(flat).item())
    w_max = float(torch.max(flat).item())
    noise = (w_max - w_min) * torch.rand(flat.numel(), generator=gen) + w_min
    obf = torch.where(mask, noise.to(dtype=flat.dtype), flat).reshape_as(weight)
    return MatrixArtifacts(obfuscated=obf, metadata={"sparse_mask": mask.reshape_as(weight)})


def obfuscate_arrowcloak(weight: torch.Tensor, seed: int) -> MatrixArtifacts:
    gen = _cpu_generator(seed)
    cols = weight.shape[1]
    coeff = torch.randint(0, 5, (cols,), generator=gen, dtype=torch.int64).to(dtype=weight.dtype)
    v = torch.matmul(weight, coeff)
    v = v * (weight.std() / (v.std() + 1e-9))
    q = torch.randint(0, 11, (cols,), generator=gen, dtype=torch.int64).to(dtype=weight.dtype) - 5.0
    p = torch.randint(1, 3, (cols,), generator=gen, dtype=torch.int64).to(dtype=weight.dtype)
    rank1_add = v.unsqueeze(1) * q.unsqueeze(0)
    mixed = (weight * p.unsqueeze(0)) + rank1_add
    perm = torch.randperm(cols, generator=gen)
    return MatrixArtifacts(
        obfuscated=mixed[:, perm],
        metadata={"perm": perm, "p": p, "q": q, "rank1": rank1_add},
    )


def _sample_oprior_low_rank(weight: torch.Tensor, gen: torch.Generator, rank: int) -> torch.Tensor:
    cols = weight.shape[1]
    coeff = torch.randint(0, 5, (cols, rank), generator=gen, dtype=torch.int64).to(dtype=weight.dtype)
    basis = torch.matmul(weight, coeff)
    basis = basis / torch.clamp(torch.linalg.norm(basis.double(), dim=0).to(dtype=weight.dtype), min=1e-8).unsqueeze(0)
    q = torch.randint(0, 11, (rank, cols), generator=gen, dtype=torch.int64).to(dtype=weight.dtype) - 5.0
    low_rank = basis @ q
    return low_rank * (weight.std() / (low_rank.std() + 1e-9))


def obfuscate_oprior(weight: torch.Tensor, seed: int, sparse_ratio: float = 0.002, sparse_strength: float = 0.5, rank: int = 1) -> MatrixArtifacts:
    gen = _cpu_generator(seed)
    cols = weight.shape[1]
    low_rank = _sample_oprior_low_rank(weight, gen, rank=rank)
    p = torch.randint(1, 3, (cols,), generator=gen, dtype=torch.int64).to(dtype=weight.dtype)
    mixed = (weight * p.unsqueeze(0)) + low_rank
    gen = _cpu_generator(seed + 17)
    noise = torch.randn(weight.shape, generator=gen, dtype=weight.dtype)
    sparse_mask = torch.rand(weight.shape, generator=gen) < sparse_ratio
    sparse = torch.where(sparse_mask, noise * (weight.std() * sparse_strength), torch.zeros_like(weight))
    perm_gen = _cpu_generator(seed)
    perm = torch.randperm(cols, generator=perm_gen)
    obf = (mixed + sparse)[:, perm]
    metadata = {"perm": perm, "p": p, "low_rank": low_rank}
    metadata["sparse_mask"] = sparse_mask
    metadata["sparse_value"] = sparse
    return MatrixArtifacts(obfuscated=obf, metadata=metadata)


def build_oprior_static_sparse_templates(
    weights: TensorMap,
    seed: int,
    sparse_ratio: float,
    sparse_strength: float,
) -> Dict[str, Dict[str, torch.Tensor]]:
    templates: Dict[str, Dict[str, torch.Tensor]] = {}
    for idx, (layer_name, weight) in enumerate(weights.items(), 1):
        if not _is_weight_only_target_layer_name(layer_name):
            continue
        gen = _cpu_generator(seed + 7919 * idx)
        w = weight.detach().cpu().float()
        mask = torch.rand(w.shape, generator=gen) < sparse_ratio
        noise = torch.randn(w.shape, generator=gen, dtype=w.dtype)
        sparse = torch.where(mask, noise * (w.std() * sparse_strength), torch.zeros_like(w))
        templates[layer_name] = {"sparse_mask": mask.cpu(), "sparse": sparse.cpu()}
    return templates


def obfuscate_oprior_with_static_sparse(
    weights: TensorMap,
    seed: int,
    templates: Dict[str, Dict[str, torch.Tensor]],
    rank: int = 1,
) -> Dict[str, MatrixArtifacts]:
    out: Dict[str, MatrixArtifacts] = {}
    idx = 0
    for layer_name, weight in weights.items():
        if not _is_weight_only_target_layer_name(layer_name):
            continue
        idx += 1
        layer_seed = seed + 1009 * idx
        gen = _cpu_generator(layer_seed)
        cols = weight.shape[1]
        low_rank = _sample_oprior_low_rank(weight, gen, rank=rank)
        p = torch.randint(1, 3, (cols,), generator=gen, dtype=torch.int64).to(dtype=weight.dtype)
        sparse = templates[layer_name]["sparse"].to(dtype=weight.dtype)
        sparse_mask = templates[layer_name]["sparse_mask"]
        mixed = (weight * p.unsqueeze(0)) + low_rank + sparse
        perm = torch.randperm(cols, generator=gen)
        out[layer_name] = MatrixArtifacts(
            obfuscated=mixed[:, perm],
            metadata={
                "perm": perm,
                "p": p,
                "low_rank": low_rank,
                "sparse_mask": sparse_mask,
                "sparse": sparse,
            },
        )
    return out


def _block_sparse_invertible(size: int, gen: torch.Generator, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a block-sparse invertible (size x size) matrix with 2x2 blocks of det=1.

    Block B = [[1, a], [b, 1 + a*b]],  inverse = [[1 + a*b, -a], [-b, 1]]
    a, b drawn from {-2, -1, 0, 1, 2} (exact in fp16/bf16). Vectorized over
    all blocks at once — avoids a Python loop that would dominate obfuscation
    time for large matrices (e.g., Qwen-large 8960-dim intermediate).
    """
    n_blocks = size // 2
    M = torch.eye(size, dtype=dtype)
    M_inv = torch.eye(size, dtype=dtype)
    if n_blocks == 0:
        return M, M_inv
    alpha = torch.randint(-2, 3, (n_blocks,), generator=gen, dtype=torch.int64).to(dtype)
    beta = torch.randint(-2, 3, (n_blocks,), generator=gen, dtype=torch.int64).to(dtype)
    ab = alpha * beta
    idx = torch.arange(0, 2 * n_blocks, 2, dtype=torch.long)
    # M[i,i]=1 (eye), M[i,i+1]=a, M[i+1,i]=b, M[i+1,i+1]=1+ab
    M[idx, idx + 1] = alpha
    M[idx + 1, idx] = beta
    M[idx + 1, idx + 1] = 1 + ab
    # M_inv[i,i]=1+ab, M_inv[i,i+1]=-a, M_inv[i+1,i]=-b, M_inv[i+1,i+1]=1 (eye)
    M_inv[idx, idx] = 1 + ab
    M_inv[idx, idx + 1] = -alpha
    M_inv[idx + 1, idx] = -beta
    return M, M_inv


def obfuscate_ourdefense(weight: torch.Tensor, seed: int, sparse_ratio: float = 0.01, sparse_strength: float = 0.6) -> MatrixArtifacts:
    """ColMix: W̃ = P_left · (W + L_w + S_w) · P_right.

    P_left and P_right are independent block-sparse invertible matrices
    (block size 2, det = 1), refreshed per round. This breaks BOTH the row
    basis (via P_left) and the column basis (via P_right), so direction-
    matching attacks targeting column-wise exposure of W cannot recover
    per-column directions from any single view.

    Heavy matmuls (P_left @ core @ P_right) are offloaded to GPU when
    available — with Qwen-large's 8960-dim intermediate the CPU cost is
    ~hours per cell.
    """
    gen = _cpu_generator(seed)
    rows, cols = weight.shape
    cpu_dtype = weight.dtype

    # L_w: rank-1 low-rank additive mask
    coeff = torch.randint(0, 5, (cols,), generator=gen, dtype=torch.int64).to(dtype=cpu_dtype)
    v = torch.matmul(weight, coeff)
    v = v * (weight.std() / (v.std() + 1e-9))
    q = torch.randint(-5, 6, (cols,), generator=gen, dtype=torch.int64).to(dtype=cpu_dtype)
    rank1_add = v.unsqueeze(1) * q.unsqueeze(0)

    # S_w: sparse additive mask
    sparse_mask = torch.rand(weight.shape, generator=gen) < sparse_ratio
    noise = torch.randn(weight.shape, generator=gen, dtype=cpu_dtype)
    sparse = torch.where(sparse_mask, noise * (weight.std() * sparse_strength), torch.zeros_like(weight))

    # Block-sparse invertible mixers acting on the row and column bases.
    gen_left = _cpu_generator(seed + 131)
    P_left, _ = _block_sparse_invertible(rows, gen_left, dtype=cpu_dtype)
    gen_right = _cpu_generator(seed + 191)
    P_right, _ = _block_sparse_invertible(cols, gen_right, dtype=cpu_dtype)

    core = weight + rank1_add + sparse
    if torch.cuda.is_available():
        # Upcast to fp32 for stable BLAS on GPU; cast back to original dtype.
        device = torch.device("cuda")
        core_gpu = core.to(device=device, dtype=torch.float32)
        Pl_gpu = P_left.to(device=device, dtype=torch.float32)
        Pr_gpu = P_right.to(device=device, dtype=torch.float32)
        obf_gpu = Pl_gpu @ core_gpu @ Pr_gpu
        obf = obf_gpu.to(device="cpu", dtype=cpu_dtype)
        del core_gpu, Pl_gpu, Pr_gpu, obf_gpu
    else:
        obf = P_left @ core @ P_right
    return MatrixArtifacts(
        obfuscated=obf,
        metadata={
            "sparse_mask": sparse_mask,
            "sparse_value": sparse,
        },
    )


def _translinkguard_prefix_and_role(layer_name: str, spec: ModelSpec) -> tuple[Optional[str], Optional[str]]:
    if spec.family in {"bert", "vit"}:
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
    if spec.family == "qwen":
        weight_name = layer_name.removesuffix(".weight")
        parts = weight_name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
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


def obfuscate_translinkguard(weights: TensorMap, seed: int, spec: ModelSpec) -> Dict[str, MatrixArtifacts]:
    out: Dict[str, MatrixArtifacts] = {}
    layer_perms: Dict[str, torch.Tensor] = {}
    idx = 0

    for layer_name, weight in weights.items():
        if not layer_name.endswith(spec.target_suffixes):
            continue

        idx += 1
        prefix, role = _translinkguard_prefix_and_role(layer_name, spec)
        if prefix is None or role is None:
            out[layer_name] = MatrixArtifacts(obfuscated=weight.clone())
            continue

        gen = _cpu_generator(seed + 1009 * idx)
        if role == "q":
            perm = torch.randperm(weight.shape[1], generator=gen)
            layer_perms[prefix] = perm
            out[layer_name] = MatrixArtifacts(obfuscated=weight[:, perm], metadata={"perm": perm})
            continue

        perm = layer_perms.get(prefix)
        if perm is None:
            perm = torch.arange(weight.shape[1])
            if role == "attn_out":
                perm = torch.arange(weight.shape[0])

        if role in {"k", "v", "ffn_in", "ffn_gate", "ffn_up"} and perm.numel() == weight.shape[1]:
            out[layer_name] = MatrixArtifacts(obfuscated=weight[:, perm], metadata={"perm": perm})
        elif role == "attn_out" and perm.numel() == weight.shape[0]:
            inv = torch.argsort(perm)
            out[layer_name] = MatrixArtifacts(obfuscated=weight[inv, :], metadata={"perm": perm, "inv_perm": inv})
        elif role == "ffn_out_row" and perm.numel() == weight.shape[0]:
            inv = torch.argsort(perm)
            out[layer_name] = MatrixArtifacts(obfuscated=weight[inv, :], metadata={"perm": perm, "inv_perm": inv})
        else:
            out[layer_name] = MatrixArtifacts(obfuscated=weight.clone())

    return out


def obfuscate_tensor_map(
    weights: TensorMap,
    seed: int,
    defense_key: str,
    spec: ModelSpec,
    use_all_layers: bool = False,
    static_sparse_templates: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    mask_rank: int = 1,
) -> Dict[str, MatrixArtifacts]:
    if defense_key == "translinkguard":
        return obfuscate_translinkguard(weights, seed=seed, spec=spec)
    if defense_key == "oprior" and static_sparse_templates is not None:
        return obfuscate_oprior_with_static_sparse(weights, seed=seed, templates=static_sparse_templates, rank=mask_rank)

    out: Dict[str, MatrixArtifacts] = {}
    idx = 0
    for layer_name, weight in weights.items():
        if defense_key == "arrowcloak" and use_all_layers:
            if not _is_weight_only_target_layer_name(layer_name):
                continue
        if not use_all_layers and not layer_name.endswith(spec.target_suffixes):
            continue
        idx += 1
        layer_seed = seed + 1009 * idx
        if defense_key == "loro":
            out[layer_name] = obfuscate_loro(weight, layer_seed)
        elif defense_key == "tsqp":
            out[layer_name] = obfuscate_tsqp(weight, layer_seed)
        elif defense_key == "nnsplitter":
            out[layer_name] = obfuscate_nnsplitter(weight, layer_seed)
        elif defense_key == "arrowcloak":
            out[layer_name] = obfuscate_arrowcloak(weight, layer_seed)
        elif defense_key == "oprior":
            out[layer_name] = obfuscate_oprior(weight, layer_seed, rank=mask_rank)
        elif defense_key == "ourdefense":
            out[layer_name] = obfuscate_ourdefense(weight, layer_seed)
        else:
            raise KeyError(f"Unsupported defense: {defense_key}")
    return out


DEFENSES = {
    "loro": obfuscate_loro,
    "tsqp": obfuscate_tsqp,
    "nnsplitter": obfuscate_nnsplitter,
    "translinkguard": obfuscate_translinkguard,
    "arrowcloak": obfuscate_arrowcloak,
    "oprior": obfuscate_oprior,
    "ourdefense": obfuscate_ourdefense,
}
