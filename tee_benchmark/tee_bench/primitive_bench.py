#!/usr/bin/env python3
"""TEE-side benchmark for obfuscated GEMM with Occlum.

Expected topology:
- This script runs inside Occlum enclave (TEE side).
- A GPU RPC server runs outside enclave and executes GEMM on GPU.

Measured steps for each run:
1) TEE obfuscation
2) GPU encrypted (obfuscated) computation
3) TEE de-obfuscation
4) TEE<->GPU transfer (round-trip minus GPU kernel time)
5) GPU plaintext baseline (same matrix size, no obfuscation)
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import sparse as sp


LEN_FMT = "!Q"
LEN_BYTES = struct.calcsize(LEN_FMT)


def recvall(sock: socket.socket, n: int) -> bytes:
    chunks = []
    left = n
    while left > 0:
        chunk = sock.recv(left)
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    header = recvall(sock, LEN_BYTES)
    payload_len = struct.unpack(LEN_FMT, header)[0]
    payload = recvall(sock, payload_len)
    return pickle.loads(payload)


def send_msg(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=5)
    sock.sendall(struct.pack(LEN_FMT, len(payload)))
    sock.sendall(payload)


class RpcClient:
    def __init__(self, host: str, port: int, timeout_s: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def ping(self) -> None:
        send_msg(self.sock, {"cmd": "ping"})
        resp = recv_msg(self.sock)
        if not resp.get("ok"):
            raise RuntimeError(f"ping failed: {resp}")

    def matmul(self, a: np.ndarray, b: np.ndarray, dtype_name: str) -> dict[str, Any]:
        req = {
            "cmd": "matmul",
            "dtype": dtype_name,
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "a_bytes": a.tobytes(order="C"),
            "b_bytes": b.tobytes(order="C"),
        }
        t0 = time.perf_counter()
        send_msg(self.sock, req)
        resp = recv_msg(self.sock)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        if not resp.get("ok"):
            raise RuntimeError(f"server error: {resp.get('error')}")
        resp["roundtrip_ms"] = roundtrip_ms
        return resp


def parse_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    if dtype_name == "float64":
        return np.float64
    raise ValueError(f"unsupported dtype: {dtype_name}")


@dataclass
class ObfResult:
    a_enc: np.ndarray
    b_enc: np.ndarray
    deobf: Callable[[np.ndarray], np.ndarray]


def identity_deobf(c: np.ndarray) -> np.ndarray:
    return c


def _permute_2d(x: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    """Apply a 2D row/column permutation with one shared implementation path."""
    return x[row_idx, :][:, col_idx]


def obf_col_scale(
    a: np.ndarray,
    b: np.ndarray,
    scales: np.ndarray,
    inv_scales: np.ndarray,
) -> ObfResult:
    # W' = W D, recover Y by Y D^{-1}
    b_enc = b * scales[None, :]

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        return c_enc * inv_scales[None, :]

    return ObfResult(a, b_enc, deobf)


def obf_col_permute(
    a: np.ndarray,
    b: np.ndarray,
    perm: np.ndarray,
    inv_perm: np.ndarray,
) -> ObfResult:
    # W' = W Π, recover Y by Y Π^{-1}
    b_enc = b[:, perm]

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        return c_enc[:, inv_perm]

    return ObfResult(a, b_enc, deobf)


def obf_sparse_mask(
    a: np.ndarray,
    b: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    s_csr: sp.csr_matrix,
) -> ObfResult:
    # W' = W + S, recover Y by Y - X S
    b_enc32 = b.copy()
    # rows/cols are sampled without replacement, so advanced indexing add is safe.
    b_enc32[rows, cols] += vals
    b_enc = b_enc32

    a_local = a

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        # Sparse-dense correction: correction = X S.
        corr = a_local @ s_csr
        return c_enc - corr

    return ObfResult(a, b_enc, deobf)


def _recover_low_rank_outer(
    a: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Recover correction with rank-wise outer products to reduce kernel-dispatch bias."""
    corr = np.zeros((a.shape[0], v.shape[0]), dtype=a.dtype)
    for i in range(u.shape[1]):
        xu_i = a @ u[:, i]
        corr += xu_i[:, None] * v[:, i][None, :]
    return corr


def obf_low_rank_matmul(
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    l: np.ndarray,
) -> ObfResult:
    # W' = W + L (L = U V^T), recover Y by Y - X L.
    b_enc = b + l
    a_local = a

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        corr = (a_local @ u) @ v.T
        return c_enc - corr

    return ObfResult(a, b_enc, deobf)


def obf_low_rank_outer(
    a: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    l: np.ndarray,
) -> ObfResult:
    # W' = W + L (L = U V^T), recover Y by Y - X L.
    # This path computes X L using rank-wise outer products.
    b_enc = b + l
    a_local = a

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        corr = _recover_low_rank_outer(a_local, u, v)
        return c_enc - corr

    return ObfResult(a, b_enc, deobf)


def _apply_blockdiag_right(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply right multiplication by block diagonal 2x2 matrices.

    For each pair (2i, 2i+1), block is:
        [[1, alpha_i], [beta_i, 1 + alpha_i * beta_i]]
    """
    y = x.copy()
    num_pairs = alpha.shape[0]
    if num_pairs == 0:
        return y

    xp = x[:, : 2 * num_pairs].reshape(x.shape[0], num_pairs, 2)
    yp = y[:, : 2 * num_pairs].reshape(x.shape[0], num_pairs, 2)

    x0 = xp[:, :, 0]
    x1 = xp[:, :, 1]
    ab = alpha * beta

    yp[:, :, 0] = x0 + x1 * beta[None, :]
    yp[:, :, 1] = x0 * alpha[None, :] + x1 * (1.0 + ab)[None, :]
    return y


def _apply_blockdiag_inv_right(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply right multiplication by inverse block diagonal 2x2 matrices.

    Inverse block:
        [[1 + alpha_i * beta_i, -alpha_i], [-beta_i, 1]]
    """
    y = x.copy()
    num_pairs = alpha.shape[0]
    if num_pairs == 0:
        return y

    xp = x[:, : 2 * num_pairs].reshape(x.shape[0], num_pairs, 2)
    yp = y[:, : 2 * num_pairs].reshape(x.shape[0], num_pairs, 2)

    x0 = xp[:, :, 0]
    x1 = xp[:, :, 1]
    ab = alpha * beta

    yp[:, :, 0] = x0 * (1.0 + ab)[None, :] - x1 * beta[None, :]
    yp[:, :, 1] = -x0 * alpha[None, :] + x1
    return y


def _apply_blockdiag_inv_left(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply left multiplication by inverse of block diagonal 2x2 matrices.

    Inverse block:
        [[1 + alpha_i * beta_i, -alpha_i], [-beta_i, 1]]
    """
    y = x.copy()
    num_pairs = alpha.shape[0]
    if num_pairs == 0:
        return y

    xp = x[: 2 * num_pairs, :].reshape(num_pairs, 2, x.shape[1])
    yp = y[: 2 * num_pairs, :].reshape(num_pairs, 2, x.shape[1])

    x0 = xp[:, 0, :]
    x1 = xp[:, 1, :]
    ab = alpha * beta

    yp[:, 0, :] = (1.0 + ab)[:, None] * x0 - alpha[:, None] * x1
    yp[:, 1, :] = -beta[:, None] * x0 + x1
    return y


def obf_sparse_multiplicative(
    a: np.ndarray,
    b: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    perm_in: np.ndarray,
    perm_out: np.ndarray,
    inv_perm_in: np.ndarray,
    inv_perm_out: np.ndarray,
) -> ObfResult:
    # W' = W P, recover Y by Y P^{-1}
    # where P = Π_out * blkdiag(B_j) * Π_in and both P/P^{-1} are sparse.
    b1 = b[:, perm_out]
    b2 = _apply_blockdiag_right(b1, alpha, beta)
    b_enc = b2[:, perm_in]

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        c1 = c_enc[:, inv_perm_in]
        c2 = _apply_blockdiag_inv_right(c1, alpha, beta)
        return c2[:, inv_perm_out]

    return ObfResult(a, b_enc, deobf)


def obf_bilateral_permute(
    a: np.ndarray,
    b: np.ndarray,
    pm: np.ndarray,
    pk: np.ndarray,
    pn: np.ndarray,
    inv_pm: np.ndarray,
    inv_pn: np.ndarray,
) -> ObfResult:
    a_enc = _permute_2d(a, pm, pk)
    b_enc = _permute_2d(b, pk, pn)

    def deobf(c_enc: np.ndarray) -> np.ndarray:
        return _permute_2d(c_enc, inv_pm, inv_pn)

    return ObfResult(a_enc, b_enc, deobf)


def apply_operator(
    op_name: str,
    op_param: str,
    a: np.ndarray,
    b: np.ndarray,
    precomputed: dict[str, Any],
) -> ObfResult:
    pre = precomputed
    if op_name == "col_scale":
        return obf_col_scale(a, b, pre["scales"], pre["inv_scales"])
    if op_name == "col_permute":
        return obf_col_permute(a, b, pre["perm"], pre["inv_perm"])
    if op_name == "sparse_mask":
        return obf_sparse_mask(a, b, pre["rows"], pre["cols"], pre["vals"], pre["s_csr"])
    if op_name == "bilateral_permute":
        return obf_bilateral_permute(
            a,
            b,
            pre["pm"],
            pre["pk"],
            pre["pn"],
            pre["inv_pm"],
            pre["inv_pn"],
        )
    if op_name == "sparse_multiplicative":
        return obf_sparse_multiplicative(
            a,
            b,
            pre["alpha"],
            pre["beta"],
            pre["perm_in"],
            pre["perm_out"],
            pre["inv_perm_in"],
            pre["inv_perm_out"],
        )
    if op_name == "low_rank_mask":
        if pre["low_rank_rec_impl"] == "outer":
            return obf_low_rank_outer(a, b, pre["u_rec"], pre["v_rec"], pre["l"])
        return obf_low_rank_matmul(a, b, pre["u_rec"], pre["v_rec"], pre["l"])
    raise ValueError(f"unsupported operator: {op_name}:{op_param}")


def precompute_operator_factors(
    op_name: str,
    op_param: str,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    low_rank_rec_impl: str,
    low_rank_gemm_pad_to: int,
) -> dict[str, Any]:
    # Precomputation is not part of the online phase timing.
    dt = b.dtype

    if op_name == "col_scale":
        n = b.shape[1]
        scales = rng.uniform(0.5, 2.0, size=n).astype(dt, copy=False)
        inv_scales = (1.0 / scales).astype(dt, copy=False)
        return {"scales": scales, "inv_scales": inv_scales}

    if op_name == "col_permute":
        n = b.shape[1]
        perm = rng.permutation(n)
        inv_perm = np.argsort(perm)
        return {"perm": perm, "inv_perm": inv_perm}

    if op_name == "sparse_mask":
        k, n = b.shape
        sparsity = 0.99
        value_scale = 0.05
        nnz = max(1, int((1.0 - sparsity) * k * n))
        flat = rng.choice(k * n, size=nnz, replace=False)
        rows = (flat // n).astype(np.int32)
        cols = (flat % n).astype(np.int32)
        vals = (rng.standard_normal(nnz) * value_scale).astype(dt, copy=False)
        s_csr = sp.csr_matrix((vals, (rows, cols)), shape=(k, n), dtype=dt)
        return {"rows": rows, "cols": cols, "vals": vals, "s_csr": s_csr}

    if op_name == "low_rank_mask":
        rank = 1
        if op_param.startswith("r="):
            rank = int(op_param.split("=", 1)[1])
        k, n = b.shape
        alpha = 0.03
        u = (rng.standard_normal((k, rank)) * alpha).astype(dt, copy=False)
        v = (rng.standard_normal((n, rank)) * alpha).astype(dt, copy=False)
        l = u @ v.T  # offline precompute so online obfuscation is just addition
        if low_rank_rec_impl == "matmul":
            return {"u_rec": u, "v_rec": v, "l": l, "low_rank_rec_impl": "matmul"}
        if low_rank_rec_impl == "outer":
            return {"u_rec": u, "v_rec": v, "l": l, "low_rank_rec_impl": "outer"}
        if low_rank_rec_impl == "gemm_padded":
            pad_to = max(1, int(low_rank_gemm_pad_to))
            r_eff = max(rank, pad_to)
            if r_eff == rank:
                u_rec = u
                v_rec = v
            else:
                u_rec = np.zeros((k, r_eff), dtype=dt)
                v_rec = np.zeros((n, r_eff), dtype=dt)
                u_rec[:, :rank] = u
                v_rec[:, :rank] = v
            # Padded path still uses the same matmul implementation.
            return {"u_rec": u_rec, "v_rec": v_rec, "l": l, "low_rank_rec_impl": "matmul"}
        raise ValueError(f"unsupported low-rank recovery impl: {low_rank_rec_impl}")

    if op_name == "sparse_multiplicative":
        n = b.shape[1]
        scale = 0.25
        num_pairs = n // 2
        alpha = rng.uniform(-scale, scale, size=num_pairs).astype(dt, copy=False)
        beta = rng.uniform(-scale, scale, size=num_pairs).astype(dt, copy=False)
        perm_in = rng.permutation(n)
        perm_out = rng.permutation(n)
        inv_perm_in = np.argsort(perm_in)
        inv_perm_out = np.argsort(perm_out)
        return {
            "alpha": alpha,
            "beta": beta,
            "perm_in": perm_in,
            "perm_out": perm_out,
            "inv_perm_in": inv_perm_in,
            "inv_perm_out": inv_perm_out,
        }

    if op_name == "bilateral_permute":
        m, k = a.shape
        n = b.shape[1]
        pm = rng.permutation(m)
        pk = rng.permutation(k)
        pn = rng.permutation(n)
        inv_pm = np.argsort(pm)
        inv_pn = np.argsort(pn)
        return {"pm": pm, "pk": pk, "pn": pn, "inv_pm": inv_pm, "inv_pn": inv_pn}

    return {}


def parse_operators(text: str) -> list[tuple[str, str]]:
    # Example:
    # col_scale,col_permute,sparse_mask,low_rank_mask:r=1,low_rank_mask:r=2,low_rank_mask:r=3,bilateral_permute,sparse_multiplicative
    out: list[tuple[str, str]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            name, param = item.split(":", 1)
            out.append((name.strip(), param.strip()))
        else:
            out.append((item, ""))
    return out


def parse_models(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_batch_sizes(text: str) -> list[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def shape_representative(model: str, batch: int, seq_len: int, vit_seq_len: int) -> tuple[str, int, int, int, str]:
    # Returns: layer_name_or_id, M, K, N, matmul_op
    if model == "qwen2.5-0.5b":
        return "model.layers.0", batch * seq_len, 896, 4864, "mlp.up_proj"
    if model == "qwen2.5-1.5b":
        return "model.layers.0", batch * seq_len, 1536, 8960, "mlp.up_proj"
    if model == "bert-base":
        return "encoder.layer.0", batch * seq_len, 768, 3072, "intermediate.dense"
    if model == "vit":
        return "encoder.layer.0", batch * vit_seq_len, 768, 3072, "intermediate.dense"
    raise ValueError(f"unsupported model: {model}")


def one_run(
    client: RpcClient,
    dtype_name: str,
    model: str,
    batch: int,
    op_name: str,
    op_param: str,
    seq_len: int,
    vit_seq_len: int,
    force_k: int | None,
    force_n: int | None,
    rng: np.random.Generator,
    check_correctness: bool,
    inner_warmup: int,
    low_rank_rec_impl: str,
    low_rank_gemm_pad_to: int,
) -> dict[str, Any]:
    dt = parse_dtype(dtype_name)
    layer_name, m, k, n, matmul_op = shape_representative(model, batch, seq_len, vit_seq_len)
    if force_k is not None:
        k = force_k
    if force_n is not None:
        n = force_n

    a = rng.standard_normal((m, k)).astype(dt)
    b = rng.standard_normal((k, n)).astype(dt)

    # Offline precomputation (not counted in stage timings).
    pre = precompute_operator_factors(
        op_name,
        op_param,
        a,
        b,
        rng,
        low_rank_rec_impl=low_rank_rec_impl,
        low_rank_gemm_pad_to=low_rank_gemm_pad_to,
    )

    # Internal warmup per run (not counted):
    # warm the exact same operator path before timing.
    for _ in range(inner_warmup):
        obf_w = apply_operator(op_name, op_param, a, b, precomputed=pre)
        resp_w = client.matmul(obf_w.a_enc, obf_w.b_enc, dtype_name=dtype_name)
        c_w = np.frombuffer(resp_w["c_bytes"], dtype=dt).reshape(tuple(resp_w["c_shape"]))
        _ = obf_w.deobf(c_w)
        _ = client.matmul(a, b, dtype_name=dtype_name)

    # 1) TEE obfuscation
    t0 = time.perf_counter()
    obf = apply_operator(op_name, op_param, a, b, precomputed=pre)
    t_tee_obf_ms = (time.perf_counter() - t0) * 1000.0

    # 2) GPU encrypted compute + 4) transfer
    resp_cipher = client.matmul(obf.a_enc, obf.b_enc, dtype_name=dtype_name)
    c_enc = np.frombuffer(
        resp_cipher["c_bytes"],
        dtype=dt,
    ).reshape(tuple(resp_cipher["c_shape"]))

    t_gpu_cipher_ms = float(resp_cipher["gpu_ms"])
    t_transfer_ms = max(0.0, float(resp_cipher["roundtrip_ms"]) - t_gpu_cipher_ms)

    # 3) TEE de-obfuscation
    t1 = time.perf_counter()
    c_plain = obf.deobf(c_enc)
    t_tee_deobf_ms = (time.perf_counter() - t1) * 1000.0

    # 5) GPU plaintext baseline
    resp_base = client.matmul(a, b, dtype_name=dtype_name)
    t_gpu_plain_ms = float(resp_base["gpu_ms"])

    notes = ""
    if check_correctness:
        c_base = np.frombuffer(resp_base["c_bytes"], dtype=dt).reshape(tuple(resp_base["c_shape"]))
        max_abs_err = float(np.max(np.abs(c_plain.astype(np.float32) - c_base.astype(np.float32))))
        notes = f"max_abs_err={max_abs_err:.4e}"

    t_secure_total = t_tee_obf_ms + t_gpu_cipher_ms + t_tee_deobf_ms + t_transfer_ms
    overhead = t_secure_total / t_gpu_plain_ms if t_gpu_plain_ms > 0 else float("inf")

    return {
        "model": model,
        "layer_name_or_id": layer_name,
        "matmul_op": matmul_op,
        "M": m,
        "N": n,
        "K": k,
        "batch_size": batch,
        "operator": op_name,
        "operator_param": op_param,
        "t_tee_obfuscate_ms": t_tee_obf_ms,
        "t_gpu_cipher_ms": t_gpu_cipher_ms,
        "t_tee_deobfuscate_ms": t_tee_deobf_ms,
        "t_transfer_ms": t_transfer_ms,
        "t_gpu_plain_baseline_ms": t_gpu_plain_ms,
        "t_secure_total_ms": t_secure_total,
        "overhead_vs_baseline_x": overhead,
        "notes": notes,
    }


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TEE benchmark for obfuscated GEMM with Occlum")
    p.add_argument("--server-host", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=18080)
    p.add_argument(
        "--models",
        type=str,
        default="qwen2.5-0.5b,qwen2.5-1.5b,bert-base,vit",
    )
    p.add_argument("--batch-sizes", type=str, default="1,4,16,64")
    p.add_argument("--seq-len", type=int, default=128, help="Text model token length")
    p.add_argument("--vit-seq-len", type=int, default=128, help="ViT token length, 197 for real ViT-B/16")
    p.add_argument(
        "--operators",
        type=str,
        default=(
            "col_scale,col_permute,sparse_mask,"
            "low_rank_mask:r=1,low_rank_mask:r=2,low_rank_mask:r=3,"
            "bilateral_permute,sparse_multiplicative"
        ),
    )
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force-k", type=int, default=None, help="Override K dimension for all tested matmuls.")
    p.add_argument("--force-n", type=int, default=None, help="Override N dimension for all tested matmuls.")
    p.add_argument(
        "--inner-warmup",
        type=int,
        default=1,
        help="Per-run internal warmup iterations (not counted in timings).",
    )
    p.add_argument(
        "--lowrank-rec-impl",
        type=str,
        default="matmul",
        choices=["matmul", "gemm_padded", "outer"],
        help=(
            "Low-rank recovery implementation: "
            "matmul=(XU)@V^T, gemm_padded=pad rank to force GEMM-like path, "
            "outer=rank-wise outer-product accumulation."
        ),
    )
    p.add_argument(
        "--lowrank-gemm-pad-to",
        type=int,
        default=4,
        help="When --lowrank-rec-impl=gemm_padded, pad rank to at least this value.",
    )
    p.add_argument("--output-csv", type=str, default="results/occlum_timing_raw.csv")
    p.add_argument("--device-label", type=str, default="occlum+gpu-rpc")
    p.add_argument("--check-correctness", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    models = parse_models(args.models)
    batches = parse_batch_sizes(args.batch_sizes)
    ops = parse_operators(args.operators)

    ensure_parent(args.output_csv)

    fieldnames = [
        "record_id",
        "model",
        "layer_name_or_id",
        "matmul_op",
        "M",
        "N",
        "K",
        "batch_size",
        "operator",
        "operator_param",
        "t_tee_obfuscate_ms",
        "t_gpu_cipher_ms",
        "t_tee_deobfuscate_ms",
        "t_transfer_ms",
        "t_gpu_plain_baseline_ms",
        "t_secure_total_ms",
        "overhead_vs_baseline_x",
        "repeat_id",
        "device",
        "notes",
    ]

    rng = np.random.default_rng(args.seed)

    client = RpcClient(host=args.server_host, port=args.server_port)
    client.ping()

    print(
        (
            f"[tee] connected to {args.server_host}:{args.server_port}; "
            f"models={models}; batch={batches}; ops={ops}; repeats={args.repeats}; warmup={args.warmup}; "
            f"lowrank_rec_impl={args.lowrank_rec_impl}; lowrank_gemm_pad_to={args.lowrank_gemm_pad_to}"
        ),
        flush=True,
    )

    record_id = 1
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model in models:
            for batch in batches:
                for op_name, op_param in ops:
                    total_iters = args.warmup + args.repeats
                    for it in range(total_iters):
                        keep = it >= args.warmup
                        repeat_id = it - args.warmup + 1
                        do_check = args.check_correctness and keep and repeat_id == 1

                        row = one_run(
                            client=client,
                            dtype_name=args.dtype,
                            model=model,
                            batch=batch,
                            op_name=op_name,
                            op_param=op_param,
                            seq_len=args.seq_len,
                            vit_seq_len=args.vit_seq_len,
                            force_k=args.force_k,
                            force_n=args.force_n,
                            rng=rng,
                            check_correctness=do_check,
                            inner_warmup=args.inner_warmup,
                            low_rank_rec_impl=args.lowrank_rec_impl,
                            low_rank_gemm_pad_to=args.lowrank_gemm_pad_to,
                        )

                        if not keep:
                            continue

                        out = {
                            "record_id": record_id,
                            **row,
                            "repeat_id": repeat_id,
                            "device": args.device_label,
                        }
                        writer.writerow(out)
                        f.flush()

                        print(
                            (
                                f"[tee] #{record_id} {model} bs={batch} op={op_name}:{op_param or '-'} "
                                f"secure={row['t_secure_total_ms']:.3f}ms "
                                f"base={row['t_gpu_plain_baseline_ms']:.3f}ms "
                                f"ovh={row['overhead_vs_baseline_x']:.2f}x"
                            ),
                            flush=True,
                        )
                        record_id += 1

    client.close()
    print(f"[tee] done. results written to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
