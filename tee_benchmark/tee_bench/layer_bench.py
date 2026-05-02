#!/usr/bin/env python3
"""Layer-level benchmark for full defense schemes on BERT-Base.

This benchmark measures one BERT layer (self-attention + feed-forward)
under three defense methods:
1) ArrowCloak: {L, D, Pi}
2) Oprior: {S, L, D, Pi}
3) OurDefense: input/weight additive masks + sparse mixing + double-sided transforms

For each method, we report four phases for self-attention and FFN:
- TEE obfuscation
- GPU compute
- TEE-GPU transfer
- TEE recovery
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import sparse as sp

from .primitive_bench import RpcClient, parse_dtype


@dataclass
class ObfResult:
    a_enc: np.ndarray
    b_enc: np.ndarray
    recover: Callable[[np.ndarray], np.ndarray]


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sample_sparse_mask(
    m: int,
    n: int,
    rng: np.random.Generator,
    dt: np.dtype,
    sparsity: float = 0.99,
    value_scale: float = 0.05,
) -> tuple[np.ndarray, sp.csr_matrix]:
    nnz = max(1, int((1.0 - sparsity) * m * n))
    flat = rng.choice(m * n, size=nnz, replace=False)
    rows = (flat // n).astype(np.int32)
    cols = (flat % n).astype(np.int32)
    vals = (rng.standard_normal(nnz) * value_scale).astype(dt, copy=False)
    dense = np.zeros((m, n), dtype=dt)
    dense[rows, cols] = vals
    csr = sp.csr_matrix((vals, (rows, cols)), shape=(m, n), dtype=dt)
    return dense, csr


def sample_low_rank(
    m: int,
    n: int,
    rank: int,
    rng: np.random.Generator,
    dt: np.dtype,
    alpha: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = (rng.standard_normal((m, rank)) * alpha).astype(dt, copy=False)
    v = (rng.standard_normal((n, rank)) * alpha).astype(dt, copy=False)
    l = u @ v.T
    return u, v, l


def sample_scale_perm(
    n: int,
    rng: np.random.Generator,
    dt: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scales = rng.uniform(0.5, 2.0, size=n).astype(dt, copy=False)
    inv_scales = (1.0 / scales).astype(dt, copy=False)
    perm = rng.permutation(n)
    inv_perm = np.argsort(perm)
    return scales, inv_scales, perm, inv_perm


def sample_sparse_mixing(
    k: int,
    rng: np.random.Generator,
    dt: np.dtype,
    scale: float = 0.25,
) -> dict[str, np.ndarray]:
    num_pairs = k // 2
    alpha = rng.uniform(-scale, scale, size=num_pairs).astype(dt, copy=False)
    beta = rng.uniform(-scale, scale, size=num_pairs).astype(dt, copy=False)
    perm_in = rng.permutation(k)
    perm_out = rng.permutation(k)
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


def perm_right_matrix(perm: np.ndarray) -> sp.csr_matrix:
    k = perm.shape[0]
    data = np.ones(k, dtype=np.float32)
    rows = perm.astype(np.int64, copy=False)
    cols = np.arange(k, dtype=np.int64)
    return sp.csr_matrix((data, (rows, cols)), shape=(k, k), dtype=np.float32)


def blockdiag_mix_matrix(alpha: np.ndarray, beta: np.ndarray, k: int, inverse: bool) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    num_pairs = alpha.shape[0]
    for i in range(num_pairs):
        r0 = 2 * i
        r1 = r0 + 1
        ab = float(alpha[i] * beta[i])
        if inverse:
            # inv(B_i) = [[1+ab, -alpha], [-beta, 1]]
            rows.extend([r0, r0, r1, r1])
            cols.extend([r0, r1, r0, r1])
            vals.extend([1.0 + ab, -float(alpha[i]), -float(beta[i]), 1.0])
        else:
            # B_i = [[1, alpha], [beta, 1+ab]]
            rows.extend([r0, r0, r1, r1])
            cols.extend([r0, r1, r0, r1])
            vals.extend([1.0, float(alpha[i]), float(beta[i]), 1.0 + ab])
    if k % 2 == 1:
        rows.append(k - 1)
        cols.append(k - 1)
        vals.append(1.0)
    mat = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(k, k), dtype=np.float32)
    return mat


def build_sparse_mixing_mats(mix: dict[str, np.ndarray], dt: np.dtype) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    k = int(mix["perm_in"].shape[0])
    p_out = perm_right_matrix(mix["perm_out"])
    p_in = perm_right_matrix(mix["perm_in"])
    p_out_inv = perm_right_matrix(mix["inv_perm_out"])
    p_in_inv = perm_right_matrix(mix["inv_perm_in"])
    b = blockdiag_mix_matrix(mix["alpha"], mix["beta"], k, inverse=False)
    b_inv = blockdiag_mix_matrix(mix["alpha"], mix["beta"], k, inverse=True)
    p = (p_out @ b @ p_in).astype(dt)
    p_inv = (p_in_inv @ b_inv @ p_out_inv).astype(dt)
    return p.tocsr(), p_inv.tocsr()


def precompute_arrowcloak(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    low_rank_rank: int,
) -> dict[str, Any]:
    dt = b.dtype
    k, n = b.shape
    u, v, l = sample_low_rank(k, n, low_rank_rank, rng, dt)
    scales, inv_scales, perm, inv_perm = sample_scale_perm(n, rng, dt)
    return {
        "u": u,
        "v": v,
        "l": l,
        "scales": scales,
        "inv_scales": inv_scales,
        "perm": perm,
        "inv_perm": inv_perm,
    }


def obf_arrowcloak(
    a: np.ndarray,
    b: np.ndarray,
    pre: dict[str, Any],
) -> ObfResult:
    b_enc = ((b + pre["l"]) * pre["scales"][None, :])[:, pre["perm"]]
    a_local = a

    def recover(c_enc: np.ndarray) -> np.ndarray:
        c = c_enc[:, pre["inv_perm"]]
        c = c * pre["inv_scales"][None, :]
        corr = (a_local @ pre["u"]) @ pre["v"].T
        return c - corr

    return ObfResult(a, b_enc, recover)


def precompute_oprior(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    low_rank_rank: int,
) -> dict[str, Any]:
    dt = b.dtype
    k, n = b.shape
    s_dense, s_csr = sample_sparse_mask(k, n, rng, dt)
    u, v, l = sample_low_rank(k, n, low_rank_rank, rng, dt)
    scales, inv_scales, perm, inv_perm = sample_scale_perm(n, rng, dt)
    return {
        "s_dense": s_dense,
        "s_csr": s_csr,
        "u": u,
        "v": v,
        "l": l,
        "scales": scales,
        "inv_scales": inv_scales,
        "perm": perm,
        "inv_perm": inv_perm,
    }


def obf_oprior(
    a: np.ndarray,
    b: np.ndarray,
    pre: dict[str, Any],
) -> ObfResult:
    b_enc = ((b + pre["s_dense"] + pre["l"]) * pre["scales"][None, :])[:, pre["perm"]]

    a_local = a

    def recover(c_enc: np.ndarray) -> np.ndarray:
        c = c_enc[:, pre["inv_perm"]]
        c = c * pre["inv_scales"][None, :]
        corr = (a_local @ pre["s_csr"]) + ((a_local @ pre["u"]) @ pre["v"].T)
        return c - corr

    return ObfResult(a, b_enc, recover)


def precompute_ourdefense(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    low_rank_rank: int,
) -> dict[str, Any]:
    dt = b.dtype
    m, k = a.shape
    _k, n = b.shape
    assert k == _k

    # Additive masks on input side
    sx_dense, _ = sample_sparse_mask(m, k, rng, dt)
    ux, vx, lx = sample_low_rank(m, k, low_rank_rank, rng, dt)
    rx_dense = sx_dense + lx

    # Additive masks on weight side
    sw_dense, _ = sample_sparse_mask(k, n, rng, dt)
    uw, vw, lw = sample_low_rank(k, n, low_rank_rank, rng, dt)
    rw_dense = sw_dense + lw

    # Sparse mixing in middle dimension k
    mix = sample_sparse_mixing(k, rng, dt)
    p, p_inv = build_sparse_mixing_mats(mix, dt)

    # Double-sided outer transforms (permutation instantiation)
    pm = rng.permutation(m)
    inv_pm = np.argsort(pm)
    pn = rng.permutation(n)
    inv_pn = np.argsort(pn)

    return {
        "rx_dense": rx_dense,
        "rw_dense": rw_dense,
        "p": p,
        "p_inv": p_inv,
        "pm": pm,
        "inv_pm": inv_pm,
        "pn": pn,
        "inv_pn": inv_pn,
    }


def obf_ourdefense(
    a: np.ndarray,
    b: np.ndarray,
    pre: dict[str, Any],
) -> ObfResult:
    rx_dense = pre["rx_dense"]
    rw_dense = pre["rw_dense"]
    p = pre["p"]
    p_inv = pre["p_inv"]
    pm = pre["pm"]
    inv_pm = pre["inv_pm"]
    pn = pre["pn"]
    inv_pn = pre["inv_pn"]

    a_enc = a + rx_dense
    a_enc = a_enc[pm, :]
    a_enc = a_enc @ p

    b_enc = b + rw_dense
    b_enc = p_inv @ b_enc
    b_enc = b_enc[:, pn]

    a_local = a
    b_local = b

    def recover(c_enc: np.ndarray) -> np.ndarray:
        c = c_enc[inv_pm, :][:, inv_pn]
        corr = (a_local @ rw_dense) + (rx_dense @ b_local) + (rx_dense @ rw_dense)
        return c - corr

    return ObfResult(a_enc, b_enc, recover)


def precompute_defense(
    defense: str,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    low_rank_rank: int,
) -> dict[str, Any]:
    if defense == "arrowcloak":
        return precompute_arrowcloak(a, b, rng, low_rank_rank=low_rank_rank)
    if defense == "oprior":
        return precompute_oprior(a, b, rng, low_rank_rank=low_rank_rank)
    if defense == "ourdefense":
        return precompute_ourdefense(a, b, rng, low_rank_rank=low_rank_rank)
    raise ValueError(f"unsupported defense: {defense}")


def apply_defense(
    defense: str,
    a: np.ndarray,
    b: np.ndarray,
    pre: dict[str, Any],
) -> ObfResult:
    if defense == "arrowcloak":
        return obf_arrowcloak(a, b, pre)
    if defense == "oprior":
        return obf_oprior(a, b, pre)
    if defense == "ourdefense":
        return obf_ourdefense(a, b, pre)
    raise ValueError(f"unsupported defense: {defense}")


def one_gemm_timed(
    client: RpcClient,
    defense: str,
    dtype_name: str,
    m: int,
    k: int,
    n: int,
    rng: np.random.Generator,
    low_rank_rank: int,
    inner_warmup: int,
    check_correctness: bool,
) -> tuple[float, float, float, float]:
    dt = parse_dtype(dtype_name)
    a = rng.standard_normal((m, k)).astype(dt)
    b = rng.standard_normal((k, n)).astype(dt)

    # Offline precomputation (not counted in stage timing)
    pre = precompute_defense(defense, a, b, rng, low_rank_rank=low_rank_rank)

    # Internal warmup (not counted)
    for _ in range(inner_warmup):
        obf_w = apply_defense(defense, a, b, pre)
        resp_w = client.matmul(obf_w.a_enc, obf_w.b_enc, dtype_name=dtype_name)
        c_w = np.frombuffer(resp_w["c_bytes"], dtype=dt).reshape(tuple(resp_w["c_shape"]))
        _ = obf_w.recover(c_w)

    # 1) obfuscation
    t0 = time.perf_counter()
    obf = apply_defense(defense, a, b, pre)
    t_obf = (time.perf_counter() - t0) * 1000.0

    # 2) gpu + 3) transfer
    resp = client.matmul(obf.a_enc, obf.b_enc, dtype_name=dtype_name)
    gpu_ms = float(resp["gpu_ms"])
    transfer_ms = max(0.0, float(resp["roundtrip_ms"]) - gpu_ms)
    c_enc = np.frombuffer(resp["c_bytes"], dtype=dt).reshape(tuple(resp["c_shape"]))

    # 4) recovery
    t1 = time.perf_counter()
    c_plain = obf.recover(c_enc)
    t_rec = (time.perf_counter() - t1) * 1000.0

    if check_correctness:
        c_ref = a @ b
        max_err = float(np.max(np.abs(c_plain.astype(np.float32) - c_ref.astype(np.float32))))
        if max_err > 1e-2:
            raise RuntimeError(f"correctness check failed: defense={defense}, max_err={max_err:.4e}")

    return t_obf, gpu_ms, transfer_ms, t_rec


def attn_ops(batch: int, seq_len: int) -> list[tuple[str, int, int, int, int]]:
    hidden = 768
    heads = 12
    head_dim = 64
    m = batch * seq_len
    return [
        ("qkv_proj", 3, m, hidden, hidden),
        ("o_proj", 1, m, hidden, hidden),
        ("qk", batch * heads, seq_len, head_dim, seq_len),
        ("attn_v", batch * heads, seq_len, seq_len, head_dim),
    ]


def ffn_ops(batch: int, seq_len: int) -> list[tuple[str, int, int, int, int]]:
    hidden = 768
    inter = 3072
    m = batch * seq_len
    return [
        ("ffn_intermediate", 1, m, hidden, inter),
        ("ffn_output", 1, m, inter, hidden),
    ]


def run_block(
    client: RpcClient,
    defense: str,
    dtype_name: str,
    ops: list[tuple[str, int, int, int, int]],
    rng: np.random.Generator,
    low_rank_rank: int,
    inner_warmup: int,
    check_correctness: bool,
) -> tuple[float, float, float, float]:
    obf_total = 0.0
    gpu_total = 0.0
    transfer_total = 0.0
    rec_total = 0.0

    for _op_name, count, m, k, n in ops:
        for _ in range(count):
            t_obf, t_gpu, t_transfer, t_rec = one_gemm_timed(
                client=client,
                defense=defense,
                dtype_name=dtype_name,
                m=m,
                k=k,
                n=n,
                rng=rng,
                low_rank_rank=low_rank_rank,
                inner_warmup=inner_warmup,
                check_correctness=check_correctness,
            )
            obf_total += t_obf
            gpu_total += t_gpu
            transfer_total += t_transfer
            rec_total += t_rec

    return obf_total, gpu_total, transfer_total, rec_total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-level defense overhead benchmark on BERT-Base")
    p.add_argument("--server-host", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=18080)
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=0, help="Number of unrecorded layer runs per defense")
    p.add_argument("--inner-warmup", type=int, default=1, help="Unrecorded warmup per GEMM")
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument("--low-rank-rank", type=int, default=1)
    p.add_argument("--output-csv", type=str, default="results/obfus_exp2_layer_raw.csv")
    p.add_argument("--summary-csv", type=str, default="results/obfus_exp2_layer_summary.csv")
    p.add_argument("--check-correctness", action="store_true")
    return p.parse_args()


def summarize_rows(rows: list[dict[str, Any]], out_csv: str) -> None:
    ensure_parent(out_csv)
    metrics = [
        "attn_obf_ms",
        "attn_gpu_ms",
        "attn_transfer_ms",
        "attn_rec_ms",
        "ffn_obf_ms",
        "ffn_gpu_ms",
        "ffn_transfer_ms",
        "ffn_rec_ms",
        "total_layer_ms",
    ]
    fieldnames = ["defense", "n_runs"] + [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r["defense"]), []).append(r)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for defense in ["arrowcloak", "oprior", "ourdefense"]:
            grp = grouped.get(defense, [])
            if not grp:
                continue
            out: dict[str, Any] = {"defense": defense, "n_runs": len(grp)}
            for m in metrics:
                vals = [float(x[m]) for x in grp]
                out[f"{m}_mean"] = f"{statistics.mean(vals):.9f}"
                out[f"{m}_std"] = f"{statistics.pstdev(vals):.9f}"
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_csv)
    ensure_parent(args.summary_csv)

    rng = np.random.default_rng(args.seed)
    defenses = ["arrowcloak", "oprior", "ourdefense"]
    attn = attn_ops(args.batch_size, args.seq_len)
    ffn = ffn_ops(args.batch_size, args.seq_len)

    print(
        (
            f"[layer] connect={args.server_host}:{args.server_port}; dtype={args.dtype}; "
            f"batch={args.batch_size}; seq={args.seq_len}; repeats={args.repeats}; warmup={args.warmup}; "
            f"inner_warmup={args.inner_warmup}; rank={args.low_rank_rank}"
        ),
        flush=True,
    )

    client = RpcClient(host=args.server_host, port=args.server_port)
    client.ping()

    fieldnames = [
        "record_id",
        "defense",
        "repeat_id",
        "attn_obf_ms",
        "attn_gpu_ms",
        "attn_transfer_ms",
        "attn_rec_ms",
        "ffn_obf_ms",
        "ffn_gpu_ms",
        "ffn_transfer_ms",
        "ffn_rec_ms",
        "total_layer_ms",
    ]

    rows: list[dict[str, Any]] = []
    record_id = 1
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for defense in defenses:
            total_iters = args.warmup + args.repeats
            for it in range(total_iters):
                keep = it >= args.warmup
                repeat_id = it - args.warmup + 1
                do_check = args.check_correctness and keep and repeat_id == 1

                a_obf, a_gpu, a_transfer, a_rec = run_block(
                    client=client,
                    defense=defense,
                    dtype_name=args.dtype,
                    ops=attn,
                    rng=rng,
                    low_rank_rank=args.low_rank_rank,
                    inner_warmup=args.inner_warmup,
                    check_correctness=do_check,
                )
                f_obf, f_gpu, f_transfer, f_rec = run_block(
                    client=client,
                    defense=defense,
                    dtype_name=args.dtype,
                    ops=ffn,
                    rng=rng,
                    low_rank_rank=args.low_rank_rank,
                    inner_warmup=args.inner_warmup,
                    check_correctness=do_check,
                )
                total = a_obf + a_gpu + a_transfer + a_rec + f_obf + f_gpu + f_transfer + f_rec

                if not keep:
                    continue

                row = {
                    "record_id": record_id,
                    "defense": defense,
                    "repeat_id": repeat_id,
                    "attn_obf_ms": a_obf,
                    "attn_gpu_ms": a_gpu,
                    "attn_transfer_ms": a_transfer,
                    "attn_rec_ms": a_rec,
                    "ffn_obf_ms": f_obf,
                    "ffn_gpu_ms": f_gpu,
                    "ffn_transfer_ms": f_transfer,
                    "ffn_rec_ms": f_rec,
                    "total_layer_ms": total,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)

                print(
                    (
                        f"[layer] #{record_id} defense={defense} repeat={repeat_id} "
                        f"attn={a_obf + a_gpu + a_transfer + a_rec:.3f}ms "
                        f"ffn={f_obf + f_gpu + f_transfer + f_rec:.3f}ms "
                        f"total={total:.3f}ms"
                    ),
                    flush=True,
                )
                record_id += 1

    client.close()
    summarize_rows(rows, args.summary_csv)
    print(f"[layer] raw={args.output_csv}", flush=True)
    print(f"[layer] summary={args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
