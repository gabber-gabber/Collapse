#!/usr/bin/env python3
"""Summarize raw timing CSV into mean/std table."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Any


NUM_COLS = [
    "t_tee_obfuscate_ms",
    "t_gpu_cipher_ms",
    "t_tee_deobfuscate_ms",
    "t_transfer_ms",
    "t_gpu_plain_baseline_ms",
    "t_secure_total_ms",
    "overhead_vs_baseline_x",
]


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize occlum benchmark raw CSV")
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument("--output-csv", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["model"],
                row["layer_name_or_id"],
                row["matmul_op"],
                row["M"],
                row["N"],
                row["K"],
                row["batch_size"],
                row["operator"],
                row["operator_param"],
            )
            groups[key].append(row)

    out_fields = [
        "model",
        "layer_name_or_id",
        "matmul_op",
        "M",
        "N",
        "K",
        "batch_size",
        "operator",
        "operator_param",
        "n_runs",
        "tee_obfuscate_mean_ms",
        "tee_obfuscate_std_ms",
        "gpu_cipher_mean_ms",
        "gpu_cipher_std_ms",
        "tee_deobfuscate_mean_ms",
        "tee_deobfuscate_std_ms",
        "transfer_mean_ms",
        "transfer_std_ms",
        "gpu_plain_baseline_mean_ms",
        "gpu_plain_baseline_std_ms",
        "secure_total_mean_ms",
        "secure_total_std_ms",
        "overhead_vs_baseline_x",
    ]

    parent = os.path.dirname(args.output_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for key in sorted(groups.keys()):
            rows = groups[key]
            vals = {name: [float(r[name]) for r in rows] for name in NUM_COLS}

            tee_obf_mean, tee_obf_std = mean_std(vals["t_tee_obfuscate_ms"])
            gpu_cipher_mean, gpu_cipher_std = mean_std(vals["t_gpu_cipher_ms"])
            tee_deobf_mean, tee_deobf_std = mean_std(vals["t_tee_deobfuscate_ms"])
            transfer_mean, transfer_std = mean_std(vals["t_transfer_ms"])
            base_mean, base_std = mean_std(vals["t_gpu_plain_baseline_ms"])
            total_mean, total_std = mean_std(vals["t_secure_total_ms"])
            ovh_mean, _ = mean_std(vals["overhead_vs_baseline_x"])

            (
                model,
                layer_name_or_id,
                matmul_op,
                m,
                n,
                k,
                batch,
                op,
                op_param,
            ) = key

            writer.writerow(
                {
                    "model": model,
                    "layer_name_or_id": layer_name_or_id,
                    "matmul_op": matmul_op,
                    "M": m,
                    "N": n,
                    "K": k,
                    "batch_size": batch,
                    "operator": op,
                    "operator_param": op_param,
                    "n_runs": len(rows),
                    "tee_obfuscate_mean_ms": tee_obf_mean,
                    "tee_obfuscate_std_ms": tee_obf_std,
                    "gpu_cipher_mean_ms": gpu_cipher_mean,
                    "gpu_cipher_std_ms": gpu_cipher_std,
                    "tee_deobfuscate_mean_ms": tee_deobf_mean,
                    "tee_deobfuscate_std_ms": tee_deobf_std,
                    "transfer_mean_ms": transfer_mean,
                    "transfer_std_ms": transfer_std,
                    "gpu_plain_baseline_mean_ms": base_mean,
                    "gpu_plain_baseline_std_ms": base_std,
                    "secure_total_mean_ms": total_mean,
                    "secure_total_std_ms": total_std,
                    "overhead_vs_baseline_x": ovh_mean,
                }
            )

    print(f"summary written to {args.output_csv}")


if __name__ == "__main__":
    main()
