from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .types import ExperimentResult, ExperimentSpec


def make_run_dir(spec: ExperimentSpec, defense_key: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(spec.output_root) / spec.model_key / spec.dataset_key / ts / defense_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result_bundle(run_dir: Path, spec: ExperimentSpec, defense_key: str, result: ExperimentResult) -> None:
    payload = {
        "experiment": asdict(spec),
        "defense": defense_key,
        "metrics": result.as_dict(),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with (run_dir / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write("defense,whitebox_acc,blackbox_acc,obfuscated_acc,attack_acc,recovered_layers\n")
        f.write(
            f"{defense_key},{result.whitebox_acc:.4f},{result.blackbox_acc:.4f},"
            f"{result.obfuscated_acc:.4f},{result.attack_acc:.4f},{result.recovered_layers}\n"
        )

    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(spec), f, indent=2, ensure_ascii=False)


def aggregate_result_roots(result_root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in result_root.glob("*/*/*/*/metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "model": payload["experiment"]["model_key"],
            "dataset": payload["experiment"]["dataset_key"],
            "defense": payload["defense"],
            **payload["metrics"],
        }
        rows.append(row)
    return rows


def write_summary(result_root: Path, rows: Iterable[dict]) -> Path:
    out_path = result_root / "summary.json"
    rows = list(rows)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
