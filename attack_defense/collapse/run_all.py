"""Run every (model, dataset, defense) cell in the experiment matrix and
write a summary CSV.

    python -m collapse.run_all [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .adapters import build_adapter
from .baselines import train_blackbox_model
from .io import aggregate_result_roots, write_summary
from .pipeline import run_and_record_experiment
from .registry import build_experiment_spec
from .specs import DEFENSE_SPECS, EXPERIMENT_MATRIX, MODEL_SPECS


def run_all(seed: int = 42) -> list[dict]:
    """Iterate the full matrix. Whitebox/blackbox baselines are computed once
    per (model, dataset) and reused across all defense cells; the trained
    blackbox model is also passed in as the prior for ArrowCloak / Oprior
    attacks instead of being retrained per cell.
    """
    rows: list[dict] = []
    for model_key, datasets in EXPERIMENT_MATRIX.items():
        for dataset_key in datasets:
            baselines, blackbox_model = train_blackbox_model(model_key, dataset_key, seed=seed)
            for defense_key in DEFENSE_SPECS:
                experiment = build_experiment_spec(model_key, dataset_key, seed=seed, defense_key=defense_key)
                adapter = build_adapter(MODEL_SPECS[model_key], experiment)
                result = run_and_record_experiment(
                    adapter, experiment, defense_key,
                    baselines=baselines,
                    blackbox_model=blackbox_model,
                )
                rows.append(result.as_dict())
    return rows


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rows = run_all(seed=seed)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    summary_path = write_summary(Path("results"), aggregate_result_roots(Path("results")))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
