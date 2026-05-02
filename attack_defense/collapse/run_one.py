"""Run a single (model, dataset, defense) experiment.

    python -m collapse.run_one <model> <dataset> <defense> [seed]
"""
from __future__ import annotations

import json
import sys

from .adapters import build_adapter
from .pipeline import run_and_record_experiment
from .registry import build_experiment_spec
from .specs import MODEL_SPECS


def run_one(model_key: str, dataset_key: str, defense_key: str, seed: int = 42) -> dict:
    experiment = build_experiment_spec(model_key, dataset_key, seed=seed, defense_key=defense_key)
    adapter = build_adapter(MODEL_SPECS[model_key], experiment)
    result = run_and_record_experiment(adapter, experiment, defense_key)
    return result.as_dict()


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python -m collapse.run_one <model> <dataset> <defense> [seed]", file=sys.stderr)
        sys.exit(1)
    model, dataset, defense = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    result = run_one(model, dataset, defense, seed=seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
