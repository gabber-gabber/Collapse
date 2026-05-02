# Artifact: Obfuscation-Based On-Device LLM Protection

Two folders, two independent experiments.

| Folder            | What it measures                                                  |
|-------------------|-------------------------------------------------------------------|
| `attack_defense/` | Attack accuracy across `(model, dataset, defense)` cells.         |
| `tee_benchmark/`  | Per-primitive / per-defense latency inside an Occlum SGX enclave. |

---

## 1. How to run

### 1.1 `attack_defense/`

Setup:

```bash
cd attack_defense
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For locally fine-tuned Qwen GLUE checkpoints, set:
```bash
export CCS_VICTIM_MODEL_ROOT=/path/to/glue_outputs
```

Run one cell:
```bash
scripts/run_one.sh <model> <dataset> <defense> [seed]
# scripts/run_one.sh bert_base mnli oprior
```

Run the full matrix + summary:
```bash
scripts/run_all.sh [seed]
```

Models, datasets, and defenses are defined in `collapse/specs.py` (`MODEL_SPECS`, `DEFENSE_SPECS`, `EXPERIMENT_MATRIX`).

Results: `results/<model>/<dataset>/<timestamp>/<defense>/{metrics.json, run_meta.json}`. Aggregate: `results/summary.csv`.

### 1.2 `tee_benchmark/`

Setup:

```bash
cd tee_benchmark
pip install -r requirements.txt
```

The benchmark has two sides. The GPU server runs on the host; the bench runs inside an Occlum enclave (or in plain Python for sanity-checking — wall-clock won't reflect SGX overhead).

```bash
# host (in one terminal):
scripts/run_gpu_server.sh

# enclave (in another terminal): runs primitive + layer bench, writes summary CSVs
scripts/run_bench.sh
```

To run inside Occlum, invoke `tee_bench.primitive_bench` and `tee_bench.layer_bench` through your existing `occlum-run`, e.g.

```bash
$OCCLUM_DIR/bin/occlum-run /bin/python3 -m tee_bench.primitive_bench \
    --output-csv /results/primitive_raw.csv
$OCCLUM_DIR/bin/occlum-run /bin/python3 -m tee_bench.layer_bench \
    --output-csv /results/layer_raw.csv
```

then summarize on the host: `python3 -m tee_bench.summarize <raw.csv> --output <summary.csv>`.

---

## 2. Code structure

### `attack_defense/`

```
attack_defense/
├── scripts/{run_one,run_all}.sh   ← entry points (call python -m collapse.run_{one,all})
├── collapse/
│   ├── run_one.py, run_all.py     ← module entry points
│   ├── pipeline.py                ← per-cell driver: baseline → obfuscate → attack → finetune → eval
│   ├── defenses.py                ← obfuscation algorithms (NNSplitter / LoRO / TSQP / TLG / ArrowCloak / Oprior / OurDefense)
│   ├── attacks.py                 ← attack pipeline; the 3-stage COLLAPSE attack against Oprior lives here
│   ├── specs.py                   ← model / dataset / defense / matrix definitions
│   └── (adapters, baselines, datasets, finetune, io, primitives, registry, types, utils)
└── requirements.txt
```

Read order: `specs.py` → `pipeline.py` → `defenses.py` + `attacks.py`.

### `tee_benchmark/`

```
tee_benchmark/
├── scripts/
│   ├── run_gpu_server.sh                     ← host: start GPU RPC server
│   └── run_bench.sh                          ← enclave: primitive + layer bench + summarize
├── tee_bench/
│   ├── gpu_server.py                         ← host: TCP server, runs GEMM on GPU
│   ├── primitive_bench.py                    ← enclave: per-primitive obfuscate/recover timing
│   ├── layer_bench.py                        ← enclave: full BERT layer under each defense
│   └── summarize.py                          ← raw csv → mean ± std
└── requirements.txt
```

Each measured round: `obfuscate (TEE) → send → GEMM (GPU) → recv → deobfuscate (TEE)` over a local TCP socket, plus a plaintext baseline that skips obfuscate/deobfuscate.
