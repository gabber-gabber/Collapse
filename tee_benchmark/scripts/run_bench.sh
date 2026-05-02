#!/usr/bin/env bash
# Run both benchmarks (per-primitive + per-defense layer) and summarize.
#
# Usage:
#     scripts/run_bench.sh                    # plain Python (for sanity-checking)
#
# Inside the Occlum enclave, invoke this via your usual occlum-run, e.g.:
#     $OCCLUM_DIR/bin/occlum-run /bin/python3 -m tee_bench.primitive_bench \
#         --output-csv /results/primitive_raw.csv
#     $OCCLUM_DIR/bin/occlum-run /bin/python3 -m tee_bench.layer_bench \
#         --output-csv /results/layer_raw.csv
#
# Requires `scripts/run_gpu_server.sh` to be running on the host first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results

python3 -m tee_bench.primitive_bench --output-csv results/primitive_raw.csv "$@"
python3 -m tee_bench.summarize results/primitive_raw.csv \
    --output results/primitive_summary.csv

python3 -m tee_bench.layer_bench --output-csv results/layer_raw.csv "$@"
python3 -m tee_bench.summarize results/layer_raw.csv \
    --output results/layer_summary.csv
