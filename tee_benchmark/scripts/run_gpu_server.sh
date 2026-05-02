#!/usr/bin/env bash
# Start the host-side GPU server. Runs OUTSIDE the enclave.
#
# Usage: scripts/run_gpu_server.sh [--host 127.0.0.1] [--port 18080] [--device cuda]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m tee_bench.gpu_server "$@"
