#!/usr/bin/env bash
# Usage: scripts/run_one.sh <model> <dataset> <defense> [seed]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m collapse.run_one "$@"
