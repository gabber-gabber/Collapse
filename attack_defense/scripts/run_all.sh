#!/usr/bin/env bash
# Usage: scripts/run_all.sh [seed]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m collapse.run_all "$@"
