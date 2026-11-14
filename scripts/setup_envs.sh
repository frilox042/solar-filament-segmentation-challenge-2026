#!/usr/bin/env bash
# Create the two Python environments of this repository in one command (README, "Quickstart").
#
#   bash scripts/setup_envs.sh            # .venv-cpu (Python 3.12: CPU stage + notebook kernel) and .venv-nnunet (3.14)
#   bash scripts/setup_envs.sh --cpu-only # only .venv-cpu: enough for the tests and the CPU evaluation (Table 1)
#
# Uses python3.12 / python3.14 from PATH; when one is missing it installs it with uv (and installs uv itself if needed).
# Prints the three `export` lines the pipeline needs. Safe to re-run: an existing environment is reused.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
CPU_ONLY=0
[ "${1:-}" = "--cpu-only" ] && CPU_ONLY=1
TORCH_INDEX=https://download.pytorch.org/whl/cu130

need_python() {   # $1 = 3.12 | 3.14
  if command -v "python$1" >/dev/null 2>&1; then return; fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "python$1 not found; installing uv (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install "$1"
  export PATH="$HOME/.local/bin:$PATH"
}

make_venv() {   # $1 = 3.12 | 3.14, $2 = directory; an environment with pip is reused, one without (a failed run) rebuilt
  if [ -x "$2/bin/pip" ]; then return; fi
  rm -rf "$2"
  if ! "python$1" -m venv "$2"; then
    # e.g. a uv-installed Python: a venv made from its ~/.local/bin link cannot find the standard library
    echo "python$1 -m venv failed; creating $2 with uv instead"
    rm -rf "$2"
    uv venv -q --seed --python "$1" "$2"
  fi
}

need_python 3.12
make_venv 3.12 .venv-cpu
.venv-cpu/bin/pip install -q -r requirements/cpu.txt -r requirements/notebook.txt pytest==9.1.1   # (+ requirements/eval-test.txt for the optional torchmetrics check)
.venv-cpu/bin/pip install -q -e .                       # the package itself (filament_seg), importable from anywhere
echo "ok: .venv-cpu ($(.venv-cpu/bin/python --version))"

if [ "$CPU_ONLY" = 0 ]; then
  need_python 3.14
  make_venv 3.14 .venv-nnunet
  .venv-nnunet/bin/pip install -q --extra-index-url "$TORCH_INDEX" -r requirements/nnunet-gpu.lock.txt
  .venv-nnunet/bin/python -c "import torch, nnunetv2; print('ok: .venv-nnunet', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
fi

cat <<EOF

Now set (adjust the data path):
  export FILAMENT_KAGGLE_ROOT=/path/to/unzipped/filament-segmentation-2026   # contains MAGFiLO_1.0_Kaggle_2026/
  export FILAMENT_WORK=$ROOT/work
  export FILAMENT_PY_CPU=$ROOT/.venv-cpu/bin/python FILAMENT_PY_NNUNET=$ROOT/.venv-nnunet/bin/python
EOF
