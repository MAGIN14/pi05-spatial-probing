#!/bin/bash
# Cold-start setup for pi05-spatial-probing on Kaggle.
# Usage:
#   bash setup_kaggle.sh          # openpi only (GPU work)
#   bash setup_kaggle.sh libero   # openpi + LIBERO sim env (CPU work is fine)
set -e

pip install -q uv

cd /kaggle/working
if [ ! -d openpi ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
fi
cd openpi
git submodule update --init --recursive

# ---- openpi env (Python 3.11) ----
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# gsutil composite-object download fix (checkpoint fails without this)
cat > /root/.boto <<'EOF'
[GSUtil]
check_hashes = never
EOF

uv run python -c "import jax; print('JAX devices:', jax.devices())"

# ---- LIBERO env (Python 3.8) ----
if [ "$1" == "libero" ]; then
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf

  uv venv --python 3.8 examples/libero/.venv
  uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match \
    --python examples/libero/.venv/bin/python
  uv pip install -e packages/openpi-client --python examples/libero/.venv/bin/python
  uv pip install -e third_party/libero    --python examples/libero/.venv/bin/python

  echo "LIBERO env ready. NOTE: first run prompts for a dataset path -> pipe 'N'."
fi

echo "Setup complete."