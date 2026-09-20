# SETUP.md — Running π0.5 on Kaggle

How to get from a blank Kaggle session to a working π0.5 inference call.
Every command below was run and verified. If something fails, check
[Troubleshooting](#troubleshooting) before improvising.

**Last verified:** 20/9/2026 · **Verified by:** Rahul Nigam

---

## 0. Environment

| Item | Value |
|---|---|
| Platform | Kaggle Notebooks (GPU accelerator, Internet ON) |
| GPU | _(fill in: T4 ×2 or P100 — check with `nvidia-smi`)_ |
| Python (uv venv) | 3.11 |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` — 11.6 GiB |
| Download time | ~3 minutes @ ~70 MiB/s |
| Inference speed | **1.88 s per `policy.infer()` call** (single sample, after warm-up) |

### Prerequisites (one-time, per account)

1. **Phone-verify your Kaggle account.** Notebooks have no internet access
   until you do, and both `uv sync` and the checkpoint download need it.
   Settings → Phone Verification.
2. In each new notebook: sidebar → **Accelerator: GPU** and **Internet: ON**.
   These are per-notebook settings.

### Session constraints

- Sessions are **ephemeral** — everything outside `/kaggle/working` is lost on
  disconnect, including the openpi install and the checkpoint.
- Setup must be repeated every session (~15–20 min including the download).
- `/kaggle/working` is ~20 GB with ~11 GB free after install — **the 11.6 GiB
  checkpoint does not fit there.** It lives in `/root/.cache/openpi`, which is
  on the overlay filesystem (~1 TB free).
- GPU sessions cap out around 9 hours; idle disconnect comes sooner.
- Weekly GPU quota is ~30 hours. Check it before starting long jobs.

---

## 1. Check the machine

Run as the first cell of every session and note the GPU in the research log.

```python
!nvidia-smi
!python --version
!df -h /root /kaggle/working
```

> **zsh/bash note:** don't paste `#` comments into `!` shell cells — they are
> not always treated as comments and get passed as arguments.

---

## 2. Install uv and clone openpi

```python
!pip install -q uv
%cd /kaggle/working
!git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
%cd /kaggle/working/openpi
```

**`--recurse-submodules` is required.** Without it you get confusing import
errors later. If already cloned without it:

```python
!git submodule update --init --recursive
```

Use the **HTTPS** URL, not the SSH one in openpi's README — there is no SSH
key on Kaggle.

---

## 3. Build the environment

```python
!GIT_LFS_SKIP_SMUDGE=1 uv sync
!GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**`GIT_LFS_SKIP_SMUDGE=1` is required on both commands** — it is needed to pull
LeRobot as a dependency. Do not drop it.

Verify JAX sees the GPU:

```python
!uv run python -c "import jax; print(jax.devices())"
```

Expect a CUDA device. CPU-only means JAX cannot see the GPU — fix that before
going further.

---

## 4. Fix the gsutil checkpoint download

**Without this the download fails.** gsutil refuses to fetch composite objects
when it cannot do fast CRC32c integrity checks, aborting with
`CommandException: N files/objects could not be transferred`.

```python
%%writefile /root/.boto
[GSUtil]
check_hashes = never
```

```python
!cat /root/.boto
!echo $HOME        # must be /root, since gsutil reads ~/.boto
```

You will then see `WARNING: Found no hashes to validate...` for each file.
**This is expected and harmless** — these are public, read-only weights, and a
corrupted checkpoint would fail loudly at load time.

> **Do not bother with `pip install crcmod`.** The wheel builds for Kaggle's
> Python (3.12), while the uv environment runs 3.11 and gsutil uses its own
> bundled interpreter — so it does not take effect.

If `BOTO_CONFIG` is set elsewhere in the environment, pass it explicitly:

```python
!cd /kaggle/working/openpi && BOTO_CONFIG=/root/.boto uv run python <script>
```

---

## 5. How to run code (important)

`uv sync` creates a virtualenv at `openpi/.venv`. **The Kaggle notebook kernel
is a different Python and cannot import openpi** — `import openpi.training`
raises `ModuleNotFoundError` in a normal cell.

**Always run openpi code through `uv run`:**

```python
!cd /kaggle/working/openpi && uv run python /kaggle/working/<script>.py
```

Write scripts with `%%writefile`, then execute them as above.

<details>
<summary>Interactive alternative (fragile)</summary>

```python
import sys
sys.path.insert(0, "/kaggle/working/openpi/.venv/lib/python3.11/site-packages")
```

Sometimes works, sometimes breaks on binary deps like jaxlib. Prefer `uv run`.
</details>

---

## 6. Verified inference script

```python
%%writefile /kaggle/working/test_infer.py
import time
from openpi.training import config as _config
from openpi.policies import policy_config, libero_policy
from openpi.shared import download

cfg = _config.get_config("pi05_libero")
ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
policy = policy_config.create_trained_policy(cfg, ckpt)

example = libero_policy.make_libero_example()
for k, v in example.items():
    print(k, "->", getattr(v, "shape", v), getattr(v, "dtype", ""))

policy.infer(example)          # warm-up: JAX JIT compiles here
t0 = time.time()
for _ in range(10):
    policy.infer(example)
print("action shape:", policy.infer(example)["actions"].shape)
print((time.time() - t0) / 10, "s per call")
```

```python
!cd /kaggle/working/openpi && uv run python /kaggle/working/test_infer.py
```

**Build the example with `libero_policy.make_libero_example()`.** Do not
hand-write the observation dict — the keys in openpi's README snippet
(`observation/exterior_image_1_left`, `observation/wrist_image_left`) are
**DROID's**, not LIBERO's, and will fail against the `pi05_libero` checkpoint.

The **warm-up call matters**: JAX compiles on first execution, and without it
the average is dominated by one-time compilation.

### Expected output

```
observation/state       -> (8,)          float64
observation/image       -> (224, 224, 3) uint8
observation/wrist_image -> (224, 224, 3) uint8
prompt                  -> do something
action shape: (10, 7)
1.88 s per call
```

---

## 7. Model facts confirmed by this run

Recorded here because they are the foundation of the Phase 2 token map.

| Fact | Value | Why it matters |
|---|---|---|
| Action chunk | `(10, 7)` | 10 future timesteps × 7 dims (6 EE deltas + gripper — confirm ordering in `LiberoOutputs`) |
| Image size | 224 × 224 | With 14×14 SigLIP patches → 16×16 grid → **256 patch tokens per camera** |
| Camera slots | 3 | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| Right wrist | **zero-padded, masked** | LIBERO has no right wrist cam. `image_mask` is `False` for π0-class models. **Do not mistake these 256 zero tokens for data.** |
| Total image tokens | 768 (256 real + 256 real + 256 padding) | |
| State | `(8,)`, passed through unchanged | Gripper pose is a **model input** → trivially decodable. Object positions are not. |
| Preprocessing | `_parse_image`: float→uint8, CHW→HWC | No-op for LIBERO's uint8 HWC input, but **verify no resize/flip downstream** before trusting pixel labels |
| Checkpoint size | 11.6 GiB | ≈ 3B params × 4 bytes → float32 weights, Orbax OCDBT format |

---

## 8. Cold-start script

Commit as `setup_kaggle.sh`; run it as cell one of every session.

```bash
#!/bin/bash
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
```

Usage:

```python
!git clone https://github.com/<user>/pi05-spatial-probing.git /kaggle/working/proj
!bash /kaggle/working/proj/setup_kaggle.sh
```

> **If using the PyTorch path**, the transformers replacement files must also be
> copied *every session* — add to the script:
> ```bash
> cp -r ./src/openpi/models_pytorch/transformers_replace/* \
>       .venv/lib/python3.11/site-packages/transformers/
> ```
> These add AdaRMS support, control activation precision, and allow the KV cache
> to be read without being updated. Only ever do this inside `.venv`.

---

---

## 9. LIBERO simulator (separate environment)

LIBERO runs in its **own Python 3.8 venv**, isolated from openpi's 3.11
environment. This is intentional — their dependencies conflict. The intended
architecture is two processes talking over a socket: simulator client +
policy server.

**Docker is the officially recommended path but does not work on Kaggle**
(no docker-compose, and `xhost +local:docker` needs a display). Use the
"without Docker" path below.

**Rendering needs no GPU** — run T3/T4-type work on a **CPU session** to save
GPU quota.

### 9.1 System libraries

```python
!apt-get update -qq && apt-get install -y -qq \
  libegl1 libgl1-mesa-glx libosmesa6-dev libglew-dev patchelf
```

### 9.2 Submodule

LIBERO ships as a submodule at `third_party/libero` — do not clone separately.

```python
%cd /kaggle/working/openpi
!git submodule update --init --recursive
```

### 9.3 LIBERO venv (Python 3.8)

```python
!pip install -q uv
%cd /kaggle/working/openpi
!uv venv --python 3.8 examples/libero/.venv
!examples/libero/.venv/bin/python --version     # expect 3.8.20

!uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match \
  --python examples/libero/.venv/bin/python
!uv pip install -e packages/openpi-client --python examples/libero/.venv/bin/python
!uv pip install -e third_party/libero    --python examples/libero/.venv/bin/python
```

`--python <path>` is required in notebooks — `source activate` does not persist
across cells.

Expect: robosuite 1.4.1, mujoco 3.2.3, libero 0.1.0, torch 1.11.0+cu113.

**Harmless warnings:** `incompatible with the project's Python requirement
>=3.11` (that is the point of a separate venv) and `Failed to hardlink files`
(Kaggle filesystem quirk).

### 9.4 The interactive-prompt trap

On first import LIBERO **prompts for a dataset path and hangs** in a notebook
cell (no stdin). Answer `N`:

```python
!echo "N" | /kaggle/working/openpi/examples/libero/.venv/bin/python <script>.py
```

Config is written to `/root/.libero/config.yaml`. **This recurs every session.**

The `[Warning]: datasets path ... does not exist!` that follows is **harmless**
— those are demonstration datasets for training; the simulator and BDDL files
are all we need.

### 9.5 Running LIBERO scripts (use this pattern)

Set `sys.path` **inside the script**, not via a shell `PYTHONPATH` export —
the export binds to the wrong command when piping `echo "N"`.

```python
import os, sys
os.environ["MUJOCO_GL"] = "egl"          # must be set BEFORE any import
sys.path.insert(0, "/kaggle/working/openpi/third_party/libero")
```

Then:

```python
!echo "N" | /kaggle/working/openpi/examples/libero/.venv/bin/python /kaggle/working/<script>.py
```

If EGL fails, the README recommends `MUJOCO_GL=glx` as the fallback.

### 9.6 Benchmark reference (π0.5 @ 30k, fine-tuned)

| Spatial | Object | Goal | Libero-10 | Average |
|---|---|---|---|---|
| 98.8 | 98.2 | 98.0 | 92.4 | **96.85** |

Use as the sanity target for T5 rollouts.

### 9.7 ⚠️ Conventions

**Read `docs/conventions.md` before writing any label code.** The image
orientation issue there (openpi applies a 180° rotation before the model sees
the frame) will silently corrupt every pixel label if missed.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CommandException: N files/objects could not be transferred` + CRC32c warnings | gsutil refuses composite objects without fast hashing | Write `/root/.boto` with `check_hashes = never` (§4) |
| `ModuleNotFoundError: No module named 'openpi.training'` | Notebook kernel ≠ uv venv | Run via `uv run` (§5) |
| `SyntaxError: ':' expected after dictionary key` | Placeholder `...` left in the example dict | Use `make_libero_example()` (§6) |
| `jax.devices()` shows CPU only | JAX built without CUDA / driver mismatch | Reinstall JAX for the correct CUDA version |
| Import errors for submodule packages | Cloned without `--recurse-submodules` | `git submodule update --init --recursive` |
| LeRobot dependency fails to pull | Missing `GIT_LFS_SKIP_SMUDGE=1` | Re-run both commands with the prefix |
| OOM on first inference | 11.6 GiB float32 weights on a 16 GB card | Check dtype handling; T4/P100 have **no native bf16** (pre-Ampere) |
| JAX grabs all GPU memory | Default allocator | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or lower) |
| Checkpoint download stalls | Network / GCS | Retry; `.partial` dir resumes. Delete it if resume misbehaves |
| No internet in notebook | Account not phone-verified, or Internet toggle off | Verify account; enable Internet in sidebar |

---

## Planning numbers

Derived from the measured **1.88 s/call**, for Phase 5 scoping:

| Frames | Single-sample inference time |
|---|---|
| 1,000 | ~31 min |
| 2,000 | ~63 min |
| 5,000 | ~2.6 h |
| 10,000 | ~5.2 h |

Against a ~30 GPU-hour weekly quota, and remembering that the random-init
baseline requires a **second full pass** over the same frames, plus rollouts
and re-runs after bugs: **target 2,000–5,000 frames.** That is ample for a
linear probe on a few hundred dimensions.

Headroom not yet used: **batching** (this measurement is one sample at a time;
JAX will do much better at batch 16–32) and **skipping the action expert** when
only backbone activations are needed. Re-measure once real extraction code
exists — `policy.infer` runs the full pipeline including all flow-matching
denoising steps.

---

## Change log

| Date | Author | Change |
|---|---|---|
| | | Initial version — Phase 1 setup verified on Kaggle |
