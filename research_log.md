# Research Log

Dated working log for **pi05-spatial-probing**.

**How to use this file:** add a new entry each working day, newest at the
bottom. Keep it short — what was run, what happened, what's next. Record
*numbers* and *decisions*, not narrative. This is the file the final report
gets written from, so a measurement written down today is a figure caption in
week 7.

**Entry template:**

```
## work - n

**Goal:**
**Done:**
**Blocked / issues:**
**Numbers / facts learned:**
**Decisions:**
**Next:**
```

---

## Work - 1

**Goal:** Clear the first fatal-if-false assumption — can we obtain and run
π0.5 at all?

**Done:**
- Confirmed local machine (MacBook Air) **cannot** run this project: no NVIDIA
  GPU, no CUDA, macOS unsupported by openpi (Ubuntu 22.04 only), and only
  9.5 GB free disk. Project is now **remote-compute**; Mac is SSH client,
  editor, and paper-reading machine only.
- Evaluated compute options. Chose **Kaggle Notebooks** over Colab free tier:
  fixed ~30 GPU-h/week quota, ~9 h sessions, persistent Datasets.
- Installed `uv`, cloned openpi with `--recurse-submodules`, ran
  `GIT_LFS_SKIP_SMUDGE=1 uv sync` + `uv pip install -e .` — successful.
- Downloaded the `pi05_libero` checkpoint (11.6 GiB, 16 objects).
- **Ran π0.5 inference successfully.** Printed observation shapes and action
  chunk shape.
- Measured per-call inference time with a warm-up call.

**Blocked / issues (both resolved):**
1. **gsutil refused the checkpoint download.**
   `CommandException: 6 files/objects could not be transferred`, with CRC32c
   integrity-check warnings on composite objects.
   → Fixed by writing `/root/.boto` with `check_hashes = never`.
   → Note: `pip install crcmod` does **not** help — the wheel builds for
     Kaggle's Python 3.12 while the uv venv runs 3.11 and gsutil uses its own
     bundled interpreter.
2. **`SyntaxError: ':' expected after dictionary key`** — the hand-written
   observation dict used DROID key names (`observation/exterior_image_1_left`,
   `observation/wrist_image_left`) with a placeholder `...`.
   → Fixed by using `libero_policy.make_libero_example()`, found at
     `src/openpi/policies/libero_policy.py:10`.
3. **`ModuleNotFoundError: No module named 'openpi.training'`** when importing
   in a normal notebook cell.
   → Cause: `uv sync` creates `openpi/.venv`, which the Kaggle kernel is not.
   → Fix: run all openpi code via `uv run`.

**Numbers / facts learned:**

| Fact | Value |
|---|---|
| Checkpoint size | 11.6 GiB (≈3B params × 4 bytes → **float32**, Orbax OCDBT) |
| Download time | ~3 min @ ~70 MiB/s — cheap enough to re-download each session |
| Inference | **1.88 s per `policy.infer()` call** (single sample, post warm-up) |
| `observation/state` | `(8,)` float64 |
| `observation/image` | `(224, 224, 3)` uint8 — agent view |
| `observation/wrist_image` | `(224, 224, 3)` uint8 |
| Action chunk | **`(10, 7)`** — 10 timesteps × 7 dims |
| Disk | `/root` on overlay, ~1 TB free. `/kaggle/working` only ~11 GB free — **checkpoint does not fit there** |

From reading `libero_policy.py`:
- Model takes **three** camera slots: `base_0_rgb`, `left_wrist_0_rgb`,
  `right_wrist_0_rgb`.
- LIBERO has no right wrist camera → that slot is **zero-padded and masked**
  (`image_mask` False for π0-class models). **768 image tokens total, of which
  256 are padding.** Must not be mistaken for data in the token map.
- 224×224 with 14×14 SigLIP patches → 16×16 grid → **256 patch tokens per
  camera**.
- `observation/state` is passed through unchanged → gripper pose is a **model
  input**, therefore trivially decodable. Object positions are not inputs —
  they are the real test of spatial perception.
- `_parse_image` handles float→uint8 and CHW→HWC only; a no-op for LIBERO's
  uint8 HWC input. **Still need to check for any resize/flip further down the
  pipeline** before trusting pixel labels.

**Decisions:**
- **Compute:** Kaggle Notebooks as primary. Re-download the checkpoint each
  session (~3 min) rather than fight the `/kaggle/working` size limit.
- **Scope:** working effectively solo for now (team members join later) →
  commit to the **Core tier** only (E2 3D position + E4 relations, single
  `pi05_libero` checkpoint). Extended tier only if teammates arrive.
- **Framework:** _(TBD — leaning PyTorch for forward hooks; note that
  `transformers_replace` files must be re-copied every session)_
- **Dataset size:** target **2,000–5,000 frames** based on the 1.88 s/call
  measurement against a ~30 GPU-h weekly quota, remembering the random-init
  baseline needs a second full pass.

**Next :** Install LIBERO, get headless rendering working
(`MUJOCO_GL=egl`), save one rendered frame. Check
`examples/libero/README.md` first — it may require a separate environment from
openpi's main one.

**Phase status:** P1 in progress. First fatal-if-false assumption **cleared**.

---

## Work - 2

**Goal:**
**Done:**
**Blocked / issues:**
**Numbers / facts learned:**
**Decisions:**
**Next:**