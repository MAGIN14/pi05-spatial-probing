"""Verify LIBERO frame and image conventions. See docs/conventions.md.

Run:
  echo "N" | /kaggle/working/openpi/examples/libero/.venv/bin/python \
      scripts/check_libero_conventions.py
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"          # must precede all imports
sys.path.insert(0, "/kaggle/working/openpi/third_party/libero")

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OUT = "/kaggle/working"

bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bench.get_task(0)
print("task:", task.language)

bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
env.seed(0)
obs = env.reset()

print("\n--- observation keys ---")
for k, v in sorted(obs.items()):
    print(f"  {k:<48} {getattr(v, 'shape', type(v))}")

# --- frame convention of *_to_robot0_eef_pos ---
print("\n--- frame convention ---")
bowl = obs["akita_black_bowl_1_pos"]
eef  = obs["robot0_eef_pos"]
rel  = obs["akita_black_bowl_1_to_robot0_eef_pos"]
quat = obs["robot0_eef_quat"]            # robosuite: (x, y, z, w)

d  = bowl - eef
Rm = R.from_quat(quat).as_matrix()       # gripper -> world
print("  norm(reported)   :", np.linalg.norm(rel))
print("  norm(bowl-eef)   :", np.linalg.norm(d))
print("  R.T @ (bowl-eef) :", Rm.T @ d,
      "MATCH" if np.allclose(rel, Rm.T @ d, atol=1e-5) else "NO MATCH")
print("  -> *_to_robot0_eef_pos is in the GRIPPER frame; "
      "use world-frame differences instead.")

# --- image orientation ---
print("\n--- image orientation ---")
raw = obs["agentview_image"]
Image.fromarray(raw).save(f"{OUT}/frame_raw.png")
Image.fromarray(np.ascontiguousarray(raw[::-1, ::-1])).save(f"{OUT}/frame_as_model_sees.png")
print("  saved frame_raw.png (upside down) and frame_as_model_sees.png")
print("  openpi applies [::-1, ::-1] (180 deg rotation) at main.py:115-116")
print("  label transform:  u_model = W-1-u ;  v_model = H-1-v")

env.close()