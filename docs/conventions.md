# Conventions and Frame Definitions

Verified empirically. **Read before writing anything in `labels/`.**
Every entry here was confirmed by running code, not by reading docs.

---

## 1. Ground-truth object poses

Available directly in the LIBERO observation dict — no MuJoCo internals needed.

| Key | Shape | Meaning |
|---|---|---|
| `<object>_1_pos` | (3,) | Object position, world frame |
| `<object>_1_quat` | (4,) | Object orientation, quaternion `(x, y, z, w)` |
| `robot0_eef_pos` | (3,) | End-effector position, world frame |
| `robot0_eef_quat` | (4,) | End-effector orientation `(x, y, z, w)` |

Example objects in `libero_spatial` task 0: `akita_black_bowl_1`,
`akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1`.

**scipy's `Rotation.from_quat()` expects `(x, y, z, w)` — same order as
robosuite. No reordering needed.**

## 2. `_to_robot0_eef_pos` is in the GRIPPER frame — do not use as a probe target

Verified: `X_to_robot0_eef_pos == R.T @ (p_object - p_eef)` where
`R = Rotation.from_quat(robot0_eef_quat).as_matrix()`.


**Why this matters:** the frame rotates with the wrist, so a stationary object
gets a changing label. Bad probe target.

**Decision:** compute relative quantities ourselves in the **world frame**.
E6 label = `p_object - p_eef` (world).

## 3. ⚠️ Image orientation — the critical one

The raw MuJoCo render is **upside down** (OpenGL bottom-left origin).

**openpi applies a 180° rotation before the model sees it**
(`examples/libero/main.py:115-116`):

```python
img       = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
```

`[::-1, ::-1]` reverses **rows AND columns** — a 180° rotation, *not* a
vertical flip. π0.5 therefore sees an image that is both vertically and
horizontally reversed relative to the raw render.

### Required label transform

For a pixel `(u, v)` obtained by projecting a 3D point into the **raw render**
of size `H x W`:

```python
u_model = W - 1 - u
v_model = H - 1 - v
patch_row = v_model // 14
patch_col = u_model // 14
```

**If this is skipped, every left/right relation is inverted and every patch
index is wrong — with no error raised.** The probe would simply appear to fail.

### Still to verify
`image_tools.resize_with_pad(img, args.resize_size, args.resize_size)` runs at
`main.py:118` *after* the rotation. At 224 -> 224 it should be a no-op.
**Check `args.resize_size`** — if padding ever occurs, the transform above
needs an extra offset term.

## 4. World frame origin

Not at the table surface:
- bowl z = 0.97
- end-effector z = 1.17

**TODO:** record the table surface height before defining "above" relations.

## 5. Model input shapes (from `libero_policy.make_libero_example()`)

| Field | Shape | Note |
|---|---|---|
| `observation/state` | (8,) float64 | Subset of the 39-dim `robot0_proprio-state` — **identify which 8** |
| `observation/image` | (224,224,3) uint8 | agentview |
| `observation/wrist_image` | (224,224,3) uint8 | eye-in-hand |
| action output | (10, 7) | 10 timesteps x 7 dims (6 EE delta + gripper — confirm ordering in `LiberoOutputs`) |

### Token map (derived)
- 224 / 14 = 16 -> **16x16 = 256 patch tokens per camera**
- Three camera slots: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
- LIBERO has no right wrist camera -> **zero-padded and masked**
  (`image_mask` False for pi0-class models)
- **768 image tokens total, of which 256 are padding.** Do not treat padding
  tokens as data.

### Confound note
`observation/state` is a **model input**, so gripper pose is trivially
decodable. Object positions are *not* inputs — they are the real test of
spatial perception.