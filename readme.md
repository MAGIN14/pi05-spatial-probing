# pi05-spatial-probing

Investigating how spatial information is encoded inside π0.5, a
vision-language-action model for robot control.

We extract internal activations from π0.5's vision encoder, language
backbone, and action expert, then train linear probes to predict
ground-truth spatial properties (3D object position, depth, relative
position) obtained from the LIBERO simulator. The goal is to determine
which spatial properties are linearly decodable, where they emerge in
the network, and whether they are causally used in action generation.

Course project — Mathematical Foundations of Robotics.

## Contents
- `PROJECT_PLAN.md` — full roadmap, phases, and requirement mapping
- `SETUP.md` — reproducing the environment on Kaggle
- `research_log.md` — dated working log

## Status
Phase 1 — environment setup. π0.5 inference reproduced on Kaggle (T4).

## Status
Phase P1 — environment setup. **T1–T4 complete** (day 2 of 50).
π0.5 inference reproduced on Kaggle; LIBERO simulator rendering headlessly;
ground-truth object poses confirmed accessible. Both fatal-if-false
assumptions cleared. Next: T5 — closed-loop rollouts.