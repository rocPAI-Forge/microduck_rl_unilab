# MotrixSim cross-simulator check (roadmap item 3)

**Result:** the clamp-trained sprint checkpoints transfer to MotrixSim at
0–0.6 m/s and do not transfer at 1.0 m/s. At and below 0.6 m/s, survival,
straight-line holding, head posture, and turn tracking on Motrix match MuJoCo.
At 1.0 m/s, 300 s survival drops by 16–36 points on all three checkpoints,
so the top-speed gait depends on MuJoCo-specific contact or servo detail.
For hardware, start with forward commands near 0.6 m/s.

## Method

All three checkpoints were trained on MuJoCo only. They were evaluated with
`scripts/eval_sprint_speed.py --clip-targets` on the
`microduck_sprint_straightspeed_flat/motrix` owner: 512 environments,
seed 777, `motrixsim-core==0.8.2`. That owner drops the four events Motrix
cannot apply (interval push, joint-armature, foot-friction, and body-inertia
randomization), so the Motrix runs are, if anything, easier than the MuJoCo
batteries they are compared with. Straight runs are loop-held for 300 s;
turns are 60 s with the loop off. MuJoCo numbers are the 512-environment,
seed-777 batteries in each example's `metrics.json`.

## Straight, 300 s survival (straight-pose pass)

| Forward command | `sprint_stand/model_59000` Motrix | MuJoCo | `sprint_steer/model_55999` Motrix | MuJoCo | `sprint_straight_head/model_52250` Motrix | MuJoCo |
|---|---|---|---|---|---|---|
| 0 (stand) | 98.6% (98.6%) | 99.8% (80.1%) | — | — | — | — |
| 0.3 m/s | 99.8% (99.4%) | 100% (100%) | — | — | — | — |
| 0.5 m/s | — | — | 100% (100%) | 100% (100%) | 100% (100%) | 100% (100%) |
| 0.6 m/s | 99.6% (99.4%) | 99.8% (99.8%) | — | — | — | — |
| 1.0 m/s | **67.4%** (63.3%) | 95.1% (93.9%) | **57.4%** (48.2%) | 93.4% (92.0%) | **79.9%** (77.9%) | 96.5% (94.7%) |

Speed tracking also transfers. On Motrix, a 1.0 m/s command measures
0.90–0.99 m/s (MuJoCo: 0.88–0.93 m/s). Head offset stays at 2.4–5.2 cm.
Standstill heading drift on Motrix is 1.4°, against 6.5° on MuJoCo.

## Turning, 60 s, measured yaw rate / survival

| Checkpoint | Forward | −0.4 rad/s Motrix | MuJoCo | +0.4 rad/s Motrix | MuJoCo |
|---|---|---|---|---|---|
| `sprint_stand` | 0.5 m/s | −0.39 / 100% | −0.40 / 100% | +0.37 / 100% | +0.37 / 100% |
| `sprint_stand` | 1.0 m/s | −0.37 / **76.8%** | −0.40 / 98.4% | +0.33 / **90.4%** | +0.34 / 98.0% |
| `sprint_steer` | 0.5 m/s | −0.37 / 100% | −0.38 / 100% | +0.39 / 100% | +0.41 / 100% |
| `sprint_steer` | 1.0 m/s | −0.38 / **70.3%** | −0.35 / 97.7% | +0.36 / **83.8%** | +0.40 / 97.3% |

## Follow-up

Cap the hardware command band near 0.6 m/s. Training with Motrix in the loop
was tried and moved the overfit instead of removing it
([`motrix_training.md`](motrix_training.md) on
[PR #11](https://github.com/rocPAI-Forge/microduck_rl_unilab/pull/11)), so the
project now trains on MuJoCo only and keeps Motrix as the held-out sim2sim
check: `scripts/sim2sim_check.py` gates the 0–0.6 m/s band and reports
1.0 m/s (see the README section "Simulators"). The stress battery was not run
on Motrix because it relies on the interval push that Motrix cannot apply.

## Reproduce

```bash
uv pip install "unilab[motrix]==1.0.0"   # adds motrixsim-core 0.8.2
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_stand/model_59000.pt --clip-targets \
  --owner microduck_sprint_straightspeed_flat/motrix \
  --speed 1.0 --duration 300 --num-envs 512 --seed 777
```
