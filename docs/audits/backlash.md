# Gearbox backlash

**Result:** with 2° of total play per servo (±1°), `sprint_stand/model_59000`
loses 3.5–4 points of 60 s survival at 0–0.6 m/s and 22 points at 1.0 m/s on
MuJoCo. Turns are unaffected. Training on the backlash robot
(`microduck_sprint_standbacklash_flat`) recovers that fully but then stands
worse on a stiff robot (89.5% on MotrixSim, below the 95% gate), so no
checkpoint is published.

## Model

`scripts/add_backlash.py` ports upstream pollen-robotics/microduck_rl's
`add_backlash.py` (Apache-2.0). After every servo joint it inserts a passive
hinge on the same body and axis, limited to ±1°, with damping 0.01, armature
0.001, and a stiff limit (`solreflimit 0.01 1`). The link angle is servo
output plus play. BAM drives the servo joint and the encoder reads it, as on
the robot, so the 61-D observation is unchanged; the task entity lists the 14
servo joints by name, so the passive joints stay out of every observation,
action, and reward.

Generated files, checked against the script by `tests/test_backlash.py`:

- `assets/robots/microduck/microduck_bam_backlash.xml`
- `assets/robots/microduck/scene_flat_bam_backlash.xml`
- `assets/robots/microduck/locomotion_task_backlash.xml` (the `home` keyframe
  with a zero after every servo joint)

Owners: `microduck_sprint_straightspeed_flat/mujoco_backlash` (evaluation) and
`microduck_sprint_standbacklash_flat` (training).

Unpowered from the home pose, the passive joints settle at up to 1.0–1.3°: the
limit is stiff but not rigid, as upstream notes.

## `model_59000` with and without backlash

MuJoCo, 256 environments, seed 777, clamp on; straight runs 60 s with the
heading loop, turns 30 s at 0.5 m/s:

| Test | No backlash | ±1° backlash |
|---|---|---|
| Stand | 100% | 96.5% |
| 0.3 m/s | 100% (0.29 m/s) | 96.1% (0.28 m/s) |
| 0.6 m/s | 100% (0.57 m/s) | 96.1% (0.57 m/s) |
| 1.0 m/s | 100% (0.88 m/s) | 77.7% (0.89 m/s) |
| Turn +0.4 / −0.4 rad/s | +0.37 / −0.41, 100% | +0.40 / −0.40, 100% |

## Training on the backlash robot

`microduck_sprint_standbacklash_flat`: the standturn recipe on the backlash
robot, resumed from `model_59000` for 3000 iterations. 60 s, 256
environments, seed 777, clamp on:

| Checkpoint | Backlash: stand / 0.6 / 1.0 m/s | No backlash (MuJoCo): stand / 0.6 / 1.0 m/s | MotrixSim: stand / 0.6 / 1.0 m/s |
|---|---|---|---|
| `model_59000` | 96.5% / 96.1% / 77.7% | 100% / 100% / 100% | — |
| `model_60000` | **100% / 100% / 97.7%** | 95% / 100% / 98% | **89.5%** / 100% / 91.8% |
| `model_61500` | 99.2% / 100% / 96.1% | 99% / 100% / 94% | — |
| `model_61999` | 97.7% / 100% / 97.7% | 88% / 100% / 98% | — |

The no-backlash column is the driver battery (128 environments, seed 42).
Training on the backlash robot recovers it fully, and 1.0 m/s on Motrix rises
from 85% to 92%. But the policy now stands worse on a stiff robot: 88–99% on
MuJoCo without backlash and 89.5% on Motrix, below the 95% sim2sim gate, where
`model_59000` holds 98.6% for 300 s. Slow walking at 0.15 m/s also stalls at
some checkpoints (0.02–0.03 m/s against 0.11). No checkpoint is published.

A policy for the robot needs to hold on both: train across backlash amounts
(0 to ±1°), for example by alternating backlash and stiff phases on MuJoCo,
since UniLab 1.0.0 cannot randomize joint ranges per environment.

## Reproduce

```bash
uv run --no-sync scripts/add_backlash.py --backlash-deg 2.0
uv run --no-sync scripts/eval_sprint_speed.py examples/sprint_stand/model_59000.pt \
  --clip-targets --owner microduck_sprint_straightspeed_flat/mujoco_backlash \
  --speed 1.0 --duration 60 --num-envs 256 --seed 777
```
