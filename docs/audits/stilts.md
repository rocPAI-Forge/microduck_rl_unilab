# Walking on stilts

**Result:** the walking policy carried through the stilt curriculum to 10 cm
stilts with a 17 × 22 mm tip. It walks straight at 0.15–0.25 m/s with 99–100%
survival over 60 s, but steering is biased: it drifts right at a standstill and
barely turns left. Simulation only; the stage-7 checkpoint is published as
[`examples/stilts/`](../../examples/stilts/) since 0.5.1.

## Source

After Vottivott/microduck-playground (Apache-2.0):
`src/mjlab_microduck/robot/stilt_constants.py` for the morphology,
`tasks/microduck_stilt_env_cfg.py` for the task, and
`experiments/stilts/TRAINING.md` for the executed curriculum. The playground's
printable stilt CAD (CC BY-NC-SA 4.0) is not copied; only the procedural
contact geometry is ported.

## Morphology

`scripts/make_stilt_robot.py --height-cm H --blend B` writes the BAM robot with
one convex loft under each sole: a rounded-rectangle root under the sole to a
tip `H` below it, the profile morphing from a 22 × 32 mm platform (`B = 0`) to
a 12 mm peg (`B = 1`); `B = 0.5` is a 17 × 22 mm tip. The stilt takes over the
`<side>_foot_collision` geom and `<side>_foot` site, so the contact sensors and
foot terms read the stilt tip; the original sole stays visible with its
collision off. Stilt mass is 12 g + 1 g/cm. The `home` keyframe is raised by
`H`. Files go to `assets/robots/microduck/generated/` (git-ignored).

## Task

`microduck_stilt_flat/mujoco` builds on the walking owner: upright 3.0, linear
tracking 2.5, yaw tracking 1.0, air time 2.0, head tracking 0.25, a quarter of
commands exact zero, forward −0.12 to 0.25 m/s, lateral ±0.06, yaw ±0.35; no
pushes, no CoM / head-range / standing curricula; action-rate cost ramped
−0.02 → −0.20; foot slip −0.1 − 0.1 × blend.

## Curriculum

`scripts/train_stilts.py` runs the playground's order (support first at 2 cm,
then height; never both in one step), resuming `velocity_flat_mujoco/model_499`
and then each previous stage, 2048 environments (about twice the playground's
iterations per stage, which ran 4096). Each stage is scored at 0.15 m/s for
10 s on 64 environments:

| Stage | Height | Blend | Iterations | Survival | Forward speed |
|---|---|---|---|---|---|
| 0 | 2 cm | 0.00 | 800 | 100% | 0.13 m/s |
| 1 | 2 cm | 0.25 | 200 | 100% | 0.13 |
| 2 | 2 cm | 0.50 | 200 | 100% | 0.13 |
| 3 | 3 cm | 0.50 | 200 | 100% | 0.16 |
| 4 | 4 cm | 0.50 | 200 | 100% | 0.18 |
| 5 | 5 cm | 0.50 | 1500 | 100% | 0.16 |
| 6 | 7.5 cm | 0.50 | 200 | 100% | 0.17 |
| 7 | 10 cm | 0.50 | 300 | 100% | 0.16 |

## 10 cm policy

`scripts/eval_walk.py`, 128 environments, seed 777:

| Command | Duration | Survival | Forward speed | Yaw rate |
|---|---|---|---|---|
| Stand | 60 s | 100% | 0.06 m/s | −0.10 rad/s |
| 0.15 m/s | 60 s | 99.2% | 0.18 | −0.16 |
| 0.25 m/s | 60 s | 99.2% | 0.24 | −0.13 |
| 0.1 m/s, +0.3 rad/s | 30 s | 100% | 0.15 | −0.05 |
| 0.1 m/s, −0.3 rad/s | 30 s | 100% | 0.14 | −0.17 |

Straight walking and speed tracking hold; the policy creeps forward and turns
right at a zero command and does not turn left. The playground reports the
straight-walking case only.

## Reproduce

```bash
uv run --no-sync scripts/train_stilts.py --load-run examples/velocity_flat_mujoco --checkpoint 499
uv run --no-sync scripts/eval_walk.py <stage-7 checkpoint> --owner microduck_stilt_flat/mujoco \
  --speed 0.15 --duration 60 --num-envs 128 \
  --override env.scene.model_file=assets/robots/microduck/generated/scene_flat_bam_stilt_h10_b050.xml \
  --override 'env.scene.fragment_files=[assets/robots/microduck/generated/locomotion_task_stilt_h10_b050.xml]'
```
