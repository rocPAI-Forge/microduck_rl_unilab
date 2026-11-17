# Scoot (seated "butt walk")

After Vottivott/microduck-playground's MicroDock v2 training: a seated duck
backs into the dock and scoots out. Owners build on the ground-contact standup
owner (`microduck_scoot_flat`, see its header for the reward layer). Accepted:
round 4, `microduck_scoot_sym_flat` `model_2999`.

## Rounds

| Round | Owner | Envs | Change |
|---|---|---|---|
| 1 | `microduck_scoot_flat` | 1024 | base recipe |
| 2 | `microduck_scoot_straight_flat` | 1024 | 30% straight resamples (75% backward), yaw-rate L1; warm start from round 1 |
| 3 | `microduck_scoot_straight_flat` | 4096 | round 2 from scratch on the convex-hull robot |
| 4 | `microduck_scoot_sym_flat` | 4096 | friction that takes effect, curriculum on the shins, mirror loss; from scratch |

4096 envs fit because `scripts/make_hull_robot.py` caps the collision meshes
at 128 hull vertices: the per-env model clone UniLab makes under per-env
randomization drops from 29.7 MB to 0.65 MB (5.3 GB peak RSS at 4096 envs,
2.3-2.8 s per iteration). Floor distances change by at most 0.32 mm on the
legs and feet and 1.3 mm on the top head shell, never closer.

## Why rounds 1-3 could not back up

- The floor's 1.0 friction wins MuJoCo's max-combination, so the seat
  friction curriculum (0.05 -> 1.0) never took effect
  ([`friction_combination.md`](friction_combination.md)).
- Replaying round 3's poses: the trunk never reaches the floor (z about 0.06 m);
  the duck rests on its shins and feet, which always gripped at 1.0.
- One leg paddled (left joint spread 2-4x the right), which is also the
  rightward spin; at backward commands the left foot touched only 12% of
  frames.

Round 4 raises robot collision geoms to contact priority 1 (the `_prio`
scene), runs the slippery-to-normal curriculum on the shins as well, and adds
the velocity tasks' mirror loss.

## Results

`scripts/eval_walk.py`, 128 envs, 10 s after a 1 s warm-up, seed 777. Shin
and seat friction at the first curriculum stage (0.05-0.25) and at the final
one (0.3-1.0, `--override "env.events.seat_friction.params.ranges=[0.3,1.0]"`).
Every run survived 100%.

| Command | Round 3 | Round 4, 0.05-0.25 | Round 4, 0.3-1.0 |
|---|---|---|---|
| vx -0.10 m/s | -0.008 m/s, -0.18 rad/s | **-0.079**, +0.011 | **-0.056**, +0.016 |
| vx -0.05 | -0.003, -0.10 | **-0.041**, -0.023 | **-0.029**, -0.006 |
| zero | 0.001, -0.02 | -0.003, -0.001 | -0.002, -0.007 |
| vx +0.05 | 0.032, -0.05 | 0.030, -0.067 | 0.021, -0.103 |
| wz +0.3 rad/s | +0.05 | +0.05 | +0.03 |
| wz -0.3 | -0.07 | -0.10 | -0.06 |

Round 4 contact replay at -0.10 m/s: each shin and foot touches the floor in
60-62% of frames, with matching left and right joint spread.

## Open

Turning on command reaches 0.03-0.10 of 0.3 rad/s, and forward 0.02-0.03 of
0.05 m/s with a rightward drift up to 0.10 rad/s.
