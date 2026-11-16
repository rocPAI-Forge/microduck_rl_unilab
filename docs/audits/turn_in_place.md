# Turning in place (roadmap item 8, parked)

**Result:** four fine-tuning rounds from `sprint_stand/model_59000` did not
teach a two-sided pivot; the best found a left pivot only. No checkpoint is published. On hardware, turn around with a
0.5 m radius turn (`[0.2, 0, ±0.4]`), which `model_59000` already does on both
simulators; see [`../DEPLOY.md`](../DEPLOY.md#commands).

## Where turning stops

`model_59000`, +0.4 rad/s command, 20 s, 128 environments, standturn owner:

| Forward command | Measured yaw | Survival |
|---|---|---|
| 0.3 m/s | 0.375 rad/s | 98.4% |
| 0.2 m/s | 0.413 rad/s | 99.2% |
| 0.1 m/s | 0.183 rad/s | 96.1% |
| 0 | 0.133 rad/s | 95.3% |

No reward term switches off at a zero forward command alone: the gait terms
and `leg_pose` gate on |forward| + |yaw|, and the head terms are in joint
space. But the yaw-tracking terms do work against a pivot.
`unilab.envs.mdp.track_angular_velocity` puts the roll and pitch rates in the
same exponent as the yaw error, and stepping in place makes both. The fine
copy of that term the sprint owners added since `sprint_steer` (std 0.2,
weight 6) therefore pays a standing robot more than a stepping one. Upstream
pollen-robotics/microduck_rl found the same dead zone with a counterfactual
(commits `e1d518d`, `4048a3e`). Rounds 1–3 ran with that term; round 4 makes
it yaw-only.

## Rounds

All rounds score on the driver battery (`straightspeed` owner, 128
environments, 30 s turns, seed 42). Yaw is measured for a ±0.4 rad/s command.

| Round | Owner | Change | Iterations | Pivot left / right | 0.15 m/s left / right |
|---|---|---|---|---|---|
| start | — | `model_59000` | — | +0.02 / −0.01 | +0.37 / −0.37 |
| 1 | `microduck_sprint_pivot_flat` | 25% turn commands, forward ceiling 0.2 → 0 m/s | 3000 | +0.03 / −0.02 | +0.37 / −0.42 |
| 2 | `microduck_sprint_pivothold_flat` | 35%, ceiling 0.06 → 0 m/s, half the round at 0 | 4000 | best +0.23 / −0.03 | +0.36 / −0.25 |
| 3 | `microduck_sprint_pivotsym_flat` | round 2 plus mirror augmentation and mirror loss 0.5 | 3000 | +0.03 / −0.03 | +0.05 / −0.06 |
| 4 | `microduck_sprint_pivotyaw_flat` | from `model_59000`: fine term yaw-only (`track_yaw_rate`, std 0.35), turn commands from 0.05 rad/s, 20%, ceiling 0.2 → 0 m/s | 3000 | best +0.22 / −0.02, final +0.15 / −0.03 | +0.39 / −0.37, final +0.45 / −0.34 |

Round 2 found a one-sided left pivot. Round 3 made the policy symmetric
(mirror loss 0.09 → 0.05) by dropping that pivot instead of learning the right
one, and it also lost the 0.15 m/s turn. Round 4 again found a left pivot
only, but kept the 0.15 m/s turns in both directions and the forward speeds
(1.0 m/s measures 0.85 against 0.88); in-place survival dipped to 70–78% at
two checkpoints. Straight walking held in every round.

## The workaround

`model_59000`, 0.2 m/s forward, 60 s, 512 environments, seed 777, clamp on:

| Simulator | +0.4 rad/s | −0.4 rad/s | Forward speed |
|---|---|---|---|
| MuJoCo | +0.42 / 100% | −0.39 / 100% | 0.19–0.20 m/s |
| MotrixSim | +0.41 / 100% | −0.38 / 100% | 0.21 m/s |

## If a pivot is needed later

The likely routes: round 4's yaw-only reward with mirror symmetry (round 3
added symmetry while the roll/pitch-coupled term was still in place), a
separate stand-and-turn policy switched in onboard, or a reward that pays for
foot yaw while standing. Upstream's own fix still leaves 0.3 rad/s in place
near zero. The `turn_in_place_lin_max`
command field and `turn_in_place_curriculum` term stay in the code (both off by
default) for that work.
