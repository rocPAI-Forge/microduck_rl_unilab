# Turning in place (roadmap item 8, parked)

**Result:** three fine-tuning rounds from `sprint_stand/model_59000` did not
teach a pivot. No checkpoint is published. On hardware, turn around with a
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
space. The policy has the incentive to turn but not a pivot.

## Rounds

All rounds score on the driver battery (`straightspeed` owner, 128
environments, 30 s turns, seed 42). Yaw is measured for a ±0.4 rad/s command.

| Round | Owner | Change | Iterations | Pivot left / right | 0.15 m/s left / right |
|---|---|---|---|---|---|
| start | — | `model_59000` | — | +0.02 / −0.01 | +0.37 / −0.37 |
| 1 | `microduck_sprint_pivot_flat` | 25% turn commands, forward ceiling 0.2 → 0 m/s | 3000 | +0.03 / −0.02 | +0.37 / −0.42 |
| 2 | `microduck_sprint_pivothold_flat` | 35%, ceiling 0.06 → 0 m/s, half the round at 0 | 4000 | best +0.23 / −0.03 | +0.36 / −0.25 |
| 3 | `microduck_sprint_pivotsym_flat` | round 2 plus mirror augmentation and mirror loss 0.5 | 3000 | +0.03 / −0.03 | +0.05 / −0.06 |

Round 2 found a one-sided left pivot. Round 3 made the policy symmetric
(mirror loss 0.09 → 0.05) by dropping that pivot instead of learning the right
one, and it also lost the 0.15 m/s turn. Straight walking and survival held in
every round.

## The workaround

`model_59000`, 0.2 m/s forward, 60 s, 512 environments, seed 777, clamp on:

| Simulator | +0.4 rad/s | −0.4 rad/s | Forward speed |
|---|---|---|---|
| MuJoCo | +0.42 / 100% | −0.39 / 100% | 0.19–0.20 m/s |
| MotrixSim | +0.41 / 100% | −0.38 / 100% | 0.21 m/s |

## If a pivot is needed later

The likely routes are a separate stand-and-turn policy switched in onboard, or
a reward that pays for foot yaw while standing. The `turn_in_place_lin_max`
command field and `turn_in_place_curriculum` term stay in the code (both off by
default) for that work.
