# Position-target audit (roadmap item 4)

**Result:** only the sprint lineage leans on the joint stops. The walk,
stand-up, and get-up checkpoints keep their position targets near the joint
range, and every one of them still finishes its task with the BAM
goal-position clamp on. None needs retraining.

## Why this was checked

The BAM action does not clamp `q_des`. A policy can hold a joint on its stop
by commanding a target far past it, which keeps the servo at full voltage. A
real servo clamps its goal position. Sprint checkpoints from `sprint_speed_1p68`
through `sprint_straight_cmd` do this on `hip_yaw` and `neck_pitch` and do
not survive with the clamp on. See
[`examples/sprint_straight_head/`](../../examples/sprint_straight_head/).

## Method

`scripts/audit_action_targets.py` runs a checkpoint under its own task owner,
records how far each requested target goes past the soft joint limits, and
reruns the same rollout with `--clip-targets`. Every row below is 128
environments, seed 42. "Upright" is the share of environments with trunk tilt
under 20° at the end of the run.

Forced-pose runs use the overrides from each example README: the
`ground_state_mix`, `prone_init_prob`, and `fell_over_disable` curricula off,
one start pose at a time, and for get-up the tilt termination off with a
stand-still command.

## Results

| Scenario | Checkpoint | Duration | Summed overshoot per step | Upright, unclamped | Upright, clamped |
|---|---|---:|---:|---:|---:|
| Walk, default owner | `velocity_flat_mujoco/model_499` | 10 s | 0.004 rad | 98.4% | 99.2% |
| Stand up from face-down | `standup/model_14999` | 8 s | 0.11 rad | 100% | 100% |
| Stand up from face-up | `standup/model_14999` | 8 s | 0.15 rad | 100% | 100% |
| Stand up from sitting | `standup/model_14999` | 8 s | 0.06 rad | 100% | 99.2% |
| Get up from face-down | `velstand_getup/model_12498` | 8 s | 0.12 rad | 100% | 100% |
| Get up from face-up | `velstand_getup/model_12498` | 8 s | 0.62 rad | 100% | 100% |
| Walk, default owner | `velstand_getup/model_12498` | 10 s | 0.001 rad | 100% | 100% |

For scale, the sprint checkpoints:

| Checkpoint | Summed overshoot per step |
|---|---:|
| `sprint_straight_cmd/model_32250` (0.6 m/s) | 141.5 rad, mostly `hip_yaw` (~101 + ~23) and `neck_pitch` (~17) |
| `sprint_straight_head/model_52250` (0.5 m/s, clamp-trained) | 0.13 rad |

No single joint in the non-sprint runs averages more than about 0.12 rad of
overshoot, and most of it falls in the get-up transient. The walk policy's
non-timeout terminations went from 4 to 9 over 1280 environment-seconds with
the clamp on, with interval pushes active in that owner.

## Reproduce

```bash
uv run --no-sync scripts/audit_action_targets.py \
  examples/standup/model_14999.pt --owner microduck_standup_flat/mujoco \
  --num-envs 128 --duration 8 [--clip-targets] \
  --override env.curriculum.ground_state_mix=null \
  --override env.events.set_ground_state.params.face_down_prob=1 \
  --override env.events.set_ground_state.params.face_up_prob=0 \
  --override env.events.set_ground_state.params.sitting_prob=0 \
  --override env.events.set_ground_state.params.standing_prob=0
```

The other rows swap the owner, checkpoint, and pose overrides; the get-up
overrides are the ones in
[`examples/velstand_getup/README.md`](../../examples/velstand_getup/README.md).
