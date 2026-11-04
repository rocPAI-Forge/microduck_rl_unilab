# Deploying a MicroDuck policy

This is the contract an onboard runtime has to reproduce to run an exported
policy. The reference export is
[`examples/sprint_stand/deploy/`](../examples/sprint_stand/deploy/):
`policy.onnx` plus `deploy_manifest.json`, from `sprint_stand/model_59000`,
which stands, walks 0–1.0 m/s, and steers with servo-clamped targets.

`scripts/export_policy.py` writes both files for any checkpoint and checks the
ONNX graph against the PyTorch policy on simulated observations. For
`model_59000` the largest action difference over 8000 observations is
2.4e-6.

## Loop

- Period: 0.02 s (50 Hz). One observation in, one action out.
- ONNX input `obs`, shape `[1, 61]`, float32; output shape `[1, 14]`. The
  observation normalizer is inside the graph, so feed raw values.
- Action to servo goal: `q_des = action * 1.0 + default_joint_pos`, in
  radians, per joint in the order below.
- Clamp `q_des` to the soft joint limits in the manifest
  (`goal_position_clamp: true`). The policy was trained with that clamp; a
  servo's goal-position limits do the same.

## Observation (61)

In this order (`deploy_contract.MICRODUCK_OBS_SEGMENTS`):

| Segment | Dim | Content |
|---|---:|---|
| `gyro` | 3 | Trunk angular velocity in the body frame, rad/s (IMU) |
| `gravity` | 3 | Gravity direction in the body frame, unit vector (IMU) |
| `joint_pos` | 14 | Measured joint position minus `default_joint_pos`, rad |
| `joint_vel` | 14 | Measured joint velocity, rad/s |
| `last_action` | 14 | The policy's previous raw action (zeros at start) |
| `twist_cmd` | 3 | Forward m/s, lateral m/s (always 0), yaw rate rad/s |
| `head_pose_cmd` | 4 | Zeros |
| `body_pose_cmd` | 6 | Zeros |

Joint order, for `joint_pos`, `joint_vel`, `last_action`, and the action:
`left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll, right_hip_yaw, right_hip_roll,
right_hip_pitch, right_knee, right_ankle`. Default positions and limits are in
the manifest.

Training perturbed the observation, which sets how much sensor error the
policy has seen:

| Segment | Uniform noise | Delay (control steps) |
|---|---|---|
| `gyro` | ±0.03 rad/s | 0–1 |
| `gravity` | ±0.01 | 0–1 |
| `joint_pos` | ±0.001 rad, plus a per-episode encoder bias | 0 |
| `joint_vel` | ±0.25 rad/s | 1 |

The IMU frame was also randomly misaligned by up to 6°. The BAM servo model
added 3–6 physics substeps of command delay and a 6.5–8.2 V supply.

## Commands

- **Stand:** `twist_cmd = [0, 0, 0]`.
- **Walk:** forward 0.15–1.0 m/s. Start near 0.6 m/s on hardware; see the
  limits below.
- **Turn:** yaw rate up to ±0.4 rad/s while walking. The policy does not
  turn in place.
- **Turn around:** send `[0.2, 0, ±0.4]`, a 0.5 m radius turn, instead of a
  pivot. Over 60 s on 512 environments it survives 100% on both MuJoCo and
  MotrixSim, turning at 0.38–0.42 rad/s and walking 0.19–0.21 m/s. A yaw
  command at a zero forward command turns at about 0.02 rad/s.
- **Hold a straight line:** run the same heading controller the policy was
  trained and scored with, and send its output as the yaw-rate command:
  `yaw = clip(2.5 * (heading_ref - heading) - 0.08 * cross_track, -0.35, 0.35)`,
  with `heading_ref` and `cross_track` measured from where the straight run
  started (cross-track positive to the left of that ray).

## Known limits

- **Top speed is simulator-dependent.** At 0–0.6 m/s the policy behaves the
  same on MuJoCo and MotrixSim. At 1.0 m/s, 300 s survival drops from about
  95% on MuJoCo to about 67% on MotrixSim
  ([`audits/motrix_crosscheck.md`](audits/motrix_crosscheck.md)).
- **Every deployable checkpoint passes the MotrixSim gate first.** Policies
  are trained on MuJoCo only; `scripts/sim2sim_check.py` checks them on the
  held-out MotrixSim over the 0–0.6 m/s band before they are published.
- **Only clamp-trained checkpoints are deployable.** `sprint_straight_head`,
  `sprint_steer`, and `sprint_stand` were trained with the clamp. The earlier
  sprint checkpoints command targets tens of radians past the `hip_yaw` and
  `neck_pitch` stops and fall within seconds once clamped
  ([`audits/position_targets.md`](audits/position_targets.md) and
  [`examples/sprint_straight_head/`](../examples/sprint_straight_head/)).
  The walk, stand-up, and get-up checkpoints keep their targets near the
  range.
- Backlash is not modeled, and the noise and delay envelopes above have not
  been calibrated against hardware.
- No turning in place; heading drifts about 6.5° over 300 s while standing
  on MuJoCo (1.4° on MotrixSim). Three training rounds did not teach a pivot
  ([`audits/turn_in_place.md`](audits/turn_in_place.md)); use the 0.5 m
  radius turn above.

## Re-export

```bash
uv run --no-sync scripts/export_policy.py examples/sprint_stand/model_59000.pt \
  --owner microduck_sprint_standturn_flat/mujoco --clip-targets \
  --output-dir examples/sprint_stand/deploy
rm examples/sprint_stand/deploy/policy.pt   # TorchScript copy, not needed onboard
```
