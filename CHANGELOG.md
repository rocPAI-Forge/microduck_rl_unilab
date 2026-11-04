# Changelog

Versions follow `project.version` in `pyproject.toml`. Each release is a git
tag `v<version>` with a matching GitHub release. Checkpoints live in
`examples/` and are versioned with the code that reproduces them.

## Unreleased

- `microduck_sprint_standturn_flat/motrix` fine-tunes the standing sprint on
  Motrix, and `scripts/alt_backend_finetune.sh` alternates Motrix and MuJoCo
  phases. The experiment in
  [`docs/audits/motrix_training.md`](docs/audits/motrix_training.md) found
  that Motrix-only training moves the overfit to Motrix, while alternating
  holds 84–88% of 300 s survival at 1.0 m/s on both simulators. No checkpoint
  is published.

## 0.2.0

Completes the motion interface of the servo-clamped sprint except for turning
in place, cross-checks it on MotrixSim, and exports it for onboard inference.
Dependencies are unchanged from 0.1.0.

### Checked policies

| Example | Command | Result |
|---|---|---|
| `sprint_steer/model_55999` | 0.3–1.0 m/s forward, ±0.4 rad/s yaw | yaw tracked within about 0.05 rad/s, 97–100% survival while turning |
| `sprint_stand/model_59000` | 0–1.0 m/s forward, ±0.4 rad/s yaw | 99.8% survival standing for 300 s; does not turn in place |

Both were trained with the goal-position clamp on; their eval, diagnose, and
render commands need `--clip-targets`. `sprint_stand` supersedes
`sprint_straight_head` as the starting point for hardware.

### Known limitations

- On MotrixSim the sprint checkpoints hold at 0–0.6 m/s but lose 16–36 points
  of 300 s survival at 1.0 m/s. Start hardware near 0.6 m/s.
- Turning in place is not learned.
- The hardware side of deployment (servo calibration, first on-robot run) is
  open. MJWarp training still needs an NVIDIA GPU.

### Changes

- `scripts/audit_action_targets.py` reports how far a checkpoint's position
  targets go past the joint limits under any task owner, with and without the
  goal-position clamp. The audit in
  [`docs/audits/position_targets.md`](docs/audits/position_targets.md) found
  that only the sprint lineage leans on the stops; the walk, stand-up, and
  get-up checkpoints finish their tasks with the clamp on.
- `examples/sprint_steer/model_55999.pt` adds yaw-rate steering to the
  head-over-trunk, servo-clamped gait: 0.3–1.0 m/s forward and ±0.4 rad/s
  yaw tracked within about 0.05 rad/s, 97–100% survival while turning. New
  owners `microduck_sprint_steer_flat` and `microduck_sprint_steermix_flat`.
- `MicroduckVelocityCommandCfg.spawn_hold_fraction` runs the spawn-heading
  loop on a random share of episodes. `heading_hold`, `heading_abs_cost`, and
  `cross_track_abs_cost` take `gate_command` to score only those episodes.
  Both default to the previous behavior.
- `eval_sprint_speed.py` and `render_straight_demos.py` take `--yaw-rate`, and
  the training driver scores a turning battery (`--turn-speeds`,
  `--turn-yaw-rates`, `--max-yaw-error`).
- `examples/sprint_stand/model_59000.pt` adds standing still to the steering
  gait: 0–1.0 m/s forward, 99.8% survival standing for 300 s with about 9 cm
  of drift. It does not turn in place. New owners
  `microduck_sprint_stand_flat` and `microduck_sprint_standturn_flat`.
- `MicroduckSprintFlat` is registered on Motrix for evaluation, with a
  `microduck_sprint_straightspeed_flat/motrix` owner, and
  `eval_sprint_speed.py` follows the owner's `sim_backend`. The cross-check in
  [`docs/audits/motrix_crosscheck.md`](docs/audits/motrix_crosscheck.md) found
  the clamp-trained sprint checkpoints transfer at 0–0.6 m/s but lose 16–36
  points of 300 s survival at 1.0 m/s.
- `scripts/export_policy.py` writes a single-file ONNX, a TorchScript copy,
  and a `deploy_manifest.json`, and checks the ONNX against the PyTorch
  policy. `examples/sprint_stand/deploy/` holds the export of `model_59000`
  (largest action difference 2.4e-6 over 8000 observations), and
  [`docs/DEPLOY.md`](docs/DEPLOY.md) documents the onboard contract.

## 0.1.0

First tagged release. Depends on `unilab[mujoco]==1.0.0`, `unilab-rl==1.0.0`,
and `unisim-core==1.0.0` from PyPI. The default install profile is ROCm with
`torch==2.11.0`; the CUDA profile is `make sync-cuda`.

### Tasks and checked policies

| Policy | Checkpoint | Status |
|---|---|---|
| `microduck_velocity_flat` PPO, MuJoCo | `examples/velocity_flat_mujoco/model_499.pt` | Checked: stands and steps at 0.4 m/s |
| `microduck_velocity_flat` PPO, Motrix | `examples/velocity_flat_motrix/model_499.pt` | Trained; no inference video |
| `microduck_standup_flat` PPO | `examples/standup/model_14999.pt` | Checked: prone, supine, and sit-to-stand |
| `microduck_velstand_flat` PPO | `examples/velstand_getup/model_12498.pt` | Checked: gets up from face-down and face-up within 8 s |
| Sprint lineage (repo only) | six checkpoints, see below | Checked |

Sprint lineage, each resumed from the previous one:

| Example | Speed command | Measured speed | Head | Position targets |
|---|---|---|---|---|
| `sprint_speed_1p68` | fixed 2.2 m/s | 1.68 m/s | inverted most of the time | far past the stops |
| `sprint_head_upright` | fixed 2.2 m/s | 1.32 m/s | crown up, neck on its stop | far past the stops |
| `sprint_straight_long` | fixed 2.2 m/s | 0.83 m/s, 300 s straight | ~8 cm ahead of the trunk | far past the stops |
| `sprint_straight_fast` | fixed 2.2 m/s | 1.15 m/s, 300 s straight | ~8 cm ahead of the trunk | far past the stops |
| `sprint_straight_cmd` | 0.3–1.2 m/s | 0.31–1.12 m/s | ~8 cm ahead of the trunk | far past the stops |
| `sprint_straight_head` | 0.3–1.0 m/s | 0.30–0.93 m/s | over the trunk, ~3 cm ahead | clamped to the joint range |

The full comparison is in the README under "Sprint lineage", and each
example directory has its `metrics.json` and reproduce commands.

### Notable behavior

- `BamVoltageActionCfg.clip_target_to_joint_limits` clamps position targets
  to the soft joint limits, like a servo's goal-position limits. It is off by
  default, so the earlier checkpoints evaluate exactly as they were trained.
- `sprint_straight_head` was trained with that clamp on. Its eval, diagnose,
  and render commands need `--clip-targets`.
- Earlier sprint checkpoints command `hip_yaw` and `neck_pitch` targets far
  past their stops and do not survive with the clamp on. Treat
  `sprint_straight_head` as the starting point for hardware.

### Known limitations

- `microduck_velocity_flat` SAC finished training but falls repeatedly at
  inference; no checkpoint is published.
- `microduck_sprint_steerfix_flat` has a config but no published weights.
- `microduck_ground_pick_flat` and `microduck_sitstand_flat` are registered
  for MJWarp only. `mujoco-warp` is not in the default install, so the two
  MJWarp tests (`test_manager_api_minimal_slice` on these owners and
  `test_mjwarp_dr_smoke`) fail without it. All other tests pass.
- Sprint policies are checked on MuJoCo only; there is no Motrix owner for
  the sprint task.
