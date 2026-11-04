# Roadmap after 0.1.0

The policy closest to hardware at 0.1.0 was
[`sprint_straight_head`](../examples/sprint_straight_head/): servo-clamped
position targets and the head over the trunk, but forward-only, with a speed
command that started at 0.3 m/s, checked on MuJoCo only. The items below close
those gaps first, then validate for hardware, then clean up infrastructure.

Each item lists what "done" means. Training rounds use 2048 environments:
on the current 8-core, 59 GB machine, 2048 environments already take about
26 GB, 3072 ran slower (27.3k vs 29.8k steps/s), and 4096 ran out of memory.
Decisions between close checkpoints use 512-environment batteries.

## Status

| Item | State | PR |
|---|---|---|
| 1. Steering | Merged | #7 |
| 2. Standing and low speed | Merged | #8 |
| 3. Cross-simulator check | Merged | #9 |
| 4. Position-target audit | Merged | #6 |
| 5. Deployment path | Software side merged; hardware side open | #10 |
| 6. MJWarp | Blocked: needs an NVIDIA GPU | — |
| 7. Leftovers | Open | — |
| 8. Turning in place (new) | In progress: left pivot partly learned, right not | `feat/turn-in-place` |
| 9. Motrix in the training loop (new) | Closed: train on MuJoCo only, Motrix is the held-out gate | #11 (record), #13 (rule) |

**Simulator rule.** Sprint policies train on MuJoCo only. MotrixSim is held
out for the sim2sim check (`scripts/sim2sim_check.py`), and a checkpoint is
published only after it passes. Failures are fixed on the MuJoCo side.

## Priority 1: complete the motion interface

- [x] **1. Steering** (#7). `sprint_steer/model_55999` tracks ±0.2 and
  ±0.4 rad/s yaw within about 0.05 rad/s at 0.3–1.0 m/s, with 97–100%
  survival while turning. Turning was learned with the spawn-heading loop off
  (`microduck_sprint_steer_flat`); mixing 30% loop-held episodes back in, with
  the spawn-ray costs gated to those episodes (`microduck_sprint_steermix_flat`),
  recovered most of the 1.0 m/s straight-line margin.
- [x] **2. Standing and low speed** (#8). `sprint_stand/model_59000` holds a
  zero command for 300 s at 99.8% survival with about 9 cm of drift and walks
  0.15–1.0 m/s, keeping the steering. A 1.0 m/s command runs 0.88 m/s. It is
  now the recommended starting point for hardware.
- [ ] **8. Turning in place (new).** `model_59000` measures about 0 rad/s for
  a yaw command at zero forward speed, so the heading loop cannot hold heading
  while standing (6.5° mean drift over 300 s on MuJoCo). Ten percent
  turn-in-place commands for 2000 iterations did not teach a pivot. Next try:
  a larger share and a pivot-specific reward, or a separate stand-and-turn
  stage.

## Priority 2: validate for hardware

- [x] **3. Cross-simulator check** (#9). The clamp-trained checkpoints
  transfer to MotrixSim at 0–0.6 m/s (98.6–100% over 300 s), but not at
  1.0 m/s (57–80%, against 93–97% on MuJoCo). Cap hardware commands near
  0.6 m/s at first.
- [x] **9. Motrix in the training loop (new), closed.** Motrix-only
  fine-tuning moved the overfit to Motrix (MuJoCo 1.0 m/s survival fell to
  7.4%), and alternating held 84–88% on both but spends the only held-out
  simulator (#11, kept as a record). Decision (#13): train on MuJoCo only and
  gate releases on Motrix over 0–0.6 m/s; 1.0 m/s is reported, not gated.
- [x] **4. Position-target audit** (#6). Only the sprint lineage leans on the
  joint stops; the walk, stand-up, and get-up checkpoints finish their tasks
  with the goal-position clamp on. None needs retraining.
- [ ] **5. Deployment path** (#10 for the software side).
  - [x] ONNX export with an onnxruntime check: largest action difference
    2.4e-6 over 8000 observations.
  - [x] `deploy_manifest.json` and `docs/DEPLOY.md`: observation and action
    order, units, clamp, rate, commands, training noise and delay, known
    limits.
  - [ ] Calibrate the IMU and servo noise and delay envelopes on hardware.
  - [ ] First on-robot run.

## Priority 3: infrastructure and leftovers

- [ ] **6. MJWarp (GPU physics).** Blocked on this machine. `warp-lang`
  1.19 sees only the CPU next to the Radeon AI PRO R9700; Warp's GPU backend
  is CUDA-only. Both MJWarp tests require "an active CUDA Warp device". This
  needs an NVIDIA machine.
- [ ] **7. Leftovers.**
  - Velocity SAC trains but falls repeatedly at inference.
  - `microduck_sprint_steerfix_flat` has no weights (superseded by item 1).
  - `microduck_ground_pick_flat` and `microduck_sitstand_flat` are
    MJWarp-only (depends on item 6).

## Open questions

- Is a MicroDuck available for testing, and when? Item 5's hardware side
  waits on it.
- Next training item: turning in place (8) or Motrix in the loop (9)?
