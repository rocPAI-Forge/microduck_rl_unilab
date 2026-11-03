# Roadmap after 0.1.0

The policy closest to hardware is
[`sprint_straight_head`](../examples/sprint_straight_head/): it was trained
with servo-clamped position targets and keeps the head over the trunk. It
only walks forward, its speed command starts at 0.3 m/s, and it has been
checked on MuJoCo only. The items below close those gaps first, then
validate for hardware, then clean up infrastructure.

Each item lists what "done" means. Training rounds use 2048 environments:
on the current 8-core, 59 GB machine, 2048 environments already take about
26 GB, 3072 ran slower (27.3k vs 29.8k steps/s), and 4096 ran out of memory.
Decisions between close checkpoints use 512-environment batteries.

## Priority 1: complete the motion interface (on `sprint_straight_head`)

- [ ] **1. Steering.** Turn off the spawn-heading loop and train the policy
  to track a yaw-rate command of about ±0.5 rad/s, keeping the forward-speed
  command and the target clamp.
  - Done when: left and right yaw-rate tracking error is small across the
    speed band, and straight-line survival, straight-pose pass, and head
    posture do not regress against `model_52250`.
  - Estimate: 2–4 rounds, about 4–8 hours of training.
  - `microduck_sprint_steerfix_flat` was written for the unclamped gait; this
    item replaces it rather than reusing its weights.
- [ ] **2. Standing and low speed.** Extend the forward command from
  0.3–1.0 m/s down to 0, so the policy stands still at a zero command and can
  start and stop.
  - Done when: 0–1.0 m/s is tracked across the band, and a zero command holds
    position without drift.
  - Estimate: 1–2 rounds, about 2–4 hours.

Together these give one policy with forward speed and yaw rate both
commandable, servo-clamped targets, and a normal head posture.

## Priority 2: validate for hardware

- [ ] **3. Cross-simulator check.** Install `unilab[motrix]`, add a Motrix
  owner for the sprint task, and run the same batteries on `model_52250` to
  confirm the policy is not tuned to MuJoCo alone.
  - Estimate: half a day.
- [ ] **4. Position-target audit of the other tasks.** Check whether the walk,
  stand-up, and get-up policies also command targets far past the joint
  stops. The walk policy (`velocity_flat_mujoco/model_499`) does not; stand-up
  and get-up have not been checked. Retrain any that do under the clamp.
  - Estimate: about 1 hour to check; retraining is extra.
  - Do this first: its result can change the order of everything else.
- [ ] **5. Deployment path.** Export ONNX, check observation and action
  dimensions and order against `deploy_contract.py`, and write down the known
  sim-to-real gaps (observation noise, delay, backlash) already listed in
  `sim2real_notes.py`.
  - Estimate: half a day to a day.
  - Timing depends on when a MicroDuck is available for testing.

## Priority 3: infrastructure and leftovers

- [ ] **6. MJWarp (GPU physics).** About 83% of each training iteration is
  CPU physics. Moving it to the GPU needs `mujoco-warp` installed and the BAM
  per-substep torque path validated on MJWarp.
- [ ] **7. Leftovers.**
  - Velocity SAC trains but falls repeatedly at inference.
  - `microduck_sprint_steerfix_flat` has no weights (superseded by item 1).
  - `microduck_ground_pick_flat` and `microduck_sitstand_flat` are
    MJWarp-only (depends on item 6).

## Suggested order

1. Item 4 check (about 1 hour).
2. Item 1, steering.
3. Item 2, standing and low speed.
4. Item 3 in the gaps between training runs.
5. Item 5 once hardware timing is known.

## Open questions

- Is a MicroDuck available for testing, and when? This decides whether item 5
  moves up.
- Should MJWarp (item 6) move ahead of the motion items to shorten every
  later training round?
