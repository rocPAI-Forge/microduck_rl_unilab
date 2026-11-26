# Changelog

Versions follow `project.version` in `pyproject.toml`. Each release is a git
tag `v<version>` with a matching GitHub release. Checkpoints live in
`examples/` and are versioned with the code that reproduces them.

## Unreleased

- Checkpoints published for the four 0.4.0 motions, each as an example with
  weights, a 15 s clip, `metrics.json` and reproduce commands, re-evaluated on
  three seeds: `examples/stilts/` (10 cm stilts, 99.2–100% over 60 s),
  `examples/sitstand/` (sits 85–89%, stands 97%), `examples/scoot/` (backs up
  at 82–88% of the command), and `examples/basketball/` (partial: 98–99% for
  60 s under small commands, 3.5% under the playground protocol). The README
  clip table and status table link them; the studies point at them.
- Fix: `scripts/eval_walk.py` now seeds the environment as well as the
  algorithm, so `--seed` changes the randomization (before, every seed gave
  the same environments).

## 0.5.0

Three more motions, all in simulation on MuJoCo, and this time with published
weights: the roulade (upstream microduck_rl's forward roll over the head,
landing on the feet), a fast forward tumble that ends standing, and ball
dribbling on the `sprint_stand` gait. The README gains a clip table for every
motion added in 0.4.0 and after. None of the new checkpoints has a MotrixSim
check or a robot run; `sprint_stand/model_59000` stays the hardware starting
point. Dependencies are unchanged.

### Known limitations

- No turning in place: four rounds found a left pivot only. Turn around with
  the 0.5 m radius turn.
- 1.0 m/s on MotrixSim and under ±1° servo backlash loses margin; wider
  randomization or backlash training trades it for speed or for standing on a
  stiff robot. Keep first hardware runs at or below 0.6 m/s.
- On Pollen's `robotd`, set the `[policy]` values in `docs/DEPLOY.md`, and
  check the target clamp, which `robotd` does not apply. Nothing has run on a
  robot yet.
- The roulade checkpoint is upstream's published policy fine-tuned here; our
  own training never rolled over the head from standing. It plants the head on
  the floor, and the tumble peaks near 10 rad/s: both are simulation demos.
- Ball "kicking" is a dribble: only the foot soles collide with the ball in
  the walking model, so the duck cannot strike it.
- The 0.4.0 limitations of stilts, sit/stand, scooting and basketball still
  hold, and walking-task foot friction below 1.0 is still roadmap item 12.

### Changes

- Docs checked for consistency before the release: the roadmap status names
  0.5.0; the READMEs point at `v0.5.0`.
- Forward roll, two policies (`docs/audits/roulade.md`), simulation only:
  - Roulade (`examples/roulade/model_999.pt`): upstream microduck_rl's roll
    over the flat top of the head, landing on the feet, in one policy.
    `roulade_terms` ports upstream's design (support- and flatness-gated
    rotation accumulator, head-top latch, capped progress, completion-gated
    landing, mid-roll spawns) as `microduck_roulade_flat`, with head and
    whole-robot floor contact sensors in `roulade_task.xml`. Our own training
    (standup warm start, and upstream's recipe from scratch to 6000
    iterations) never rolled over the head from standing; upstream's published
    `roulade.onnx` does in our simulator, so the checkpoint is that policy
    converted with the new `scripts/onnx_to_rsl_rl.py` and fine-tuned 1000
    iterations: 100% complete in 0.86 s and 100% standing, rolling once.
  - Tumble (`examples/tumble/model_17997.pt`): a fast forward tumble that
    ends standing (`microduck_roll_flat`, then `microduck_tumble_flat` with
    completion-gated stand-up terms): 100% full turn in 0.56 s, 100% standing.
  - `scripts/eval_roll.py` scores either from standing starts.
- Ball dribbling on the `sprint_stand` gait (`docs/audits/kick.md`,
  `examples/kick/model_60499.pt`): `kick_terms` (`BallPursuitCommand` walks the
  duck behind the ball and through it along a goal direction; ball and goal in
  the head-pose command slot; ball-velocity rewards), `microduck_kick_flat`,
  a 3.5 cm ball scene, and `scripts/eval_kick.py`. 99.6–100% survival and the
  ball pushed a median 1.0–1.4 m along the goal in 12 s; it dribbles rather
  than strikes, since only the foot soles collide in the walking model.
- README: a clip table for the motions added in 0.4.0 and after.

## 0.4.0

Four new motion tasks after upstream microduck_rl and Vottivott's
microduck-playground, all in simulation on MuJoCo: walking on 10 cm stilts,
commanded sit and stand, seated scooting, and balancing on a free basketball.
No new checkpoint is published (none of the four has a MotrixSim gate), and
`sprint_stand/model_59000` stays the hardware starting point. Dependencies
are unchanged.

### Known limitations

- No turning in place: four rounds found a left pivot only. Turn around with
  the 0.5 m radius turn.
- 1.0 m/s on MotrixSim and under ±1° servo backlash loses margin; wider
  randomization or backlash training trades it for speed or for standing on a
  stiff robot. Keep first hardware runs at or below 0.6 m/s.
- On Pollen's `robotd`, set the `[policy]` values in `docs/DEPLOY.md`, and
  check the target clamp, which `robotd` does not apply. Nothing has run on a
  robot yet.
- The new tasks are simulation results only. Stilts steer with a right bias;
  sit/stand drops the last ~4 cm onto the seat; scooting turns and moves
  forward weakly; basketball collapses under large switching commands
  (3.5% survival at 60 s under the playground protocol).
- Foot friction randomization below 1.0 never took effect on MuJoCo for the
  walking tasks (roadmap item 12); only scoot uses the fix.

### Changes

- Docs checked for consistency before the release: the roadmap status names
  0.4.0 and lists item 12; the READMEs list the scoot and basketball owners
  and their studies.
- Walking on stilts, after Vottivott/microduck-playground:
  `scripts/make_stilt_robot.py` writes the robot with procedural stilts
  (height, and a blend from platform to peg), `microduck_stilt_flat/mujoco`
  is the task, `scripts/train_stilts.py` runs the support-then-height
  curriculum, and `scripts/eval_walk.py` scores any walking owner at a fixed
  command. From the walking policy to 10 cm stilts, every stage held 100% at
  0.15 m/s; the 10 cm policy walks straight at 99–100% over 60 s but steers
  with a right bias (`docs/audits/stilts.md`). No checkpoint is published.
- Commanded sit and stand on MuJoCo, after upstream microduck_rl's sitstand
  recipe: `sitstand_terms` (posture rewards that track the
  `SitStandCommand`'s slewed target, rise and sit bootstraps, speed caps),
  the `microduck_sitstand_flat/mujoco` owner on the standup base, the
  `microduck_sitstand_warm_flat` and `microduck_sitstand_fix_flat` variants,
  `MicroduckSitStandFlat` registered on MuJoCo, and `scripts/eval_sitstand.py`.
  After five rounds one policy sits 85–89% and stands 97% on command,
  upright, with ≤ 0.2% falls, but drops the last ~4 cm onto the seat;
  two gentler-descent variants (`microduck_sitstand_gentle_flat`,
  `microduck_sitstand_ramp_flat`) made it stop sitting instead
  (`docs/audits/sitstand.md`). No checkpoint is published.
- Seated scooting ("butt walk") on MuJoCo, after microduck-playground:
  `scoot_terms` (command progress, yaw-rate L1, stillness at zero command,
  head and height costs), the `microduck_scoot_flat`,
  `microduck_scoot_straight_flat` and `microduck_scoot_sym_flat` owners on the
  ground-contact standup base. Round 4 backs up at 56–80% of the command with
  100% survival; turning reaches 0.03–0.10 of 0.3 rad/s
  (`docs/audits/scoot.md`). No checkpoint is published.
- `scripts/make_hull_robot.py` caps collision meshes at 128 hull vertices so
  ground-contact tasks fit 4096 envs (29.7 MB to 0.65 MB per env clone), and
  writes a contact-priority variant that lets robot friction govern floor
  contacts.
- Recorded that friction randomization below the floor's 1.0 never took
  effect on MuJoCo (`docs/audits/friction_combination.md`); fixed for scoot
  only, the walking tasks are roadmap item 12.
- Basketball balance on MuJoCo, after microduck-playground's b11: a free
  ball entity and `BallHoldAction` hold curriculum (`basketball_terms`),
  `scripts/make_basketball_scene.py`, a blind LSTM actor, the
  `microduck_basketball_*` owners (rounds 1–13), `lin_vel_error_l1` in
  `scoot_terms`, and `scripts/eval_basketball.py` /
  `scripts/eval_basketball_playground.py`. Partial result: round 10 balances
  98–99% at 60 s under small fixed commands and follows them weakly, but
  under the playground protocol (random ±0.30 m/s commands, frequent pushes)
  survives 3.5% at 60 s, against b11's reported 97%
  (`docs/audits/basketball.md`). No checkpoint is published.

## 0.3.1

Studies and deployment groundwork after 0.3.0; no new checkpoint, and
`sprint_stand/model_59000` stays the hardware starting point. The MicroDuck
meshes are now documented as CC BY-NC-SA 4.0, as upstream licenses them.
Dependencies are unchanged.

### Known limitations

- No turning in place: four rounds found a left pivot only. Turn around with
  the 0.5 m radius turn.
- 1.0 m/s on MotrixSim and under ±1° servo backlash loses margin; wider
  randomization or backlash training trades it for speed or for standing on a
  stiff robot. Keep first hardware runs at or below 0.6 m/s.
- On Pollen's `robotd`, set the `[policy]` values in `docs/DEPLOY.md`, and
  check the target clamp, which `robotd` does not apply. Nothing has run on a
  robot yet.

### Changes

- Docs checked for consistency before the release: the roadmap gains item 11
  (backlash) and records the fourth pivot round and the `robotd` settings
  under item 5; `docs/DEPLOY.md`, the READMEs, and
  `examples/sprint_stand/README.md` point at the backlash study and `robotd`.
- The 47 MicroDuck STL meshes are licensed CC BY-NC-SA 4.0, following
  upstream `LICENSE-MODELS.md` (commit `7cc1961`): no commercial use, share
  alike, credit Pollen Robotics. `assets/robots/microduck/LICENSE-MODELS.pollen-robotics.md`
  scopes it; the code stays Apache-2.0. The READMEs gain a License section, and
  the meshes are recorded as the upstream export `a8321d2`.
- `sprint_terms.track_yaw_rate` is a yaw-only Gaussian.
  `unilab.envs.mdp.track_angular_velocity` folds the roll and pitch rates into
  its exponent, so its narrow copy in the sprint owners worked against
  stepping in place, the dead zone upstream microduck_rl found in `e1d518d`.
  Owner `microduck_sprint_pivotyaw_flat` uses it. A fourth pivot round still
  found a left pivot only (+0.22 / −0.02 rad/s for ±0.4) but kept the slow
  turns; `docs/audits/turn_in_place.md` is corrected and extended. No
  checkpoint is published.
- `scripts/add_backlash.py` (ported from upstream microduck_rl) writes a
  backlash robot with a passive ±1° hinge after every servo joint, with the
  `microduck_sprint_straightspeed_flat/mujoco_backlash` eval owner and the
  `microduck_sprint_standbacklash_flat` training owner. On it `model_59000`
  loses 3.5–4 points of 60 s survival at 0–0.6 m/s and 22 at 1.0 m/s.
  Training on it recovers that but stands at 89.5% on MotrixSim, below the
  gate, so no checkpoint is published (`docs/audits/backlash.md`).
- `docs/DEPLOY.md` "Running on Pollen's robotd": the official daemon builds
  the same 61-D observation, so the exported ONNX loads unchanged, but its
  default action scale 0.9 and head/leg low-pass slow every gait by 7–15%; the
  section gives the `[policy]` settings that run it as trained. The runtime
  gains `TargetShaping` / `ROBOTD_DEFAULTS`, and `scripts/deploy_sim_loop.py`
  gains `--shaping robotd` and an optional `straight_1.0` scenario; its target
  check now skips rows reset that step.
- Top speed on MotrixSim is parked. Owners `microduck_sprint_standwide_flat`
  (wider armature, servo delay, servo friction, and foot friction) and
  `microduck_sprint_standwidefast_flat` (plus heavier top-speed tracking)
  raised 1.0 m/s 60 s survival on Motrix from 85% to 92–94% only by slowing
  the gait to 0.76–0.82 m/s
  ([`docs/audits/motrix_topspeed.md`](docs/audits/motrix_topspeed.md)). No
  checkpoint is published.
- `eval_sprint_speed.py --override` applies extra Hydra overrides after the
  owner, for parameter sweeps.
- Roadmap item 7 is settled: velocity SAC stays documented as failed
  validation, `microduck_sprint_steerfix_flat` stays as the first stage of the
  `sprint_straight_long` reproduce chain, and the MJWarp-only tasks wait on an
  NVIDIA machine.
- Docs aligned with the state after 0.3.0: README and README.zh (runtime,
  turn-around, top-speed study, layout, clone URL now
  `rocPAI-Forge/microduck_rl_unilab`, MJWarp test note),
  `examples/sprint_stand/README.md` (deploy folder and gate), `docs/DEPLOY.md`,
  and `docs/ROADMAP.md` (work stops until hardware is available).

## 0.3.0

Prepares the standing sprint for a first hardware run: a reference onboard
runtime checked closed-loop on both simulators, fixed simulator roles with a
MotrixSim release gate, and a documented way to turn around. No new
checkpoint; `sprint_stand/model_59000` stays the hardware starting point and
passes the Motrix gate. Dependencies are unchanged from 0.2.0.

### Known limitations

- No turning in place; turn around with the 0.5 m radius turn.
- 1.0 m/s does not transfer to MotrixSim (67.4% over 300 s); roadmap item 10.
- The robot side of deployment (IMU and servo conventions, calibration, first
  run) is untested.

### Changes

- `microduck_rl_unilab.deploy` is a reference onboard runtime (numpy and
  onnxruntime only): observation builder, ONNX runner with the goal-position
  clamp, the training heading hold, and stand / walk / straight / turn-around
  commands. `scripts/deploy_sim_loop.py` runs it closed-loop on MuJoCo and
  MotrixSim; for `sprint_stand` the observation matches the training contract
  to 1e-6 and every scenario survives 100%. See `docs/DEPLOY.md`
  "Reference runtime".
- Simulator roles are fixed: sprint policies train on MuJoCo only, and
  MotrixSim is the held-out sim2sim check. `scripts/sim2sim_check.py` runs the
  Motrix gate (300 s straight at 0, 0.3, and 0.6 m/s; 60 s turns at 0.5 m/s
  and ±0.4 rad/s; at least 95% survival and at most 0.10 rad/s yaw error) and
  reports 1.0 m/s without gating it. A checkpoint is published only after it
  passes. See the README section "Simulators".
- Turning in place is parked. `MicroduckVelocityCommandCfg.turn_in_place_lin_max`
  and `turn_in_place_curriculum` bridge turn commands from walking turns down
  to a pivot (both off by default), with owners `microduck_sprint_pivot_flat`,
  `microduck_sprint_pivothold_flat`, and `microduck_sprint_pivotsym_flat`
  (mirror augmentation). Three rounds did not teach a pivot
  ([`docs/audits/turn_in_place.md`](docs/audits/turn_in_place.md)), so
  `docs/DEPLOY.md` documents a 0.5 m radius turn (`[0.2, 0, ±0.4]`) instead:
  100% survival over 60 s on MuJoCo and MotrixSim.

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
