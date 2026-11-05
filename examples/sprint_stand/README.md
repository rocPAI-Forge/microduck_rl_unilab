# Sprint standing example (0–1.0 m/s, yaw rate, stand still)

Continuation of [`../sprint_steer/`](../sprint_steer/). Same servo-clamped
targets, head-over-trunk posture, and yaw-rate steering, with the forward
command extended down to 0: the policy stands still at a zero command and
starts and stops as the command changes.

Before this training, `sprint_steer/model_55999` walked backward at a zero
command (−0.2 m/s) and fell in every one of 128 environments within 60 s.

This is the starting point for hardware. It passes the MotrixSim sim2sim
gate (`scripts/sim2sim_check.py`), and [`deploy/`](deploy/) holds its ONNX
export, `deploy_manifest.json`, and the closed-loop results of the reference
onboard runtime on MuJoCo and MotrixSim (`sim_loop_*.json`); see
[`docs/DEPLOY.md`](../../docs/DEPLOY.md). It has not run on a robot yet.

## Battery

512 environments, seed 777, targets clamped (`--clip-targets`). Straight is
loop-held for 300 s; stress is push 0.10 m/s, trunk CoM 10 mm, head CoM 6 mm,
2° reset tilt for 60 s.

| Forward command | Measured speed | 300 s survival | 300 s straight-pose pass | Stress survival |
|---|---:|---:|---:|---:|
| 0 (stand) | 0.000 m/s; 9 cm cross-track drift (p90) | 99.8% | 80.1% | 98.4% |
| 0.15 m/s | 0.11 m/s | 99.4% | 97.5% | — |
| 0.3 m/s | 0.29 m/s | 100% | 100% | 99.8% |
| 0.6 m/s | 0.57 m/s | 99.8% | 99.8% | — |
| 1.0 m/s | 0.88 m/s | 95.1% | 93.9% | 97.9% |

The 80% straight-pose pass at a standstill is heading drift, not falls or
travel: mean heading drifts 6.5° over 300 s and some environments pass 25°.

Turning, 60 s, measured yaw rate / survival:

| Forward command | −0.4 rad/s | −0.2 rad/s | +0.2 rad/s | +0.4 rad/s |
|---|---|---|---|---|
| 0 (in place) | −0.01 / 100% | — | — | +0.01 / 99.2% |
| 0.5 m/s | −0.40 / 100% | −0.22 / 100% | +0.17 / 100% | +0.37 / 100% |
| 1.0 m/s | −0.40 / 98.4% | −0.22 / 98.2% | +0.15 / 98.4% | +0.34 / 98.0% |

Raw numbers: [`metrics.json`](metrics.json). Clips, side view at 50 fps with
the clamp on:

- [`play_video_stand.mp4`](play_video_stand.mp4) — 8 s at a zero command;
  the trunk moves about 1 cm.
- [`play_video_side_v0.15.mp4`](play_video_side_v0.15.mp4) — 8 s at 0.15 m/s.

## Known limits

- **No turning in place.** A yaw command at zero forward speed measures about
  0 rad/s. The policy turns by stepping unevenly while walking and has no
  pivot. That is also why the spawn-heading loop cannot hold heading while
  standing. To turn around, command 0.2 m/s with ±0.4 rad/s: a 0.5 m radius
  turn that survives 100% over 60 s on MuJoCo and MotrixSim
  ([`docs/DEPLOY.md`](../../docs/DEPLOY.md#commands)).
- Left turns under-track by up to about 0.06 rad/s (+0.4 rad/s measures +0.34
  at 1.0 m/s); right turns are on target.
- A 0.15 m/s command runs 0.11 m/s, and a 1.0 m/s command 0.88 m/s.
- Training pushed at up to 0.10 m/s, the same magnitude as the stress battery.
- Eval, render, and diagnose commands must pass `--clip-targets`.

## What changed

1. **Extend the band to a standstill** (`microduck_sprint_stand_flat`).
   Forward commands are sampled in 0–1.0 m/s, 10% of resamples are a
   stand-still command, and the high-speed bias drops from 0.4 to 0.3.
   Commands resample every 6–12 s, so episodes include starts and stops. The
   gait-shaping terms already switch off below their command threshold, so a
   zero command does not pay for stepping in place. The robot learned to
   stand, but drifted 23° in heading over 300 s, ignored yaw commands at a
   standstill, and left turns slipped.
2. **Practice turning in place** (`microduck_sprint_standturn_flat`). 10% of
   resamples are turn-in-place commands. The robot did not learn to pivot,
   but standstill heading drift fell from 22.6° to 6.5°, the 1.0 m/s 300 s
   straight line rose from 93.0% to 95.1% survival, and left turns improved.
   That checkpoint is the one published here.

## Train

```bash
TURNS="--turn-yaw-rates=-0.4,0.4 --turn-duration 30"
SCORE="--eval-clip --eval-speeds 0.0,0.15,0.3,0.6,1.0 --eval-duration 60 --eval-envs 128 \
  --stress-speeds 0.0,1.0 --stress-duration 60 --max-yaw-error 0.10 --max-head-forward 0.05"

# 1. Stand still and walk slowly, from the steering example.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_stand_flat \
  --load-run "$(pwd)/examples/sprint_steer" --checkpoint 55999 \
  --iterations 2500 --save-interval 500 --algo-log-name rsl_rl_ppo_stand_t1 \
  $SCORE --turn-speeds 0.5,1.0 $TURNS --max-speed-error 0.10
# keep model_58498

# 2. Add turn-in-place commands.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_standturn_flat \
  --load-run <run-1> --checkpoint 58498 \
  --iterations 2000 --save-interval 500 --algo-log-name rsl_rl_ppo_stand_t2 \
  $SCORE --turn-speeds 0.0,0.5,1.0 $TURNS --max-speed-error 0.13
# keep model_59000 (picked on 512 environments)
```

Neither round passed its automatic gate: the 1.0 m/s command runs 0.88 m/s
(error 0.12) and the turn-in-place cases measure no turn. `model_59000` was
picked on the 512-environment comparison above.

## Eval / play

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_stand/model_59000.pt --clip-targets \
  --owner microduck_sprint_straightspeed_flat/mujoco \
  --speed 0.0 --duration 300 --num-envs 512 --seed 777

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_stand/model_59000.pt --clip-targets \
  --out-dir examples/sprint_stand --only follow --speed 0.0 \
  --follow-seconds 8 --stride 1 --follow-azimuth 90 --follow-elevation -8 \
  --follow-distance 0.9 --follow-name play_video_stand.mp4
```
