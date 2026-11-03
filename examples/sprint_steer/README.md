# Sprint steering example (forward speed + yaw rate)

Continuation of [`../sprint_straight_head/`](../sprint_straight_head/). Same
servo-clamped targets and head-over-trunk posture, and the policy now
follows a yaw-rate command as well as a forward-speed command: 0.3–1.0 m/s
forward, ±0.5 rad/s yaw (checked to ±0.4 rad/s).

Straight walking uses the same spawn-heading loop as the earlier examples,
acting as an external yaw controller: it writes the yaw command that holds
the spawn ray. Turning sets the yaw command directly with that loop off.

## Battery

512 environments, seed 777, targets clamped (`--clip-targets`).

**Turning** (`--yaw-rate`, loop off, 60 s). Measured yaw rate / survival:

| Forward command | −0.4 rad/s | −0.2 rad/s | +0.2 rad/s | +0.4 rad/s |
|---|---|---|---|---|
| 0.3 m/s | −0.41 / 100% | −0.20 / 100% | +0.20 / 100% | +0.39 / 100% |
| 0.5 m/s | −0.38 / 100% | −0.18 / 100% | +0.22 / 100% | +0.41 / 100% |
| 1.0 m/s | −0.35 / 97.7% | −0.17 / 98.4% | +0.20 / 96.7% | +0.40 / 97.3% |

Worst yaw-rate error is about 0.05 rad/s. Before this training,
`model_52250` overshot by 1.3–1.75× with the loop off, turned harder to the
right than to the left, and at 1.0 m/s survived a −0.4 rad/s turn only 63% of
the time.

**Straight, loop-held** (300 s) and **stress** (push 0.10 m/s, trunk CoM
10 mm, head CoM 6 mm, 2° reset tilt; 60 s):

| Forward command | Measured speed | 300 s survival | 300 s straight-pose pass | Stress survival |
|---|---:|---:|---:|---:|
| 0.3 m/s | 0.29 m/s | 100% | 100% | 99.6% |
| 0.5 m/s | 0.49 m/s | 100% | 100% | 100% |
| 0.75 m/s | 0.72 m/s | 100% | 100% | 99.8% |
| 1.0 m/s | 0.90 m/s | 93.4% | 92.0% | 96.5% |

The head stays 2–3 cm ahead of the trunk. Raw numbers:
[`metrics.json`](metrics.json).

Clips, all with the clamp on:

- [`play_video_turn_left_v0.5.mp4`](play_video_turn_left_v0.5.mp4) — 10 s side
  follow, 0.5 m/s with a +0.4 rad/s turn.
- [`play_video_turn_right_v1.0.mp4`](play_video_turn_right_v1.0.mp4) — 10 s
  side follow, 1.0 m/s with a −0.4 rad/s turn.
- [`play_video_circles.mp4`](play_video_circles.mp4) — 16 s overhead of eight
  robots, 0.5 m/s with a +0.4 rad/s turn, one full circle each.

## Known limits

- The loop-held straight line at 1.0 m/s over 300 s is about 3 points below
  `model_52250` (93.4% vs 96.5% survival, 92.0% vs 94.7% straight-pose pass).
  At 0.3–0.75 m/s straight and at every tested turn, this policy matches or
  beats it. For long straight runs at the top speed only,
  [`../sprint_straight_head/`](../sprint_straight_head/) is still slightly
  steadier.
- Turns were checked to ±0.4 rad/s; the training band reaches ±0.5 rad/s.
- Training pushed at up to 0.10 m/s, the same magnitude as the stress battery.
- Eval, render, and diagnose commands must pass `--clip-targets`.

## What changed

The actor contract (61D observation, 14D action) is unchanged; the yaw-rate
command was already in the observation.

1. **Steer** (`microduck_sprint_steer_flat`). The spawn-heading loop is off,
   yaw-rate commands are sampled in ±0.5 rad/s, and a narrow yaw-rate kernel
   (std 0.2, weight 6) joins the wide one. The spawn-ray costs (heading hold,
   heading error, cross-track) are dropped because they fight every turn. Yaw
   tracking became accurate and symmetric, but the loop-held 1.0 m/s straight
   line lost about 5 points of 300 s survival, mostly from single heading
   excursions past 25°. Retuning the external loop at eval time did not
   recover it.
2. **Mix loop-held episodes back in** (`microduck_sprint_steermix_flat`). 30%
   of episodes run the spawn-heading loop (`spawn_hold_fraction: 0.3`),
   exactly as the straight battery does. The spawn-ray costs come back with
   `gate_command: twist`, so they only score those episodes. That recovered
   most of the straight-line margin while keeping the turning.

## Train

```bash
TURNS="--turn-speeds 0.5,1.0 --turn-yaw-rates=-0.4,-0.2,0.2,0.4 --turn-duration 30"
SCORE="--eval-clip --eval-speeds 0.3,0.5,0.75,1.0 --eval-duration 60 --eval-envs 128 \
  --stress-speeds 0.3,1.0 --stress-duration 60 $TURNS \
  --max-speed-error 0.10 --max-yaw-error 0.10 --max-head-forward 0.05"

# 1. Steer from the head-over-trunk example.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_steer_flat \
  --load-run "$(pwd)/examples/sprint_straight_head" --checkpoint 52250 \
  --iterations 3000 --save-interval 500 --algo-log-name rsl_rl_ppo_steer_s1 $SCORE
# keep model_54000

# 2. Mix loop-held straight episodes back in.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_steermix_flat \
  --load-run <run-1> --checkpoint 54000 \
  --iterations 2000 --save-interval 500 --algo-log-name rsl_rl_ppo_steer_s2 $SCORE
# keep model_55999 (picked on 512 environments)
```

Negative yaw rates must be passed as `--turn-yaw-rates=-0.4,...` or
`--yaw-rate=-0.4`; a separate `-0.4` argument is read as an option name.

## Eval / play

```bash
# Turning, loop off
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_steer/model_55999.pt --clip-targets \
  --owner microduck_sprint_straightspeed_flat/mujoco \
  --speed 0.5 --yaw-rate 0.4 --duration 60 --num-envs 512 --seed 777

# Straight, loop-held
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_steer/model_55999.pt --clip-targets \
  --owner microduck_sprint_straightspeed_flat/mujoco \
  --speed 1.0 --duration 300 --num-envs 512 --seed 777

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_steer/model_55999.pt --clip-targets \
  --out-dir examples/sprint_steer --only follow --speed 0.5 --yaw-rate 0.4 \
  --follow-seconds 10 --stride 1 --follow-azimuth 90 --follow-elevation -12 \
  --follow-distance 1.0 --follow-name play_video_turn_left_v0.5.mp4
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_steer/model_55999.pt --clip-targets \
  --out-dir examples/sprint_steer --only radial --speed 0.5 --yaw-rate 0.4 \
  --radial-seconds 16 --radial-name play_video_circles.mp4
```
