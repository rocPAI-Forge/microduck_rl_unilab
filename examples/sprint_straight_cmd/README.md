# Sprint speed-command straight example (0.3–1.2 m/s)

Continuation of [`../sprint_straight_fast/`](../sprint_straight_fast/). Same
upright head, alternating gait, and closed spawn-heading loop, but the
forward speed now follows the command instead of running about 1.15 m/s
whatever the command says.

Priority stays the same as the other straight examples: keep posture, then
run straight. On top of that, speed has to land on the command.

## Battery

512 environments, seed 777, closed loop on
`microduck_sprint_straightspeed_flat/mujoco` (clean) and
`microduck_sprint_straighthold_flat/mujoco --stress` (push 0.10 m/s, trunk CoM
10 mm, head CoM 6 mm, 2° reset tilt):

| Command | Measured speed | Clean 300 s survival | Clean 300 s straight-pose pass | Stress 60 s survival |
|---|---:|---:|---:|---:|
| 0.3 m/s | 0.305 m/s | 98.4% | 97.5% | 95.3% |
| 0.6 m/s | 0.580 m/s | 99.6% | 99.6% | 99.6% |
| 0.9 m/s | 0.878 m/s | 99.4% | 96.9% | 99.6% |
| 1.2 m/s | 1.120 m/s | 94.7% | 90.0% | 95.1% |

For comparison, `sprint_straight_fast/model_27000` runs 1.10–1.14 m/s at
every one of these commands.

Mean heading error stays at 1.6–2.7° and the head is inverted for at most
0.4% of the time at any speed. Speed is set mostly by stride length: foot
scissor grows from 0.13 m at 0.6 m/s to 0.18 m at 1.2 m/s, while cadence stays
near 3.4–3.6 strides per second. Thigh and foot sign-flip stay at 0.47–0.48.

Raw numbers: [`metrics.json`](metrics.json). Clips:

- [`play_video_side_v0.3.mp4`](play_video_side_v0.3.mp4),
  [`play_video_side_v0.6.mp4`](play_video_side_v0.6.mp4),
  [`play_video_side_v0.9.mp4`](play_video_side_v0.9.mp4),
  [`play_video_side_v1.2.mp4`](play_video_side_v1.2.mp4) — 8 s close side view
  at full 50 fps for each command, from a standing start.
- [`play_video_radial.mp4`](play_video_radial.mp4) — 7 s far overhead of eight
  robots at a 1.2 m/s command on headings 0°, 45°, …, 315°.

## Known limits

- The top of the band is slightly under-tracked: a 1.2 m/s command runs about
  1.12 m/s.
- At 1.2 m/s the 300 s straight-pose pass is about 90%, below
  `model_27000`'s 93–99% at the same speed. Controllability cost some
  long-horizon margin at the top speed.
- Every 0.3 m/s fall, clean or stressed, happens within the first 6 s after a
  standing start. Once walking, the slow gait holds.
- Training pushed at up to 0.10 m/s, the same magnitude as the stress
  battery, so the stress numbers are held out for CoM and tilt but not for
  push size.

## What changed

The actor contract (61D observation, 14D action) is unchanged; the forward
command was already in the observation.

1. **Reachable commands, tracking reward.** `microduck_sprint_straightcmd_flat`
   samples 0.3–1.2 m/s, resamples every 6–12 s, drops the uncapped forward and
   along-ray progress rewards, and pays for a wide (std 0.6, weight 4) plus a
   narrow (std 0.25, weight 10) lin-vel kernel. At weights 2 / 4 with a 0.2
   narrow kernel, the cheaper-to-hold slow gait won and the whole response
   collapsed toward the middle of the band.
2. **Keep the ray across resamples.** Timed command resamples used to recapture
   the spawn heading and origin, so the yaw loop would hold a new ray while the
   heading and cross-track costs still measured the old one. Only an episode
   reset recaptures now.
3. **Keep the top of the band in practice.** With uniform sampling the 1.2 m/s
   gait forgot its long-horizon robustness (300 s survival 80.5%,
   straight-pose pass 69%). `high_speed_fraction: 0.4` draws 40% of resamples
   from 0.9–1.2 m/s.
4. **Practice the slow start under perturbation.** Reset tilt up to 2.5° and
   30 s episodes (twice the standing starts) halved 0.3 m/s stress falls on
   512 environments (46 → 24).

## Train

```bash
# 1. Speed command: resume the fast straight example.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straightcmd_flat \
  --load-run "$(pwd)/examples/sprint_straight_fast" --checkpoint 27000 \
  --iterations 3000 --save-interval 250 \
  --algo-log-name rsl_rl_ppo_straightcmd_c2 \
  --eval-speeds 0.3,0.6,0.9,1.2 --eval-duration 60 --eval-envs 128 \
  --max-speed-error 0.10
# keep model_29999

# 2. Bias resamples toward the top of the band.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straightcmd_flat \
  --load-run <run-1> --checkpoint 29999 \
  --iterations 2000 --save-interval 250 \
  --algo-log-name rsl_rl_ppo_straightcmd_c3 \
  --eval-speeds 0.3,0.6,0.9,1.2 --eval-duration 120 --eval-envs 128 \
  --max-speed-error 0.10 \
  --override +env.commands.twist.high_speed_fraction=0.4 \
  --override '+env.commands.twist.high_speed_range=[0.9,1.2]'
# keep model_31500

# 3. Slow standing starts under larger reset tilt.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straightcmd_flat \
  --load-run <run-2> --checkpoint 31500 \
  --iterations 1500 --save-interval 250 \
  --algo-log-name rsl_rl_ppo_straightcmd_c4 \
  --eval-speeds 0.3,0.6,0.9,1.2 --eval-duration 120 --eval-envs 128 \
  --max-speed-error 0.10 --stress-speeds 0.3,1.2 --stress-duration 60 \
  --override +env.commands.twist.high_speed_fraction=0.4 \
  --override '+env.commands.twist.high_speed_range=[0.9,1.2]' \
  --override 'env.events.reset_base.params.pose_range.roll=[-0.0436,0.0436]' \
  --override 'env.events.reset_base.params.pose_range.pitch=[-0.0436,0.0436]' \
  --override env.max_episode_seconds=30.0
# keep model_32250 (picked on 512 environments; 128-environment scores sit
# within noise of each other)
```

## Eval / play

```bash
for v in 0.3 0.6 0.9 1.2; do
  uv run --no-sync scripts/eval_sprint_speed.py \
    examples/sprint_straight_cmd/model_32250.pt \
    --owner microduck_sprint_straightspeed_flat/mujoco \
    --speed $v --duration 300 --num-envs 512 --seed 777
  uv run --no-sync scripts/eval_sprint_speed.py \
    examples/sprint_straight_cmd/model_32250.pt \
    --owner microduck_sprint_straighthold_flat/mujoco --stress \
    --speed $v --duration 60 --num-envs 512 --seed 777
done

for v in 0.3 0.6 0.9 1.2; do
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
    --checkpoint examples/sprint_straight_cmd/model_32250.pt \
    --out-dir examples/sprint_straight_cmd --only follow --speed $v \
    --follow-seconds 8 --stride 1 --follow-azimuth 90 --follow-elevation -8 \
    --follow-distance 0.9 --follow-name play_video_side_v$v.mp4
done
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_straight_cmd/model_32250.pt \
  --out-dir examples/sprint_straight_cmd --only radial --speed 1.2 --radial-seconds 7
```
