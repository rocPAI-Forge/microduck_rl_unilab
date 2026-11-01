# Sprint fast straight example (1.15 m/s)

Continuation of [`../sprint_straight_long/`](../sprint_straight_long/). Same
upright head, alternating gait, and closed spawn-heading loop, about 39%
faster along the spawn ray, and steadier under pushes.

Priority stays the same as the long-horizon example: keep posture, then run
straight. Speed only counts when the 300 s straight-pose pass holds.

## Battery vs the long-horizon example

300 s, 128 environments, 2.2 m/s command, closed loop on
`microduck_sprint_straightspeed_flat/mujoco`:

| Metric | Long `19297` (seed 123 / 456) | This `27000` (seed 42 / 123 / 456) |
|---|---|---|
| Body-forward speed | 0.83 m/s | **1.15 m/s** |
| Spawn-forward distance | 247.9 / 247.4 m | **347 / 350 / 346 m** |
| Survival | 100% / 100% | 99.2% / 100% / 98.4% |
| Mean \|heading\| | 1.71° / 1.74° | 1.74° / 1.69° / 1.81° |
| Straight-pose pass | 95.3% / 93.8% | 93.8% / 99.2% / 93.0% |
| Head inverted | 0 | ≤ 0.02% of time |

600 s dual-seed check: survival 96.9% / 95.3%, mean heading 1.82° / 1.75°,
straight-pose pass 94.5% / 94.5% (long example: 87.5% / 79.7%).

Stress battery (push 0.10 m/s, trunk CoM 10 mm, head CoM 6 mm, 2° reset tilt,
60 s, seed 123 / 456):

| Metric | Long `19297` | This `27000` |
|---|---|---|
| Survival | 93.0% / 93.8% | **96.1% / 96.9%** |
| Straight-pose pass | 91.4% / 93.0% | 95.3% / 92.2% |
| Speed | 0.81 / 0.84 m/s | 1.13 / 1.14 m/s |

The last training stage pushed at up to 0.10 m/s, so the stress battery is
held out for CoM and tilt size but not for push size.

Gait: thigh and foot sign-flip 0.48, foot scissor 0.17 m peak-to-peak, head
yaw and roll within 0.3°.

Repeating a 300 s battery on the same seed moves the straight-pose pass by
about 4 points, so read single-seed gaps below that as noise.

Raw numbers: [`metrics.json`](metrics.json). Clips at a 2.2 m/s command:

- [`play_video_side.mp4`](play_video_side.mp4) — 10 s close side view at
  full 50 fps (11.1 m covered). The thighs trade places at about 3.4 strides
  per second with short flight phases. The crown stays up, but the neck holds
  the head well forward of the trunk.
- [`play_video_follow.mp4`](play_video_follow.mp4) — 8 s close rear-quarter follow.
- [`play_video_radial.mp4`](play_video_radial.mp4) — 7 s far overhead of eight
  robots leaving the origin on headings 0°, 45°, …, 315° (shorter than the
  long example's 10 s so the faster robots stay in frame).

## What changed

The actor contract (61D observation, 14D action) is unchanged.

1. **A speed gradient that reaches 0.83 m/s.** The default
   `tracking_lin_vel` kernel (std 0.39) is flat at 0.83 m/s against a
   1.65–2.2 m/s command, and raising the along-ray reward alone did not move
   speed. Widening the kernel to std 1.0 did. Past about 1.25 m/s the gait
   trades heading for speed, so the checkpoint is taken before that.
2. **Pin the line.** With speed kept, pin the along-ray, heading, cross-track,
   and straightness weights at their converged values and train on 60 s
   episodes (`microduck_sprint_straighthold_flat`).
3. **Price the rare event.** Most 300 s failures were not mean heading but a
   single stumble. A fall penalty and interval pushes cut falls; a
   `head_tilt_excess_cost` from 60° to 95° head tilt cut the brief head flips
   that ended most straight-pose runs.
4. **Train at stress-level perturbation.** Pushes of 0.05 and 0.08 m/s did not
   change stress survival. Raising pushes to 0.10 m/s together with wider
   trunk/head CoM and reset tilt did, without losing clean-run speed.

## Train

The whole chain runs through `scripts/train_sprint_straight_speed.py`, which
scores every checkpoint on the same push-free closed-loop battery:

```bash
# 1. Speed: resume the long example with a wide lin-vel kernel.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --algo-log-name rsl_rl_ppo_straightspeed_r2 \
  --override reward.tracking_lin_vel.params.std=1.0 \
  --override reward.tracking_lin_vel.weight=3.0 \
  --override algo.algorithm.learning_rate=2.0e-4 \
  --override algo.algorithm.entropy_coef=0.01
# keep model_21000

# 2. Hold the line at that speed.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --load-run <run-1> --checkpoint 21000 --iterations 1500 \
  --algo-log-name rsl_rl_ppo_straightspeed_r3 --eval-duration 120 \
  --override reward.tracking_lin_vel.params.std=1.0 \
  --override reward.tracking_lin_vel.weight=1.0 \
  --override env.curriculum.spawn_progress_weight=null \
  --override env.curriculum.heading_abs_weight=null \
  --override reward.spawn_progress.weight=5.0 \
  --override reward.heading_abs.weight=-0.75 \
  --override reward.cross_track.weight=-0.8 \
  --override reward.straightness.weight=-0.5 \
  --override env.max_episode_seconds=60.0 \
  --override 'env.commands.twist.resampling_time_range=[60.0,60.0]'
# keep model_22499

# 3. Fall penalty and mild pushes.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straighthold_flat \
  --load-run <run-2> --checkpoint 22499 --iterations 2000 \
  --algo-log-name rsl_rl_ppo_straighthold_r4 --eval-duration 120
# keep model_24000

# 4. Head-flip cost.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straighthold_flat \
  --load-run <run-3> --checkpoint 24000 --iterations 2000 \
  --algo-log-name rsl_rl_ppo_straighthold_r5 \
  --eval-duration 300 --eval-envs 128 \
  --override reward.head_tilt_excess.weight=-20.0
# keep model_24250

# 5. Stronger head-flip cost, half learning rate.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straighthold_flat \
  --load-run <run-4> --checkpoint 24250 --iterations 1000 --save-interval 125 \
  --algo-log-name rsl_rl_ppo_straighthold_r6 \
  --eval-duration 300 --eval-envs 128 \
  --override reward.head_tilt_excess.weight=-40.0 \
  --override algo.algorithm.learning_rate=5.0e-5
# keep model_25125

# 6. Stress-level pushes, CoM, and reset tilt.
uv run --no-sync scripts/train_sprint_straight_speed.py \
  --task microduck_sprint_straighthold_flat \
  --load-run <run-5> --checkpoint 25125 --iterations 2000 \
  --algo-log-name rsl_rl_ppo_straighthold_r8 \
  --eval-duration 300 --eval-envs 128 \
  --override reward.head_tilt_excess.weight=-40.0 \
  --override 'env.events.push_robot.params.velocity_range.x=[-0.10,0.10]' \
  --override 'env.events.push_robot.params.velocity_range.y=[-0.10,0.10]' \
  --override 'env.events.base_com.params.com_range.x=[-0.008,0.008]' \
  --override 'env.events.base_com.params.com_range.y=[-0.008,0.008]' \
  --override 'env.events.base_com.params.com_range.z=[-0.008,0.008]' \
  --override 'env.events.head_com.params.com_range.x=[-0.005,0.005]' \
  --override 'env.events.head_com.params.com_range.y=[-0.005,0.005]' \
  --override 'env.events.head_com.params.com_range.z=[-0.005,0.005]' \
  --override 'env.events.reset_base.params.pose_range.roll=[-0.0262,0.0262]' \
  --override 'env.events.reset_base.params.pose_range.pitch=[-0.0262,0.0262]'
# keep model_27000
```

Step 2's overrides are the `microduck_sprint_straighthold_flat` weights
without its push, fall penalty, and head-flip cost.

## Eval / play

```bash
uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_straight_fast/model_27000.pt \
  --owner microduck_sprint_straightspeed_flat/mujoco \
  --duration 300 --num-envs 128

uv run --no-sync scripts/eval_sprint_speed.py \
  examples/sprint_straight_fast/model_27000.pt \
  --owner microduck_sprint_straighthold_flat/mujoco --stress \
  --duration 60 --num-envs 128

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_straight_fast/model_27000.pt \
  --out-dir examples/sprint_straight_fast --only follow \
  --follow-seconds 10 --stride 1 --follow-azimuth 90 --follow-elevation -8 \
  --follow-distance 0.9 --follow-name play_video_side.mp4
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_straight_fast/model_27000.pt \
  --out-dir examples/sprint_straight_fast --only follow
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --no-sync scripts/render_straight_demos.py \
  --checkpoint examples/sprint_straight_fast/model_27000.pt \
  --out-dir examples/sprint_straight_fast --only radial --radial-seconds 7
```
