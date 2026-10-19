# Velstand get-up example

Published checkpoint for `microduck_velstand_flat`. One policy stands from
both a face-down and a face-up start, then holds the stand under a
stand-still command and under the default walk commands.

The actor is the standup specialist copied into the velstand policy. The
velstand critic was kept (standup is 74-dim, velstand is 76-dim). It was then
fine-tuned for 800 PPO iterations on stand-still get-up from both poses, and
1200 more iterations with walk commands mixed back in (2048 environments,
learning rate 3e-4). `model_12498.pt` is the checkpoint that passed the
forced-fall check.

| File | What it shows |
|---|---|
| [`model_12498.pt`](model_12498.pt) | Selected weights, stored iteration 12498 |
| [`play_video_getup.mp4`](play_video_getup.mp4) | Side-by-side, about 8 s at half speed. Left: face-down. Right: face-up. |

Raw numbers: [`metrics.json`](metrics.json). Source run
`logs/rsl_rl_ppo/MicroduckVelstandFlat/2026-10-18_20-57-22_mujoco`.
Final mean reward 115.61, mean episode length 1000.

On 64 forced falls (half face-down, half face-up), both poses stand within
8 s, and the mean tilt at 8 s is 1.2°. The same score holds with the default
walk commands. A seed-42 single-env probe reaches 0.6° at 0.5 s from
face-down and 0.9° at 1.0 s from face-up.

## Play

A plain eval starts upright. The step-0 `prone_init_prob` curriculum writes
`prone_prob` back to 0 on the first reset, and the tilt termination resets a
fallen robot. Both have to be turned off. Record each pose, then join them
side by side at half speed. Face-down:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run microduck-eval --algo ppo \
  --task microduck_velstand_flat --sim mujoco \
  --load-run "$(pwd)/examples/velstand_getup" \
  algo.checkpoint=12498 \
  training.play_render_mode=record \
  training.play_env_num=1 training.play_steps=200 \
  training.cam_distance=0.65 training.cam_elevation=-8 training.cam_azimuth=-83.6 \
  training.cam_tracking=true training.cam_tracking_env_idx=0 \
  training.cam_tracking_extra_envs=0 \
  training.device=cuda:0 \
  env.events.random_prone_init.params.prone_prob=1.0 \
  env.events.random_prone_init.params.face_down_prob=1.0 \
  env.events.random_prone_init.params.crouch_prob=0.0 \
  env.curriculum.prone_init_prob=null \
  env.curriculum.fell_over_disable=null \
  env.terminations.tilt=null \
  env.commands.twist.rel_standing_envs=1.0 \
  env.commands.twist.rel_forward_envs=0.0 \
  env.commands.twist.turn_in_place_fraction=0.0 \
  env.commands.twist.ranges.lin_vel_x=[0.0,0.0] \
  env.commands.twist.ranges.lin_vel_y=[0.0,0.0] \
  env.commands.twist.ranges.ang_vel_z=[0.0,0.0] \
  env.curriculum.standing_envs=null
```

Repeat with `face_down_prob=0.0` for the face-up clip. `cam_azimuth=-83.6`
is the side of the seed-42 spawn. `algo.load_run` must be a directory that
contains `model_12498.pt`.
