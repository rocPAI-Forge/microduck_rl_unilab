# Velstand get-up example

Published checkpoint for `microduck_velstand_flat`. The actor was warm-started
from the standup specialist (the critic does not transfer: standup is 74-dim,
velstand is 76-dim), then trained for 4000 PPO iterations at 2048 environments.
`model_2250.pt` is the checkpoint whose side-view playback was checked.

| File | What it shows |
|---|---|
| [`model_2250.pt`](model_2250.pt) | Selected weights, iteration 2250 |
| [`play_video_side.mp4`](play_video_side.mp4) | 3 s side view, real time. Prone, push-up, crouch, stand. |
| [`play_video_side_slow.mp4`](play_video_side_slow.mp4) | Same clip at 3× slow motion |

Raw numbers: [`metrics.json`](metrics.json). Source run
`logs/rsl_rl_ppo/MicroduckVelstandFlat/2026-10-18_02-21-14_mujoco`
(seed 42, final mean reward 93.53, best 127.84).

With four prone starts at seed 42, environments 0 and 2 are upright within
1 s. Environments 1 and 3 stay near 90° of tilt for the full 8 s fallen
window. The video follows environment 2.

## Play

A plain eval starts upright. The step-0 `prone_init_prob` curriculum writes
`prone_prob` back to 0 on the first reset, and the tilt termination resets a
prone robot immediately. Both have to be turned off, and the camera has to
stay on the robot that actually stands:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run microduck-eval --algo ppo \
  --task microduck_velstand_flat --sim mujoco \
  --load-run "$(pwd)/examples/velstand_getup" \
  algo.checkpoint=2250 \
  training.play_render_mode=record \
  training.play_env_num=4 training.play_steps=150 \
  training.cam_distance=0.65 training.cam_elevation=-8 training.cam_azimuth=1.1 \
  training.cam_tracking=true training.cam_tracking_env_idx=2 \
  training.cam_tracking_extra_envs=0 \
  training.device=cuda:0 \
  env.events.random_prone_init.params.prone_prob=1.0 \
  env.events.random_prone_init.params.face_down_prob=0.5 \
  env.events.random_prone_init.params.crouch_prob=0.0 \
  env.curriculum.prone_init_prob=null \
  env.curriculum.fell_over_disable=null \
  env.terminations.tilt=null
```

`cam_azimuth=1.1` is the side of environment 2 at `env.seed=42`.
`algo.load_run` must be a directory that contains `model_2250.pt`.
