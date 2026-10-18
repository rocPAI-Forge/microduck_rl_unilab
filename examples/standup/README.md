# Standup example

Published checkpoint for `microduck_standup_flat`. 15000 PPO iterations at
2048 environments, seed 42. [`play_video.mp4`](play_video.mp4) is the checked
clip: eight robots, fixed camera, getting up from prone, supine, and sitting
poses.

| File | Contents |
|---|---|
| [`model_14999.pt`](model_14999.pt) | Final weights, iteration 14999 |
| [`play_video.mp4`](play_video.mp4) | 12 s fixed-camera playback |

Raw numbers: [`metrics.json`](metrics.json). Final mean reward 46.97, best
56.69.

A plain eval uses the step-0 mix (standing 0.40, sitting 0.40, face-down 0.20,
face-up 0). Turn that curriculum off to keep the hard mix:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run microduck-eval --algo ppo \
  --task microduck_standup_flat --sim mujoco \
  --load-run "$(pwd)/examples/standup" \
  algo.checkpoint=14999 \
  training.play_render_mode=record \
  training.play_env_num=8 training.play_steps=600 \
  training.cam_distance=2.0 training.cam_elevation=-14 training.cam_azimuth=135 \
  training.cam_tracking=false \
  training.device=cuda:0 \
  env.events.set_ground_state.params.standing_prob=0.15 \
  env.events.set_ground_state.params.sitting_prob=0.20 \
  env.events.set_ground_state.params.face_down_prob=0.30 \
  env.events.set_ground_state.params.face_up_prob=0.35 \
  env.curriculum.ground_state_mix=null
```

`algo.load_run` must be a directory that contains `model_14999.pt`.
