# Standup example

Published checkpoint for `microduck_standup_flat`. 15000 PPO iterations at
2048 environments, seed 42. [`play_video.mp4`](play_video.mp4) is a 3× slow
side view, about 22 s, in three parts: prone push-up, supine roll to stand,
then sit-to-stand.

| File | Contents |
|---|---|
| [`model_14999.pt`](model_14999.pt) | Final weights, iteration 14999 |
| [`play_video.mp4`](play_video.mp4) | 22 s slow side view of the three get-ups |

Raw numbers: [`metrics.json`](metrics.json). Final mean reward 46.97, best
56.69.

A plain eval uses the step-0 mix (standing 0.40, sitting 0.40, face-down 0.20,
face-up 0). The published clip forces one pose at a time and turns that
curriculum off. Repeat with the other two probability triples
`(face_down, face_up, sitting, standing) = (0,1,0,0)` and `(0,0,1,0)`,
then join the three recordings at 3× slow motion. Prone example:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run microduck-eval --algo ppo \
  --task microduck_standup_flat --sim mujoco \
  --load-run "$(pwd)/examples/standup" \
  algo.checkpoint=14999 \
  training.play_render_mode=record \
  training.play_env_num=1 training.play_steps=120 \
  training.cam_distance=0.7 training.cam_elevation=-8 training.cam_azimuth=58.5 \
  training.cam_tracking=true training.cam_tracking_extra_envs=0 \
  training.device=cuda:0 \
  env.events.set_ground_state.params.face_down_prob=1 \
  env.events.set_ground_state.params.face_up_prob=0 \
  env.events.set_ground_state.params.sitting_prob=0 \
  env.events.set_ground_state.params.standing_prob=0 \
  env.curriculum.ground_state_mix=null
```

`algo.load_run` must be a directory that contains `model_14999.pt`.
`cam_azimuth` is the side of the seed-42 spawn for that pose.
