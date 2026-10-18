# Velocity flat, MuJoCo

Final PPO checkpoint for `microduck_velocity_flat` on MuJoCo. 500 iterations,
2048 environments, seed 42. Final mean reward 93.01.

[`play_video.mp4`](play_video.mp4) is a 10 s tracking shot at a 0.4 m/s forward
command. The robot stays up and steps. The head pitches forward.

| File | Contents |
|---|---|
| [`model_499.pt`](model_499.pt) | Final weights, iteration 499 |
| [`play_video.mp4`](play_video.mp4) | 10 s forward walk |

Numbers: [`metrics.json`](metrics.json).

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run microduck-eval --algo ppo \
  --task microduck_velocity_flat --sim mujoco \
  --load-run "$(pwd)/examples/velocity_flat_mujoco" \
  algo.checkpoint=499 \
  training.play_render_mode=record \
  training.play_env_num=1 training.play_steps=500 \
  training.cam_distance=1.0 training.cam_elevation=-12 training.cam_azimuth=130 \
  training.cam_tracking=true training.cam_tracking_extra_envs=0 \
  training.device=cuda:0 \
  env.commands.twist.rel_standing_envs=0.0 \
  env.commands.twist.rel_forward_envs=1.0 \
  'env.commands.twist.ranges.lin_vel_x=[0.4,0.4]' \
  'env.commands.twist.ranges.lin_vel_y=[0.0,0.0]' \
  'env.commands.twist.ranges.ang_vel_z=[0.0,0.0]' \
  env.curriculum.standing_envs=null
```

`algo.load_run` must be a directory that contains `model_499.pt`.
