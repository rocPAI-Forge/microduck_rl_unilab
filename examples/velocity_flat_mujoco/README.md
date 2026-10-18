# Velocity flat, MuJoCo

Final PPO checkpoint for `microduck_velocity_flat` on MuJoCo. 500 iterations,
2048 environments, seed 42. Final mean reward 93.01. No inference video was
recorded for this run.

[`model_499.pt`](model_499.pt). Numbers: [`metrics.json`](metrics.json).

```bash
uv run microduck-eval --algo ppo --task microduck_velocity_flat --sim mujoco \
  --load-run "$(pwd)/examples/velocity_flat_mujoco" \
  algo.checkpoint=499
```
