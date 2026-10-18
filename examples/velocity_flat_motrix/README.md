# Velocity flat, MotrixSim

Final PPO checkpoint for `microduck_velocity_flat` on MotrixSim
(`motrixsim-core==0.8.2`). 500 iterations, 2048 environments, seed 42. Final
mean reward 145.44. No inference video was recorded for this run.

[`model_499.pt`](model_499.pt). Numbers: [`metrics.json`](metrics.json).

```bash
uv run microduck-eval --algo ppo --task microduck_velocity_flat --sim motrix \
  --load-run "$(pwd)/examples/velocity_flat_motrix" \
  algo.checkpoint=499
```
