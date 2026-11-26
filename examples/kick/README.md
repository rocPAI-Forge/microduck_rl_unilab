# Ball dribbling example

The duck walks to a free ball and pushes it along a goal direction with its
feet. It dribbles; it does not strike the ball. Simulation only; no MotrixSim
check, no robot run. [`play_video.mp4`](play_video.mp4) is a 12 s tracking view.

| File | Contents |
|---|---|
| [`model_60499.pt`](model_60499.pt) | Weights after 1500 iterations from `sprint_stand` |
| [`play_video.mp4`](play_video.mp4) | 12 s: walks behind the ball and dribbles it |
| [`metrics.json`](metrics.json) | Evaluation on three seeds |

## How it works

`microduck_kick_flat/mujoco` keeps the published `sprint_stand` gait
(`examples/sprint_stand/model_59000.pt` is the warm start):

- `BallPursuitCommand` rewrites the twist command every step: walk to a point
  behind the ball on the goal line, then through the ball along the goal
  (forward at most 0.3 m/s, yaw rate at most 0.8 rad/s). Velocity tracking does
  the walking.
- The policy sees the ball and the goal in the head-pose command slot, which
  is zero in `sprint_stand`. The values are scaled down because the warm
  start's normalizer has a std of 0.004-0.018 there.
- Rewards add ball velocity along the goal (weight 2) and charge ball velocity
  across it (weight -0.5).
- Scene: the walking model plus a 3.5 cm, 45 g ball
  (`assets/robots/microduck/scene_flat_bam_kick.xml`). Only the foot soles
  collide with it in this model, so the shins pass through the ball and a real
  strike is not possible.

## Battery

`scripts/eval_kick.py`, 256 environments, 12 s, pushes off, seeds 101 / 202 / 303:

| Metric | `sprint_stand` with the pursuit command, no training | `model_60499` |
|---|---:|---:|
| Survival | 66-71% | 99.6-100% |
| Touched the ball | 90-95% | 98-99% |
| Ball along the goal, median over all envs | 1.06-1.55 m | 1.02-1.41 m |
| Ball past 0.5 m along the goal | 57-65% | 57-68% |
| Ball across the goal, mean | 0.37-0.46 m | 0.22-0.23 m |

Training removed the falls and halved the sideways error; the distance the
ball travels did not change.

```bash
uv run --no-sync python scripts/eval_kick.py examples/kick/model_60499.pt --seed 101
```

Upstream microduck_rl's `BallKick` task (a 70 mm, 15 g ball, the
ground-contact model, a ball-blind actor) and its published
`ball_kick_left/right.onnx` were not tried here.
