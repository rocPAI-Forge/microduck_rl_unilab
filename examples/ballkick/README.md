# Ball kick example

A real kick: the duck stands with a 70 mm, 15 g ball in front of one foot,
swings that leg, sends the ball forward at about 1.3 m/s, and stands back up.
One policy per foot. Simulation only; no MotrixSim check, no robot run.
[`play_video_right.mp4`](play_video_right.mp4) and
[`play_video_left.mp4`](play_video_left.mp4) are 6 s at 2× slow.

| File | Contents |
|---|---|
| [`model_right.pt`](model_right.pt) | Right-foot kick |
| [`model_left.pt`](model_left.pt) | Left-foot kick |
| [`play_video_right.mp4`](play_video_right.mp4), [`play_video_left.mp4`](play_video_left.mp4) | 2× slow tracking views |
| [`metrics.json`](metrics.json) | Evaluation, three seeds per foot |

## Where the weights come from

These are upstream microduck_rl's published `ball_kick_right.onnx` and
`ball_kick_left.onnx`
([pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies),
revision `236d1835`, Apache-2.0), converted with `scripts/onnx_to_rsl_rl.py`
and not retrained. The critic is the standup example's (same observation
layout); it only matters if you resume training. They run in this repo's
simulator unchanged, so no fine-tune was needed.

The setup follows upstream's `BallKick`:

- The ground-contact duck, whose legs collide with the ball.
  `examples/kick/` uses the walking model, where only the soles touch it,
  which is why that one dribbles.
- A 70 mm, 15 g hollow ball 9 cm ahead of the kicking foot and 4.2 cm to its
  side, with ±1.5 cm noise.
- A ball-blind actor: the operator aims the robot.

`microduck_ballkick_flat/mujoco` is that scene, reset and observation layout;
`scripts/make_ballkick_scene.py` writes the scene.

## Battery

`scripts/eval_ballkick.py`, 256 environments, standing start, zero commands,
4 s with the policy running the whole time, seeds 101 / 202 / 303:

| Foot | Kicked (ball > 5 cm forward) | Peak ball speed (median) | Ball forward at 4 s | Ball sideways at 4 s | Standing at the end |
|---|---:|---:|---:|---:|---:|
| Right | 100% | 1.30-1.33 m/s | 4.0-4.1 m | 0.76-0.78 m | 99.2-100% |
| Left | 99.2-99.6% | 1.28-1.33 m/s | 4.0-4.1 m | 0.52-0.56 m | 98.0-99.2% |

The ball leaves 7-11° off the heading. Upstream's target speed is 1.0 m/s.
Upstream deploys the kick for 0.5 s and hands back to its walking policy; here
the policy simply keeps running and stands back up on its own. Holding the
default joint targets after 0.5 s is not a stand: with no balance control the
duck falls.

The ball-kick scene is generated (git-ignored); write it first:

```bash
uv run --no-sync scripts/make_ballkick_scene.py
uv run --no-sync python scripts/eval_ballkick.py examples/ballkick/model_right.pt --foot right --duration 4 --seed 101
```
