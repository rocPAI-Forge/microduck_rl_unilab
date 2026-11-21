# Tumble example

A fast forward tumble that ends standing, in one policy. This is not
upstream's roulade (see [`../roulade/`](../roulade/)): it whips over the head
and back in about half a second rather than planting the top of the head and
going through a headstand. Simulation only; it has not run on a robot.
[`play_video.mp4`](play_video.mp4) is a 2× slow side view from a standing start.

| File | Contents |
|---|---|
| [`model_17997.pt`](model_17997.pt) | Final weights, iteration 17997 |
| [`play_video.mp4`](play_video.mp4) | 6 s, 2× slow: crouch, dive, tumble over, land on the feet, stand |
| [`metrics.json`](metrics.json) | Evaluation on three seeds |

## Training

Two stages, both 4096 environments on the convex-hull ground-contact model:

1. `microduck_roll_flat/mujoco`, 1500 iterations from the published standup
   policy (`examples/standup/model_14999.pt`): paid for forward rotation up to
   one full turn. It tumbled in about 0.8 s and then lay curled; getting up
   needed a switch to the standup policy.
2. `microduck_tumble_flat/mujoco`, 1500 iterations from stage 1: the same
   rewards plus stand-up terms gated on the completed turn (standing composite
   score, linear upright, upward velocity, and a tax on every step below
   standing height). Now one policy tumbles and stands.

## Battery

`scripts/eval_roll.py`, 512 environments, standing starts, 4.9 s, seeds 101 /
202 / 303:

| Metric | Value |
|---|---:|
| Full turn | 100% |
| Time to full turn (median) | 0.56 s |
| Rotation maximum (median) | 416° |
| Peak pitch rate (median) | 9.7-9.9 rad/s |
| Stood up after the turn | 100%, median 0.78-0.80 s from the start |
| Standing at the end | 100% |

The peak pitch rate is above upstream's 7 rad/s overspeed threshold for the
roulade: this is a violent maneuver, kept as a simulation demo.

The owner uses the convex-hull scene, which `scripts/make_hull_robot.py`
writes into `assets/robots/microduck/generated/` (git-ignored); run it once.

```bash
uv run --no-sync scripts/make_hull_robot.py
uv run --no-sync python scripts/eval_roll.py examples/tumble/model_17997.pt \
  --owner microduck_tumble_flat/mujoco --num-envs 512 --seed 101
```
