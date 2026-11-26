# Basketball example

Balancing on a free basketball, after Vottivott/microduck-playground's b11.
**Partial result**: it balances well under small commands and follows them
weakly, but collapses under large switching commands. Simulation only; no
robot run. [`play_video.mp4`](play_video.mp4) is 15 s on a
free ball: still 5 s, forward 0.10 m/s 5 s, turning +0.3 rad/s 5 s.

MotrixSim cross-check: **fails** (60 s survival 7.8-10.9% on Motrix against 99.2-100% on MuJoCo; [`docs/audits/sim2sim_tricks.md`](../../docs/audits/sim2sim_tricks.md)).

| File | Contents |
|---|---|
| [`model_12491.pt`](model_12491.pt) | Round 10 weights (LSTM actor, hence the size) |
| [`play_video.mp4`](play_video.mp4) | 15 s tracking view |
| [`metrics.json`](metrics.json) | Fixed-command and playground-protocol evaluations |

## Training

Ten rounds, 4096 environments: a hold curriculum that lowers a spring/damper
pin on the ball to zero, then tracking fixes (clean command sampling, tight
kernels, linear speed and yaw costs, action-rate −1.0, wider centering).
Round 10 is `microduck_basketball_ft9_flat/mujoco`, 500 iterations from
round 9. Details in [`docs/audits/basketball.md`](../../docs/audits/basketball.md).

## Battery

Fixed commands, `scripts/eval_basketball.py`, free ball, 256 environments, 60 s,
pushes on, seeds 101 / 202 / 303 (means):

| Command (vx, wz) | Survival 60 s | Forward speed | Mean yaw rate |
|---|---:|---:|---:|
| (−0.10, 0) | 97.9% | −0.024 m/s | +0.08 rad/s |
| (0, 0) | 99.1% | +0.023 m/s | +0.08 rad/s |
| (+0.10, 0) | 97.9% | +0.071 m/s | +0.11 rad/s |
| (0, −0.30) | 99.2% | +0.024 m/s | −0.10 rad/s |
| (0, +0.30) | 98.7% | +0.022 m/s | +0.25 rad/s |

Playground protocol, `scripts/eval_basketball_playground.py`, 1024
environments, 60 s, random commands (±0.30 m/s forward, ±0.20 lateral,
±1.0 rad/s yaw) resampled every 3-8 s:

| Pushes | Survival 60 s |
|---|---:|
| every 0.5-1.0 s (±0.09 m/s) | 3.5% |
| every 3-6 s | 32% |
| none | 43% |

Playground b11 reports 97% under its protocol. The ablation in the study
shows the cause: large forward commands and switching together.

The basketball scene is generated (git-ignored); write it first:

```bash
uv run --no-sync scripts/make_basketball_scene.py
uv run --no-sync scripts/eval_basketball.py examples/basketball/model_12491.pt \
  --owner microduck_basketball_ft9_flat/mujoco --duration 60 --seed 101
```
