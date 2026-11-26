# Scoot example

Seated scooting ("butt walk"), after Vottivott/microduck-playground's MicroDock
v2 training: the duck sits and moves backward or forward on command.
Simulation only; no robot run.
[`play_video.mp4`](play_video.mp4) is 15 s: backward 0.10 m/s for 10 s, then stop.

MotrixSim cross-check: passes (95.3-98.4% on Motrix, 100% on MuJoCo; [`docs/audits/sim2sim_tricks.md`](../../docs/audits/sim2sim_tricks.md)).

| File | Contents |
|---|---|
| [`model_2999.pt`](model_2999.pt) | Round 4 weights, iteration 2999 |
| [`play_video.mp4`](play_video.mp4) | 15 s tracking view |
| [`metrics.json`](metrics.json) | Evaluation, three seeds × four commands |

## Training

Round 4, `microduck_scoot_sym_flat/mujoco`, 3000 iterations from scratch at
4096 environments on the convex-hull ground-contact robot with contact
priority 1, so the shin and seat friction curriculum takes effect. Mirror loss
on. Rounds 1-3 could not back up because the floor's friction overrode the
curriculum. Details in [`docs/audits/scoot.md`](../../docs/audits/scoot.md) and
[`docs/audits/friction_combination.md`](../../docs/audits/friction_combination.md).

## Battery

`scripts/eval_walk.py`, 128 environments, 10 s, seeds 101 / 202 / 303.
Curricula are off, so shin and seat friction stay at the first stage
(0.05-0.25):

| Command | Survival | Forward speed | Yaw rate |
|---|---:|---:|---:|
| −0.10 m/s | 100% | −0.082 to −0.085 m/s | +0.01 rad/s |
| −0.05 m/s | 100% | −0.043 to −0.044 m/s | −0.01 to −0.03 rad/s |
| 0 | 100% | ≈ 0 | ≈ 0 |
| +0.05 m/s | 100% | 0.029-0.036 m/s | −0.06 to −0.07 rad/s |

Backing up reaches 82-88% of the command. Forward is weaker and drifts right.

The hull robot is generated (git-ignored); write it first:

```bash
uv run --no-sync scripts/make_hull_robot.py
uv run --no-sync scripts/eval_walk.py examples/scoot/model_2999.pt --owner microduck_scoot_sym_flat/mujoco \
  --speed -0.10 --duration 10 --num-envs 128 --seed 101
```
