# Roulade example

Forward roll over the flat top of the head, landing on the feet, in one
policy: upstream microduck_rl's roulade, reproduced on MuJoCo. Simulation only;
it has not run on a robot.
[`play_video.mp4`](play_video.mp4) is a 2× slow side view from a standing start.

MotrixSim cross-check: passes (full turn and standing 99.2% on Motrix, 94.5-95.3% on MuJoCo; [`docs/audits/sim2sim_tricks.md`](../../docs/audits/sim2sim_tricks.md)).

| File | Contents |
|---|---|
| [`model_999.pt`](model_999.pt) | Weights after 1000 fine-tuning iterations |
| [`play_video.mp4`](play_video.mp4) | 8 s, 2× slow: crouch, chin tuck, head plant, headstand, over the back, land, stand |
| [`metrics.json`](metrics.json) | Evaluation on three seeds |

## Where the weights come from

The actor starts from upstream's published `roulade.onnx`
([pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies),
revision `236d1835`, Apache-2.0), converted with `scripts/onnx_to_rsl_rl.py`
(same 61→512→256→128→14 network; the baked-in observation normalizer is
recovered). The critic comes from our own from-scratch roulade run. Then 1000
PPO iterations on `microduck_roulade_flat/mujoco` (4096 envs, learning rate
1e-4, action std 0.2 at the start).

Our own training did not get there: three attempts (warm start from standup,
the same with a roll-back cost, and upstream's recipe from scratch to 6000
iterations) learned the second half but never rolled over the head from a
standing start. The published policy does it in our simulator unchanged, so the
gap was the training, not the physics; see
[`docs/audits/roulade.md`](../../docs/audits/roulade.md).

Fine-tuning changed one thing that matters here: upstream runs this policy for
1 s and hands over to its walking policy, so on its own it often keeps rolling
(rotation p90 969°, standing at the end 82%). After fine-tuning it rolls once
and stands.

## Battery

`scripts/eval_roll.py`, 512 environments, standing starts, 4.9 s, seeds 101 /
202 / 303 (identical to three digits):

| Metric | Upstream `roulade.onnx` in our sim (256 envs) | `model_999` |
|---|---:|---:|
| Roll complete (past 260° with the head-top latch) | 98.4% | 100% |
| Time to complete (median) | 1.00 s | 0.86 s |
| More than one turn | yes (p90 969°) | 0% (p90 353°) |
| Peak pitch rate (median) | 8.9 rad/s | 7.9 rad/s |
| Standing at the end | 81.6% | 100% |

Stood up after the roll: 100%, median 1.04 s from the start.

The owner uses the convex-hull scene, which `scripts/make_hull_robot.py`
writes into `assets/robots/microduck/generated/` (git-ignored); run it once.

```bash
uv run --no-sync scripts/make_hull_robot.py
uv run --no-sync python scripts/eval_roll.py examples/roulade/model_999.pt \
  --owner microduck_roulade_flat/mujoco --num-envs 512 --seed 101
```
