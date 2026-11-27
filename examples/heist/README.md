# Stop and go (Microduck Arena 6 m Sock Heist)

`model_59500`: [`../sprint_stand/`](../sprint_stand/) `model_59000` resumed
for 500 iterations on `microduck_heist_flat`. The command is re-drawn every
0.6-3.0 s, 45% of draws a stand, walking at 0.5-1.2 m/s, and
`scoot_terms.stillness_at_zero_command` (weight 4) rewards a still trunk on a
zero command. Arena entry
[`heist-stopgo-rocm`](https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M8F0CH7QB5B3H5F6PQBT):
**13.516 s, 5/5 finishes** ([`arena_score.txt`](arena_score.txt)).

The Sock Heist is red light, green light: on red the Arena zeroes the command,
allows 0.4 s to stop, then catches a duck whose trunk strays more than 3 cm.
`sprint_stand` finishes only at 0.30-0.45 m/s (25.3 s): from 0.5 m/s it is
caught at the first red. This checkpoint finishes at 0.30-1.20 m/s except
1.10 m/s, and scores at 1.05 m/s.

Later checkpoints of the same run were worse (16.3 s at 60000, 26-29 s with
3/5 after), so the stand-still term is not yet balanced against walking.

## Train / score

```bash
uv run --no-sync microduck-train --algo ppo --task microduck_heist_flat --sim mujoco \
  algo.load_run="$(pwd)/examples/sprint_stand" algo.checkpoint=59000 \
  algo.num_envs=2048 training.no_play=true

uv run --no-sync scripts/export_policy.py <run>/model_59500.pt \
  --owner microduck_heist_flat/mujoco --output-dir examples/heist/deploy
# in a Microduck Arena checkout (docs/ARENA.md):
uv run arena score .../examples/heist/deploy/policy.onnx --event sock-heist-6m
```
