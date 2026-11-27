# Sprint race, straight (Microduck Arena 2 m Sprint)

`model_12250`: [`../sprint_speed_1p68/`](../sprint_speed_1p68/) resumed for
250 iterations under the straighthold heading and cross-track terms, with the
head terms off (`microduck_sprint_race_straight_flat`). Arena entry
[`sprint-race-straight-rocm`](https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M8EW74ZW0J4SBM9WW2CT):
**1.411 s, 5/5 finishes**, and every one of the 21 swept commands finishes
([`arena_score.txt`](arena_score.txt)).

`sprint_speed_1p68` itself finishes in 1.42 s when it stays straight, but in
the Arena's 0.8 m lane it turned into the wall on 3 of 5 starts.

`deploy/policy.onnx` is the export with the forward command remapped: the
Arena sweeps 0.20-1.20 m/s and the checkpoint was trained on 1.65-2.20 m/s,
so `scripts/remap_onnx_command.py` maps one onto the other inside the graph.
It is a race policy only: it does not stand or follow a speed.

## Train / score

```bash
uv run --no-sync microduck-train --algo ppo --task microduck_sprint_race_straight_flat --sim mujoco \
  algo.load_run="$(pwd)/examples/sprint_speed_1p68" algo.checkpoint=11997 \
  algo.num_envs=2048 training.no_play=true

uv run --no-sync scripts/export_policy.py <run>/model_12250.pt \
  --owner microduck_sprint_race_straight_flat/mujoco --output-dir /tmp/race
uv run --no-sync scripts/remap_onnx_command.py /tmp/race/policy.onnx --to 1.65 2.2 \
  --output examples/sprint_race_straight/deploy/policy.onnx
# in a Microduck Arena checkout (docs/ARENA.md):
uv run arena score .../examples/sprint_race_straight/deploy/policy.onnx --event sprint-2m
```

Over the run (3000 iterations) the score moved between 1.41 and 1.53 s, with
some checkpoints at 3/5 or 4/5; `model_14996` also scored 1.410 s but fell at
two swept commands, so the earlier, cleaner checkpoint is the entry.
