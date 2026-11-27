# Microduck Arena

[Microduck Arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena)
races Microduck policies server-side. It takes the same ONNX contract as
`robotd` and as this repo's exports (`scripts/export_policy.py`): 61
observations, 14 joint position targets, 50 Hz. Every `examples/*/deploy*/policy.onnx`
loads there unchanged.

## Entries

| Event | Entry | Policy | Score |
|---|---|---|---|
| 2 m Sprint | [`sprint-head-upright-rocm`](https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M8EGZDY83CFT2H7WCN30) | `examples/sprint_head_upright/model_14500` | 1.587 s, 5/5 |
| 2 m Sprint | [`sprint-stand-rocm`](https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M8EE2Y3DECSJM6B6K11D) | `examples/sprint_stand/deploy` | 1.997 s, 5/5 |
| Got a Talent | [`domino-ducks`](https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M8EE5YY1XQB82HACVNRX) | `examples/tumble/deploy` plus three borrowed ducks | voted |

The Arena's times reproduce on x86_64 Linux, so a local `arena score` here gives
the board's number (both sprint entries scored the same on the Space).

The sprint sweeps forward commands 0.20-1.20 m/s and keeps the best. Our walks
follow the command, so they race at about 1 m/s. `microduck_sprint_race_flat`
resumed `sprint_head_upright` with the speed-tracking term off and forward
progress rewarded to 3.0 m/s: it ran flat out at every command but no faster
(best 1.646 s, then slower), so it was not entered. The gait terms (stride cap,
leg alternation) are the likely limit.

## Got a Talent: Domino ducks

`arena_entries/talent/make_timeline.py` writes `timeline.json`: eight ducks on
the stage, one policy each.

- Three tumble ducks (the entry itself, `"self"`): `twist = [-1, 0, 0]` holds
  the tumble policy curled up; releasing to zeros sets off one sideways tumble
  about 0.2 s later, and it can be set off again after about 3 s.
- Two roulade ducks: the roulade policy rolls forward once, at the start, and
  no command delays it.
- Two sitstand ducks: `twist[0]` is the sit flag.
- One sprint_stand walker: from standing it needs 0.3 m/s; 0.2 m/s does not
  start a step.

A troupe borrows other ducks from public Hub repos pinned to a commit, so the
roulade, sitstand and walker policies are published as repos of their own
(`alexhegit/microduck-roulade`, `-sitstand`, `-sprint-stand`).

## Tools

Install the Arena once (its own environment; the vendored meshes and reference
policy come with the Space snapshot):

```bash
uv run --no-sync python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('pollen-robotics/microduck-arena', repo_type='space', local_dir='../microduck-arena')"
cd ../microduck-arena && uv sync
uv run arena score ../microduck_rl_unilab/examples/sprint_stand/deploy/policy.onnx --event sprint-2m
```

Play a stage with local ONNX files (the Arena itself only borrows from the Hub),
then render the trace on this repo's visual model (the Arena's stage model is
collision geometry only):

```bash
cd ../microduck-arena && uv run python ../microduck_rl_unilab/scripts/arena_stage_preview.py \
  ../microduck_rl_unilab/arena_entries/talent/timeline.json \
  --policies tumble.onnx tumble.onnx tumble.onnx roulade.onnx roulade.onnx \
             sitstand.onnx sitstand.onnx sprint_stand.onnx \
  --trace-out /tmp/talent.npz
cd ../microduck_rl_unilab && MUJOCO_GL=egl uv run --no-sync python scripts/render_arena_trace.py \
  /tmp/talent.npz --video arena_entries/talent/preview.mp4
```

Publish and enter (`hf auth login` with a write token first). This does what
microduck-challenges' `publish --onnx ... --enter` does, without its mjlab
dependencies: a public repo tagged `microduck-policy` with `policy.onnx`, a
`manifest.json` and a README (plus `timeline.json` for a stage), then the
Arena's submit endpoint with the Hub token.

```bash
uv run --no-sync scripts/arena_publish.py examples/sprint_stand/deploy/policy.onnx \
  --repo <user>/microduck-sprint-stand --event sprint-2m --enter
```
