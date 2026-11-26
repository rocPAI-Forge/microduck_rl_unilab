# Stilts example

Walking on 10 cm stilts (peg ends), after Vottivott/microduck-playground.
Simulation only; no MotrixSim check, no robot run.
[`play_video.mp4`](play_video.mp4) is 15 s: 0.15 m/s for 9 s, then 0.10 m/s
while turning left at 0.4 rad/s.

| File | Contents |
|---|---|
| [`model_4091.pt`](model_4091.pt) | Final weights of the curriculum's last stage (10 cm, blend 0.5) |
| [`play_video.mp4`](play_video.mp4) | 15 s tracking view |
| [`metrics.json`](metrics.json) | Evaluation, three seeds × three commands |

## Training

`scripts/train_stilts.py` from `examples/velocity_flat_mujoco/model_499.pt`,
on `microduck_stilt_flat/mujoco`. Each stage writes its robot with
`scripts/make_stilt_robot.py` and resumes the previous stage. Support first
(platform to peg at 2 cm), then height; the two never get harder in the same
step:

| Stage | Height | Blend | Iterations |
|---|---|---|---|
| 0-2 | 2 cm | 0 → 0.25 → 0.5 | 800 / 200 / 200 |
| 3-5 | 3 → 4 → 5 cm | 0.5 | 200 / 200 / 1500 |
| 6-7 | 7.5 → 10 cm | 0.5 | 200 / 300 |

Details in [`docs/audits/stilts.md`](../../docs/audits/stilts.md).

## Battery

`scripts/eval_walk.py`, 128 environments, 60 s, seeds 101 / 202 / 303:

| Command | Survival | Forward speed | Yaw rate |
|---|---:|---:|---:|
| Stand | 100% | 0.064-0.081 m/s | −0.10 rad/s |
| 0.15 m/s | 100% | 0.169-0.182 m/s | −0.14 to −0.16 rad/s |
| 0.25 m/s | 99.2-100% | 0.239-0.249 m/s | −0.14 to −0.15 rad/s |

It walks and stands on stilts reliably but drifts forward at a standstill and
steers right by about 0.1-0.16 rad/s.

The stilt robot is generated (git-ignored); write the 10 cm stage first:

```bash
uv run --no-sync scripts/make_stilt_robot.py --height-cm 10 --blend 0.5
uv run --no-sync scripts/eval_walk.py examples/stilts/model_4091.pt --owner microduck_stilt_flat/mujoco \
  --speed 0.15 --duration 60 --num-envs 128 --seed 101 \
  --override env.scene.model_file=assets/robots/microduck/generated/scene_flat_bam_stilt_h10_b050.xml \
  --override 'env.scene.fragment_files=[assets/robots/microduck/generated/locomotion_task_stilt_h10_b050.xml]'
```
