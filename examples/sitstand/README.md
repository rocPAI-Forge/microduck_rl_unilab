# Sit/stand example

Commanded sit and stand in one policy, after upstream microduck_rl's sitstand
recipe. The command is a posture flag in `twist.vx` (1 sit, 0 stand), slewed
over 2 s. Simulation only; no MotrixSim check, no robot run.
[`play_video.mp4`](play_video.mp4) is 15 s: stand 3 s, sit 6 s, stand.

| File | Contents |
|---|---|
| [`model_6500.pt`](model_6500.pt) | Round 5 weights, iteration 6500 |
| [`play_video.mp4`](play_video.mp4) | 15 s tracking view |
| [`metrics.json`](metrics.json) | Evaluation on three seeds |

## Training

Five rounds on the standup base; round 5 is `microduck_sitstand_fix_flat/mujoco`,
2000 iterations from round 4 (height-gated sharp upright, a 5× joint-limit
cost, heavier head tracking). Details in
[`docs/audits/sitstand.md`](../../docs/audits/sitstand.md).

## Battery

`scripts/eval_sitstand.py`, 512 environments, stand 1 s, sit until 7 s, stand
until 12 s:

| Seed | Sit | Seated hold | Rise | Falls | Peak descent p95 | Peak rise p95 |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 87.9% | 87.7% | 97.1% | 0% | 0.87 m/s | 0.43 m/s |
| 202 | 89.3% | 89.2% | 97.5% | 0% | 0.88 m/s | 0.43 m/s |
| 303 | 85.2% | 85.2% | 97.3% | 0.2% | 0.88 m/s | 0.43 m/s |

Open: the trunk reaches about 0.88 m/s at the end of the descent, a drop of
about 4 cm onto the seat. Two gentler-descent variants made it stop sitting.

```bash
uv run --no-sync scripts/eval_sitstand.py examples/sitstand/model_6500.pt \
  --owner microduck_sitstand_fix_flat/mujoco --num-envs 512 --seed 101
```
