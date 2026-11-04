# MotrixSim in the training loop (roadmap item 9, exploratory)

**Result:** fine-tuning on one simulator moves the overfit instead of removing
it. Alternating between MotrixSim and MuJoCo gives a gait that holds on both,
at the price of about 11 points of MuJoCo top-speed survival. No checkpoint
from this experiment is published.

## Method

All runs resume `examples/sprint_stand/model_59000.pt` with the
`microduck_sprint_standturn_flat` recipe, 2048 environments.

- **Motrix only:** 1500 iterations on the `microduck_sprint_standturn_flat/motrix`
  owner, giving `model_60499`.
- **Alternating:** 8 phases of 200 iterations, Motrix first and MuJoCo last,
  each phase resuming the previous one (`scripts/alt_backend_finetune.sh`),
  giving `model_60592`.

Scoring uses `scripts/eval_sprint_speed.py --clip-targets` with 512 environments
and seed 777. Straight runs are loop-held for 300 s. Turns are 60 s with the
loop off. `mujoco_nodr` is the MuJoCo owner with the four events Motrix cannot
apply removed, so it shares the Motrix randomization set.

## 1.0 m/s, 300 s survival (straight-pose pass)

| Checkpoint | Motrix | MuJoCo | MuJoCo, Motrix DR set |
|---|---|---|---|
| `model_59000` (MuJoCo only) | 67.4% (63.3%) | 95.1% (93.9%) | 98.0% (97.7%) |
| `model_60499` (Motrix only) | 97.1% (95.1%) | 7.4% (4.7%) | 13.9% (8.2%) |
| `model_60592` (alternating) | 87.7% (85.7%) | 84.2% (83.4%) | 90.8% (89.8%) |

## Turning at 1.0 m/s, −0.4 rad/s, 60 s: measured yaw rate / survival

| Checkpoint | Motrix | MuJoCo |
|---|---|---|
| `model_59000` | −0.37 / 76.8% | −0.40 / 98.4% |
| `model_60499` | −0.40 / 99.6% | −0.44 / 45.3% |
| `model_60592` | −0.36 / 95.1% | −0.37 / 98.0% |

The Motrix-only checkpoint also holds 0 and 0.6 m/s on Motrix (100% / 100%),
but loses 0 and 0.6 m/s on MuJoCo (92.8% / 96.7%).

The MuJoCo collapse of `model_60499` is the same with and without the extra
randomization, so it comes from simulator dynamics, not from the narrower
randomization set.

## Open

- The alternating checkpoint has not been scored at 0–0.6 m/s or under the
  MuJoCo stress battery.
- Longer or finer alternation (shorter phases, more of them) may close the
  MuJoCo gap. Mixing both backends inside one rollout batch is not supported
  by the current UniLab runner.
