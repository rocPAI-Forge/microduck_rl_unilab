# Top speed on MotrixSim (roadmap item 10, parked)

**Result:** under the simulator rule (train on MuJoCo only), wider servo and
contact randomization makes a 1.0 m/s command sturdier on MotrixSim only by
slowing it down. Sixty-second survival on Motrix rises from 85% to 92–94% when
the measured speed falls from 0.88 to 0.76–0.82 m/s; a checkpoint back at
0.85 m/s is back at 86%. No checkpoint is published. `sprint_stand/model_59000`
stays the hardware starting point; first runs stay at or below 0.6 m/s, where
it already holds 98–100% on both simulators.

## Which parameter matters

`model_59000`, 1.0 m/s, 60 s, 256 environments, seed 777, clamp on, MuJoCo with
one parameter pinned (`scripts/eval_sprint_speed.py --override`):

| Pinned parameter (training range) | Survival | Speed |
|---|---|---|
| none | 98.0% | 0.88 m/s |
| **MotrixSim instead of MuJoCo** | **85.2%** | **0.91** |
| armature ×0.5 (×0.9–1.1) | 0% | 1.14 |
| armature ×1.6 | 59.8% | 0.69 |
| servo delay 0–2 substeps (3–6) | 0% | 1.21 |
| servo delay 7–9 substeps | 14.5% | 0.63 |
| servo gain `kp_fw` 150 (fixed 200) | 21.1% | 0.90 |
| servo gain `kp_fw` 260 | 86.3% | 0.85 |
| servo friction ×1.6 (×0.9–1.1) | 55.9% | 0.85 |
| supply 6.2 V (6.5–8.2 V) | 82.0% | 0.89 |
| foot friction 1.8 (0.7–1.3) | 9.8% | 0.99 |
| foot friction 0.4, servo friction ×0.5, mass ×1.12 | 98–100% | — |

The top-speed gait fits a narrow band of servo dynamics. Motrix runs it
slightly faster, the same direction as a quicker servo or more foot grip.

## Training rounds

Both rounds resume the stand recipe and score on the driver battery.

| Round | Owner | Change | Start |
|---|---|---|---|
| 1 | `microduck_sprint_standwide_flat` | armature ×0.7–1.4, servo delay 2–8 substeps, servo friction ×0.75–1.35, foot friction 0.6–1.5 | `model_59000`, 3000 iterations |
| 2 | `microduck_sprint_standwidefast_flat` | round 1 plus fine forward-speed kernel weight 10 → 14 and high-speed share 0.3 → 0.45 | round 1 `model_60000`, 2500 iterations |

1.0 m/s, 60 s, 256 environments, seed 777:

| Checkpoint | MuJoCo survival / speed | Motrix survival / speed |
|---|---|---|
| `sprint_stand/model_59000` | 98.0% / 0.88 | 85.2% / 0.91 |
| round 1 `model_60000` | 98.4% / 0.82 | 91.8% / 0.82 |
| round 1 `model_61999` | 98.8% / 0.76 | 93.8% / 0.75 |
| round 2 `model_60500` | — / 0.82 | 92.2% / 0.82 |
| round 2 `model_61500` | — / 0.84 | 85.9% / 0.85 |

Round 2's heavier tracking moved speed only from 0.82 to 0.84 m/s, and the
Motrix margin went with it. Round 1 `model_60000`, and round 2 resumed from it,
also stopped walking at a 0.15 m/s command (0.01 m/s, against 0.13 for
`model_59000`). Standing, 0.3–0.6 m/s, and turns held in every round.

## If top speed matters later

Revisit after the first hardware run shows which simulator the robot
resembles. Untried here: randomizing the servo gain (`kp_fw`, among the most
sensitive parameters, has no randomization hook yet), and contact softness,
which UniLab 1.0.0 cannot randomize per environment.
