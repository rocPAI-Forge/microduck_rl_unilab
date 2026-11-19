# Basketball balance: result and rounds

Task `MicroduckBasketballFlat`: the duck stands on the apex of a free ball and is
meant to follow the twist command by rolling it. The ball is held by
`BallHoldAction` at a curriculum level that goes to 0 (free ball) during
training. Port of the playground basketball experiment (b11).

## Result

Partial. The policy balances a free ball well, but it follows commands only
weakly: forward speed is ordered with the command, yaw is not tracked at all.

Accepted: Round 4 (`microduck_basketball_ft3_flat`, checkpoint `model_9996`).

## Round table

Evaluation: `scripts/eval_basketball.py`, hold 0 (free ball), 60 s, 256 envs,
3 seeds (101, 202, 303), pushes on. Each row is the mean over 3 seeds.
Survival is the fraction of envs still up at 60 s; the forward speed is the
signed mean body-x velocity over surviving samples.

| Round | Owner | Start | Change | Survival @60 s (avg of 9 commands) | vx at -0.10 / 0 / +0.10 m/s | Yaw error at wz=0 (rad/s) |
|---|---|---|---|---|---|---|
| 1 | `microduck_basketball_flat` | scratch, 6000 iters | base recipe | - | - | - |
| 2 | `microduck_basketball_ft_flat` | R1 `model_5999` | action-rate -0.2, hold 0.1/0.03/0 | 0.72 (vx 0) | 0.052 / 0.055 / 0.058 | 0.81 |
| 3 | `microduck_basketball_ft2_flat` | R2 `model_8998` | commands +-0.30/+-0.20/+-1.0, tracking 2.5 / 1.25 | 0.99 | 0.063 / 0.067 / 0.072 | 0.54 |
| 4 | `microduck_basketball_ft3_flat` | R3 `model_9497` | lin kernel std 0.1 (was 0.316), weight 4.0; yaw std 0.3, weight 2.0 | 0.98 | 0.029 / 0.041 / 0.054 | 0.74 |
| 5 | (not kept) | R4 `model_9996` | weights doubled (8.0 / 4.0) | 0.83 | 0.007 / 0.018 / 0.041 | 1.08 |

Round 5 was discarded: balance fell and yaw got worse, and the forward response
compressed toward zero.

## Why round 3 looked like tracking but was not

The linear tracking kernel is `exp(-err^2 / std^2)` with std 0.316 m/s. A
motionless duck given a 0.15 m/s command already earns about 0.8 of the reward,
so the reward barely distinguishes "follows" from "stands". The training
tracking reward rose during round 3 while the evaluated speed response stayed
flat, which is what this predicts. Round 4 tightens the kernels (std 0.1 m/s
linear, 0.3 rad/s yaw) to make the response matter.

## Open items

- Yaw is not tracked in any round. Yaw error grows with the yaw weight and
  with survival pressure, so a yaw fix probably needs its own kernel or command
  design rather than a larger weight.
- A zero command is sampled as about +0.04 m/s in the probe (`/tmp/bb_cmdprobe.py`,
  not in the repo). The cause is not yet traced; it may shift the training
  command distribution.
- Forward MAE is not discriminative at this command range. A constant
  0.05 m/s predictor gives MAE about 0.154 on commands uniform in +-0.30 m/s,
  which matches playground b11's reported 0.156.
- Round 4 has no re-rendered video; the earlier Round 2 clips are the latest
  visual reference.
