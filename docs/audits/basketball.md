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

## Command distribution bug (found after round 5)

The walking recipe sets `rel_forward_envs: 0.2`. `UniformVelocityCommand`
rewrites 20% of commands to `(max(|vx|, 0.3), 0, 0)`: above the basketball
range, never backward, zero yaw. Rounds 1 to 5 all trained on it, and
`scripts/eval_basketball.py` did not clear it, so 20% of eval envs were
commanded +0.3 m/s whatever `--speed` said. This is why a zero command averaged
+0.04 m/s (0.8 x cmd + 0.2 x 0.3 matches the probe for -0.2, 0 and +0.2).
The eval now clears `rel_forward_envs` and `rel_world_envs`, and also reports
the signed mean yaw rate (`yaw_rate_mean`).

Round 4 re-evaluated clean (60 s, 3 seeds): survival 0.94 to 0.995; vx
+0.014 / +0.030 / +0.047 at -0.10 / 0 / +0.10 m/s (slope about 0.17, offset
+0.03); mean yaw rate -0.148 / -0.106 / -0.051 at wz -0.30 / 0 / +0.30 (slope
about 0.16, bias -0.1). So yaw is weakly ordered too; the 0.75 rad/s error is
mostly oscillation. The yaw Gaussian (std 0.3) is flat at that error, so
weights do not damp it (round 5).

Round 6 (`microduck_basketball_ft5_flat`, `model_10495`, 500 iters from round 4):
`rel_forward_envs` 0, 20% turn-in-place episodes, and a linear
`yaw_rate_error_l1` reward (weight 1.0, cap 1.0). Clean eval (60 s, 3 seeds):

| cmd (vx, wz) | run | survival @60 s | vx | yaw error | mean yaw rate |
|---|---|---|---|---|---|
| (-0.10, 0) | r4 | 0.943 | +0.014 | 0.83 | -0.32 |
| (-0.10, 0) | r6 | 0.991 | -0.017 | 0.66 | -0.22 |
| (0, 0) | r4 | 0.987 | +0.030 | 0.74 | -0.11 |
| (0, 0) | r6 | 0.988 | +0.005 | 0.65 | -0.01 |
| (+0.10, 0) | r4 | 0.995 | +0.047 | 0.69 | +0.12 |
| (+0.10, 0) | r6 | 0.992 | +0.034 | 0.69 | +0.22 |
| (0, -0.30) | r4 | 0.987 | +0.028 | 0.76 | -0.15 |
| (0, -0.30) | r6 | 0.986 | +0.006 | 0.69 | -0.06 |
| (0, +0.30) | r4 | 0.980 | +0.032 | 0.78 | -0.05 |
| (0, +0.30) | r6 | 0.984 | +0.005 | 0.69 | +0.03 |

Round 6 removes the zero-command bias and gives the right sign in both
directions: vx slope about 0.26, yaw mean slope about 0.15 (about 11% of the
command at +0.30). Survival holds at 98% to 99%. Yaw error drops from about
0.75 to 0.67 but is still mostly oscillation. Forward and yaw following remain
weak.

## Open items

- Round 4 has no re-rendered video; the earlier Round 2 clips are the latest
  visual reference.
- Forward MAE is not discriminative at this command range. A constant
  0.05 m/s predictor gives MAE about 0.154 on commands uniform in +-0.30 m/s,
  which matches playground b11's reported 0.156.
- The round table above predates the eval fix; its vx column includes the 20%
  forced-forward envs. Round 4 and 6 clean numbers are in the Round 6 table.
