# Basketball balance: result and rounds

Task `MicroduckBasketballFlat`: the duck stands on the apex of a free ball and is
meant to follow the twist command by rolling it. The ball is held by
`BallHoldAction` at a curriculum level that goes to 0 (free ball) during
training. Port of the playground basketball experiment (b11).

## Result

Partial, and still moving. The policy balances a free ball (about 95% or
better at 60 s from round 3 on). Command following is ordered but weak.
Current best is round 10 (`microduck_basketball_ft9_flat`, `model_12491`):
survival 98% to 99% at 60 s, forward slope about 0.48 (vx -0.024 / +0.023 /
+0.071 at -0.10 / 0 / +0.10 m/s), yaw slope about 0.58, yaw error about 0.33.
Zero command still drifts at about +0.023 m/s.
Round 4 was the best point before the tracking-kernel and command-sampling
fixes; its numbers are in the round 6 table.

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

## Round 7: action-rate penalty -1.0 (chatter)

Round 6 yaw spectrum (zero command): 46% of power at 2 to 10 Hz, and joints
1-3 and 10-12 chatter at 2 to 10 Hz (`/tmp/act_spec.py`, traces in
`logs/rsl_rl_ppo_bb_r6/traces/`).

Round 7 (`microduck_basketball_ft6_flat`, `model_10994`, 500 iters from round 6):
action-rate weight -0.2 to -1.0. Clean eval (60 s, 3 seeds):

| cmd (vx, wz) | run | survival @60 s | vx | yaw error | mean yaw rate |
|---|---|---|---|---|---|
| (-0.10, 0) | r6 | 0.991 | -0.017 | 0.66 | -0.22 |
| (-0.10, 0) | r7 | 0.965 | -0.023 | 0.38 | +0.00 |
| (0, 0) | r6 | 0.988 | +0.005 | 0.65 | -0.01 |
| (0, 0) | r7 | 0.986 | +0.011 | 0.40 | +0.04 |
| (+0.10, 0) | r6 | 0.992 | +0.034 | 0.69 | +0.22 |
| (+0.10, 0) | r7 | 0.992 | +0.049 | 0.47 | +0.10 |
| (0, -0.30) | r6 | 0.986 | +0.006 | 0.69 | -0.06 |
| (0, -0.30) | r7 | 0.996 | +0.011 | 0.44 | -0.05 |
| (0, +0.30) | r6 | 0.984 | +0.005 | 0.69 | +0.03 |
| (0, +0.30) | r7 | 0.993 | +0.011 | 0.42 | +0.14 |

Yaw error drops by about 40% (0.65 to 0.40), survival holds at 97% to 99.6%,
and the yaw slope rises from about 0.15 to about 0.31. Forward slope rises
from about 0.26 to about 0.36. Action std falls from 0.46 to 0.28 and action
rate std from 0.18 to 0.12. Chatter remains at about 3 Hz in joints 1-3 and
10-12 (it used to sit at 5-10 Hz), and yaw power at 2 to 5 Hz is now 44%.
Those joints are the stance joints (left/right hip roll, hip pitch, knee), so
the chatter is the balance correction, not the head. The action-rate penalty
was not raised further.

## Round 8: linear forward cost, yaw L1 weight 2

`microduck_basketball_ft7_flat`, `model_11493`, 500 iters from round 7. Adds
`lin_vel_error_l1` (weight 2, cap 0.4) and doubles `yaw_rate_l1` to 2. The
training cost plateaued by about iteration 11100 (lin_vel_l1 near -0.21,
yaw_rate_l1 near -1.16). Clean eval:

| cmd (vx, wz) | run | survival @60 s | vx | yaw error | mean yaw rate |
|---|---|---|---|---|---|
| (-0.10, 0) | r7 | 0.965 | -0.023 | 0.38 | +0.00 |
| (-0.10, 0) | r8 | 0.952 | -0.027 | 0.33 | +0.05 |
| (0, 0) | r7 | 0.986 | +0.011 | 0.40 | +0.04 |
| (0, 0) | r8 | 0.984 | +0.007 | 0.32 | +0.05 |
| (+0.10, 0) | r7 | 0.992 | +0.049 | 0.47 | +0.10 |
| (+0.10, 0) | r8 | 0.984 | +0.042 | 0.37 | +0.05 |
| (0, -0.30) | r7 | 0.996 | +0.011 | 0.44 | -0.05 |
| (0, -0.30) | r8 | 0.987 | +0.010 | 0.37 | -0.10 |
| (0, +0.30) | r7 | 0.993 | +0.011 | 0.42 | +0.14 |
| (0, +0.30) | r8 | 0.983 | +0.004 | 0.34 | +0.19 |

Yaw error falls again (about 0.42 to 0.34) and the yaw slope rises from about
0.31 to about 0.48. Survival stays at 95% to 99%. Forward slope does not
improve (about 0.35, was 0.36): weight 2 on the linear term plateaued.

## Round 9: linear forward cost weight 4

`microduck_basketball_ft8_flat`, `model_11992`, 500 iters from round 8. The
logged lin_vel_l1 ends near -0.40, so the mean forward error is still about
0.10 m/s (the weight doubled and the cost doubled). Clean eval:

| cmd (vx, wz) | run | survival @60 s | vx | yaw error | mean yaw rate |
|---|---|---|---|---|---|
| (-0.10, 0) | r8 | 0.952 | -0.027 | 0.33 | +0.05 |
| (-0.10, 0) | r9 | 0.971 | -0.021 | 0.33 | +0.09 |
| (0, 0) | r8 | 0.984 | +0.007 | 0.32 | +0.05 |
| (0, 0) | r9 | 0.990 | +0.026 | 0.35 | +0.06 |
| (+0.10, 0) | r8 | 0.984 | +0.042 | 0.37 | +0.05 |
| (+0.10, 0) | r9 | 0.980 | +0.067 | 0.41 | +0.06 |
| (0, -0.30) | r8 | 0.987 | +0.010 | 0.37 | -0.10 |
| (0, -0.30) | r9 | 0.987 | +0.025 | 0.39 | -0.10 |
| (0, +0.30) | r8 | 0.983 | +0.004 | 0.34 | +0.19 |
| (0, +0.30) | r9 | 0.991 | +0.027 | 0.34 | +0.23 |

Forward slope rises from about 0.35 to about 0.44, and +0.10 m/s is answered
with +0.067 m/s. Survival holds. Yaw slope rises from about 0.48 to about
0.55; yaw error stays near 0.35. The zero-command forward drift grows from
+0.007 to +0.026 m/s. Another weight increase is the wrong next step: the
centered reward (std 0.04 m, per-step reward about 1.8 of a weight-2 term)
keeps the duck within about 1 cm of the ball centre, which is what rolling
has to give up.

## Round 10: wider centering kernel (kept)

`microduck_basketball_ft9_flat`, `model_12491`, 500 iters from round 9, centered
std 0.08 m. The policy stayed on the peak (centered reward still about 1.92).
Clean eval versus round 9: forward slope 0.44 to 0.48, yaw slope 0.55 to 0.58,
yaw error slightly down, survival unchanged at 98% to 99%. A small gain, and
the best checkpoint.

## Round 11: centering weight 0.5 (rejected)

`microduck_basketball_ft10_flat`, `model_12990`, 500 iters from round 10.
Centered weight 2.0 to 0.5. Training falls rose from about 6 to about 24 per
log window. Clean eval survival at 60 s is 0.00 to 0.15. Not kept. The
centering reward holds the duck on the ball; cutting it does not buy tracking.

## Playground-protocol eval (round 10, model_12491)

`scripts/eval_basketball_playground.py`: 1024 envs, 60 s, random commands from
the owner's training ranges (+-0.30 m/s forward, +-0.20 lateral, +-1.0 rad/s
yaw) resampled as in training (3 to 8 s), free ball, first fall counts as
failure. The resample interval and the exact push protocol of playground b11
are assumptions here; pushes are +-0.09 m/s at 0.5 to 1.0 s, as recorded for b11.

| condition | survival 10 s | survival 60 s | forward MAE (zero / constant) | yaw MAE (zero / constant) |
|---|---|---|---|---|
| pushes 0.5-1 s | 0.52 | 0.035 | 0.106 (0.144 / 0.092) | 0.595 (0.506 / 0.606) |
| pushes 3-6 s (training) | 0.81 | 0.32 | 0.095 (0.145 / 0.078) | 0.508 (0.503 / 0.531) |
| no pushes | 0.84 | 0.43 | 0.094 (0.147 / 0.076) | 0.492 (0.500 / 0.517) |

Fixed-command eval (earlier tables) gives 98% to 99% at 60 s, so the gap is
mostly the random command mix (lateral and large yaw), not the pushes.
Playground b11 reports 97% at 60 s under its protocol. We are far below it.

The MAEs do not show tracking: the constant predictor (the run mean of vx and
wz) beats the policy on both, and the yaw MAE is worse than predicting zero.
The playground's 0.156 forward MAE and 1.26 yaw MAE should not be compared
with these numbers without matching the baselines.

## Playground-protocol ablation (round 10, no pushes, 1024 envs, 60 s)

Each row enables one part of the random command mix. "resampled" uses the
owner's training resample (3 to 8 s); "fixed" holds one command per episode.

| command mix | survival 60 s |
|---|---|
| forward +-0.10, resampled | 0.99 |
| forward +-0.15, resampled | 0.98 |
| forward +-0.30, fixed per episode | 0.95 |
| yaw +-1.0 only, resampled | 0.985 |
| lateral +-0.2 only, resampled | 0.94 |
| forward +-0.30 only, resampled | 0.67 |
| all three, fixed per episode | 0.89 |
| forward +-0.30, lateral +-0.2, yaw +-0.3, resampled | 0.45 |
| all three ranges, resampled (full training mix) | 0.43 |

Forward magnitude alone is survivable (0.95 fixed). Command switching at a
small range is survivable (0.98). The failure is both together: large forward
commands that change every few seconds. Lateral and yaw add loss on top of that
but are not the cause. The fixed-command eval in the earlier tables (+-0.10 m/s)
sits in the safe region, which is why it read 98% to 99%.

## Round 12-13: forward-command curriculum (did not help)

Stage 1 (`microduck_basketball_ft11_flat`, `model_12990`, 500 iters from round 10):
forward +-0.20 with switching. At its own range: survival 0.70 at 60 s,
round 10 at the same range: 0.72. Not an improvement.

Stage 2 (`microduck_basketball_ft12_flat`, `model_13489`, 500 iters from stage 1):
forward +-0.30 restored, lateral +-0.2, yaw +-1.0. Playground protocol:
full (pushes 0.5-1 s) 0.017 at 60 s; no pushes 0.277. Round 10: 0.035 and 0.434.
Fixed +-0.30 per episode, no pushes: 0.913 (round 10: 0.95).

Neither stage changed the switching cliff. Stage 1 is also worse than round 10
at the full +-0.30 range, which is expected since it never trained there.

Candidate cause, not yet tested: the centered reward (std 0.04 m, weight 2)
pays for the root sitting over the ball centre, but rolling the ball forward
requires a fore-aft offset that grows with the commanded speed. Round 11 showed
that removing centering collapses balance, so the reward is needed, but a
fixed centre target may conflict with a large commanded speed. A probe of the
offset during +-0.30 switching would test it before any training.

## Open items

- Round 4 has no re-rendered video; the earlier Round 2 clips are the latest
  visual reference.
- Forward MAE is not discriminative at this command range. A constant
  0.05 m/s predictor gives MAE about 0.154 on commands uniform in +-0.30 m/s,
  which matches playground b11's reported 0.156.
- The round table above predates the eval fix; its vx column includes the 20%
  forced-forward envs. Round 4 and 6 clean numbers are in the Round 6 table.
