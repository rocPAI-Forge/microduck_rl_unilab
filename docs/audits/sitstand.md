# Commanded sit and stand on MuJoCo

**Result:** after five rounds, one policy sits and stands on command: on a
fixed stand → sit → stand schedule it sits 85–89% of the time, stands back up
97% of the time, stands upright (tilt p90 3.3°), and almost never falls
(≤ 0.2%, 512 environments, three seeds). It is not gentle at the end of the
descent: the trunk peaks at about 0.88 m/s, a drop of the last ~4 cm onto the
seat. Simulation only; no checkpoint is published (there is no MotrixSim owner
for this task yet, so the sim2sim gate cannot run).

## Recipe

After upstream pollen-robotics/microduck_rl `microduck_sitstand_env_cfg.py`
(develop; their real-robot policy). `microduck_sitstand_flat/mujoco` builds on
the standup owner (same BAM ground-contact robot, observations, DR, head
command, CoM and action-rate curricula):

- the twist command is `SitStandCommand`: `[sit_flag, 0, 0]`, dwell 3.5–6.5 s,
  with an internal target blend slewed over 2 s;
- `sitstand_terms` track that moving target: leg pose (Gaussian + L1), trunk
  height (two Gaussians + L1), a multiplicative height × upright × pose × head
  score, stillness at the commanded height once the ramp is done, a rise
  bootstrap, descent and rise speed caps, and an upright-while-tall term;
- resets are 50% standing, 50% seated; episodes are 12 s; pushes start only
  after iteration 1000.

The ground-contact model takes about 30 MB per environment here (17 GB at 512,
31 GB at 1024), so 2048 environments run out of the machine's 59 GB; every
round used 1024 (upstream: 4096).

## Rounds

Scored by `scripts/eval_sitstand.py` (stand 1 s, sit until 7 s, stand until
12 s; success = within 12 mm of the target height and under 25° tilt).

| Round | Owner | Start | Iterations | Sit | Rise | Falls | Standing tilt p50 |
|---|---|---|---|---|---|---|---|
| 1 | `microduck_sitstand_flat` | scratch | 3000 | 82% | 0% | 18% | 31.7° |
| 2 | `microduck_sitstand_flat` | `examples/standup` | 2500 | 0% | 100% | 0% | 2–3° |
| 3 | `microduck_sitstand_warm_flat` | `examples/standup` | 2000 | 0% | 100% | 0% | 2° |
| 4 | `microduck_sitstand_fix_flat` | round 1 | 2000 | 32% | 45% | 23% | 3° (p90 29°) |
| 5 | `microduck_sitstand_fix_flat` | round 4 | 2000 | **91%** | **99%** | **0%** | **1.2°** |

- **Round 1** sat but stood in a tripod: left knee on its −90° limit, neck at
  −110° with the head on the floor, trunk ~31°. The multiplicative score fell
  to ~0 early and the remaining terms still paid for the tripod.
- **Rounds 2–3** started from the checked standup policy and never tried to
  sit; a capped descent bootstrap (`posture_sit_bootstrap`, round 3) and a
  heavier wrong-posture height cost did not change that.
- **Rounds 4–5** resumed round 1 with the standup task's height-gated sharp
  upright Gaussian, a 5× joint-limit cost, and heavier head tracking. Round 4
  swung between sitting well and standing upright; round 5 converged at
  `model_6500`.

## `model_6500`, 512 environments

| Seed | Sit | Seated hold | Rise | Falls | Peak descent p95 | Peak rise p95 | Tilt p90 |
|---|---|---|---|---|---|---|---|
| 101 | 87.9% | 87.7% | 97.1% | 0% | 0.87 m/s | 0.43 m/s | 3.4° |
| 202 | 89.3% | 89.2% | 97.5% | 0% | 0.88 | 0.43 | 3.3° |
| 303 | 85.2% | 85.2% | 97.3% | 0.2% | 0.88 | 0.43 | 3.3° |

## Gentler descent: two attempts, both stopped sitting

Both resumed `model_6500` for 1500 iterations; 256 environments, seed 101.

| Owner | Change | Sit | Rise | Peak descent p95 |
|---|---|---|---|---|
| — (`model_6500`) | — | 84–85% | 97–98% | 0.88 m/s |
| `microduck_sitstand_gentle_flat` | descent cap 30 → 60, \|a_z\| 0.3 | 0.4–23% | 98–100% | 0.12–0.87 |
| `microduck_sitstand_ramp_flat` | target ramp 3.5 s, descent cap 15 → 30, \|a_z\| 0.15, dwell 5–8 s | 2–6% | 98–100% | 0.15–0.92 |

Whenever sitting got costlier or slower, the policy stopped sitting within
250 iterations and kept standing; the low peak speeds are checkpoints that no
longer sit. The drop is part of the only sit this policy found. `model_6500`
stays the result.

## Open

- The final ~4 cm of the descent is a drop. Heavier speed and shock costs and
  a slower target made the policy stop sitting rather than sit gently; an
  impact-force term, or training from scratch with gentleness from the start
  on 4096 environments, are the remaining routes.
- Upstream's seated fall-back hardening (head-command jerks, backward tip
  kicks, neck-strain cost) is not ported.
- No MotrixSim owner; the scoot-to-dock task (microduck-playground butt walk)
  builds on this.

## Reproduce

```bash
uv run --no-sync microduck-train --algo ppo --task microduck_sitstand_flat --sim mujoco \
  algo.num_envs=1024 algo.max_iterations=3000 algo.algo_log_name=sitstand_r1
uv run --no-sync microduck-train --algo ppo --task microduck_sitstand_fix_flat --sim mujoco \
  algo.num_envs=1024 algo.max_iterations=2000 algo.load_run=<round 1 run> algo.checkpoint=2999
uv run --no-sync microduck-train --algo ppo --task microduck_sitstand_fix_flat --sim mujoco \
  algo.num_envs=1024 algo.max_iterations=2000 algo.load_run=<round 4 run> algo.checkpoint=4998
uv run --no-sync scripts/eval_sitstand.py <round 5 run>/model_6500.pt --num-envs 512 --seed 101
```
