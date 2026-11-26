# MotrixSim cross-check for the motion examples

Training is MuJoCo only and MotrixSim is held out, as for the sprint gait
(`scripts/sim2sim_check.py`). `scripts/sim2sim_tricks.py` runs each motion
example's checkpoint on both backends with the same protocol, scores one
headline metric itself, and passes when Motrix is no more than 10 points below
MuJoCo.

Result: five of eight pass (roulade, tumble, ball kick, scoot, stilts).
Sit/stand, the dribble and basketball do not.

## Method

To run an owner on Motrix the script registers its task on the backend and
removes what Motrix cannot run or does not accept:

- the four events the walking Motrix owners drop: interval push,
  joint-armature, geom-friction and body-inertia randomization;
- the roulade contact sensors (`subtree1` with `geom2`; Motrix wants
  `subtree2` with `subtree1`), by using the plain task fragment;
- every reward term, which an evaluation does not use;
- the basketball hold action, a body wrench Motrix lacks. On the free ball it
  is zero anyway.

Commands are zero unless the protocol sets them; pushes are off on both
backends. 128 envs, seeds 101 / 202 / 303.

## Result

| Example | Protocol | Headline metric | MuJoCo | Motrix | Pass |
|---|---|---|---:|---:|---:|
| roulade | standing start, 4.9 s | full turn and standing at the end | 94.5-95.3% | 99.2% | 3/3 |
| tumble | standing start, 4.9 s | full turn and standing at the end | 96.1-100% | 97.7-99.2% | 3/3 |
| ballkick (right) | standing start, 4 s | ball past 0.5 m and standing at the end | 100% | 94.5-99.2% | 3/3 |
| scoot | seated, backward 0.10 m/s, 10 s | survived and backed up ≥ 0.05 m/s | 100% | 95.3-98.4% | 3/3 |
| stilts | 0.15 m/s, 60 s | survival | 99.2-100% | 99.2-100% | 3/3 |
| sitstand | stand 1 s, sit until 7 s, stand until 12 s | seated at 6.5 s and standing at the end | 81.2-89.1% | 63.3-73.4% | 0/3 |
| kick (dribble) | 12 s | ball past 0.5 m along the goal, not fallen | 56.2-57.8% | 18.0-23.4% | 0/3 |
| basketball | free ball, still, 60 s | survival | 99.2-100% | 7.8-10.9% | 0/3 |

The three that fail all depend on sustained contact with something other than
the soles under load:

- Sit/stand sits and rises on the seat.
- The dribble pushes a ball with the soles at every step.
- Basketball balances on a ball. On Motrix about 90% of envs fall off within
  60 s; most are back on the ball by the end after a reset, so the
  standing-at-end share stays high (95%).

The dynamic tricks that pass (roulade, tumble, kick) touch the floor or the
ball only briefly.

These numbers are not the protocol of each example's own battery (zero
commands, pushes off, 128 envs), so they are below or above those batteries
in places. The MuJoCo column is the reference.

```bash
uv run --no-sync scripts/make_hull_robot.py
uv run --no-sync scripts/make_basketball_scene.py
uv run --no-sync scripts/make_ballkick_scene.py
uv run --no-sync scripts/make_stilt_robot.py --height-cm 10 --blend 0.5
uv run --no-sync scripts/sim2sim_tricks.py --num-envs 128 --seed 101
```

## What this does and does not clear

Passing means the motion does not lean on a MuJoCo-specific detail that
Motrix lacks. It is the gate the walking gait passed before it became the
hardware starting point, and none of these has run on a robot. The three
failures should not go on hardware before a fix: training with contact
randomization the way `docs/audits/motrix_topspeed.md` did for the sprint, or
a Motrix-specific look at the seat and ball contacts.
