# Friction randomization below the floor's friction never took effect

Found while diagnosing scoot (round 3), 2026-11-17. Fixed for scoot; for the
walking tasks the low-friction margin was measured and a retrain tried
(2026-11-27, below): `sprint_stand/model_59000` stays the hardware start.

## What happens

Every MicroDuck scene's floor (`scene_flat*.xml`) has MuJoCo's default
friction 1.0 and priority 0, the same priority as the robot geoms. MuJoCo
combines two equal-priority geoms by taking the larger friction, so a floor
contact grips at `max(robot geom friction, 1.0)`:

| Owner family | Configured foot friction | Effective on MuJoCo |
|---|---|---|
| velocity, velstand, standup, sitstand, sprint (most) | 0.7–1.3 | 1.0–1.3 |
| `microduck_sprint_standwide_flat` and descendants | 0.6–1.5 | 1.0–1.5 |
| scoot rounds 1–3, seat/body geoms | 0.05–1.0 curriculum | 1.0 throughout |

So no MuJoCo-trained policy has seen a floor contact below 1.0. Separately,
in the top-speed sweep
([`motrix_topspeed.md`](motrix_topspeed.md)), a pinned foot friction of 0.4
most likely ran at 1.0; the 1.8 row is unaffected. How MotrixSim combines
friction has not been checked.

## Fix (applied to scoot only)

Raising the robot collision geoms to contact priority 1 makes their own
friction govern floor contacts; floor and robot geoms share
solref/solimp/condim, so nothing else changes. `scripts/make_hull_robot.py`
writes such a scene (`scene_flat_groundcontact_bam_hull_prio.xml`), used by
`microduck_scoot_sym_flat`; `tests/test_hull_robot.py` checks that a 0.1
robot friction survives contact with the floor.

## Walking tasks (2026-11-27)

`scripts/make_prio_walk_robot.py` writes the walking model with its collision
geoms at priority 1 (`generated/scene_flat_bam_prio.xml`);
`tests/test_prio_walk_robot.py` checks that a 0.1 foot friction reaches the
contact there and not on the plain model. On that scene the foot friction
event takes effect, so fixed-friction batteries measure what a floor of that
friction does.

### How far the current policy reaches

`sprint_stand/model_59000` on the official straight protocol
(`microduck_sprint_straightspeed_flat/mujoco`, 60 s, 256 environments,
seed 777, `--clip-targets`, scene and friction overridden):

| Foot friction | Stand | 0.3 m/s | 0.6 m/s |
|---|---|---|---|
| 1.0 | 100% | 100% | 100% |
| 0.7 | 100% | 100% | 100% |
| 0.5 | 99.6% | 100% | 100% |
| 0.3 | 99.6% | 91.8% | 38.3% |
| 0.2 | 64.8% | 0% | 0% |
| 0.1 | 12.9% | — | 0% |

Speed tracking holds at 0.5 and above (0.29 and 0.55-0.57 m/s). The policy
trained at an effective 1.0 already covers floors down to 0.5; walking breaks
near 0.3 and standing near 0.2.

### Retrain on 0.4-1.3

`microduck_sprint_stand_prio_flat` resumed `model_59000` on the priority scene
with foot friction 0.4-1.3, 3000 iterations. `model_61999`:

| Check | `model_59000` | `model_61999` |
|---|---|---|
| 0.3 friction, 0.6 m/s | 38.3% | 84.4% (87.3% on 512 envs, seed 778) |
| 0.3 friction, 0.3 m/s | 91.8% | 99.6% |
| 0.2 friction, stand | 64.8% | 93.8% |
| 0.5-1.0 friction, 0-0.6 m/s | 99.6-100% | 99.2-100% |
| 1.0 friction, 0.9 m/s (512 envs, seed 778) | 100% | 97.9% |
| Trunk height standing | 0.106 m | 0.080 m |
| MotrixSim gate, stand 300 s | pass | 41% (fail) |
| MotrixSim gate, 0.3 / 0.6 m/s, turns | pass | pass |

It learned to stand crouched, 2.6 cm lower with the head about 10° further
down; walking height is unchanged (0.12-0.125 m). The crouch is what fails the
MotrixSim standing gate. Every late checkpoint (61000-61999) loses about 2% at
0.9 m/s on high friction.

A second stage (`microduck_sprint_stand_prio_pose_flat`) added
`standup_terms.standing_height_l1_penalty` (L1 on trunk height towards
0.106 m, stand commands only, weight 30; `leg_pose` at 1.0 saturated and did
nothing). After 2000 iterations it stood at 0.092 m, with stand survival at
1.0 friction down to 95.3% and the low-friction gains gone (0.3 friction,
0.6 m/s: 68%; 0.2 friction, stand: 66%).

### Outcome

Low-friction robustness and the upright stand traded against each other in
both stages, so neither checkpoint was published and `model_59000` stays the
hardware start. It is fine on floors of friction 0.5 and above. For a floor
below about 0.4, a stand height command (so a crouch is asked for rather than
learned) is the next thing to try. How MotrixSim combines friction is still
unchecked.

`scripts/eval_sprint_speed.py` changes made along the way: `high_speed_fraction`
and `straight_fraction` are zeroed with the other command fractions (on the
`sprint_stand` owner 30% of environments had been re-sampled to 0.75-1.0 m/s),
and `trunk_height_m` is reported.
