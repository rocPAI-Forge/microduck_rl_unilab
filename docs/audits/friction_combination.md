# Friction randomization below the floor's friction never took effect

Found while diagnosing scoot (round 3), 2026-11-17. Not yet fixed for the
walking tasks; recorded here so the next sprint retrain can take it on.

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

## Open

For the walking tasks the same change widens the effective training
distribution, so it needs a sprint retrain, the Motrix gate, and a check of
how MotrixSim combines friction. That is a separate change.
