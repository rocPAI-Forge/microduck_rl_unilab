# Ball kicking: a dribble on the sprint gait

Result: the duck walks to a free ball and dribbles it along a goal direction
without falling; it does not strike the ball. A real kick from a standing start
is upstream's policy (below and `examples/ballkick/`). Checkpoint
`examples/kick/model_60499.pt`, owner `microduck_kick_flat/mujoco`.

## Design

Relearning a gait for this task was not worth it, so the published
`sprint_stand` policy is the warm start and keeps walking through its own
rewards. A scripted command does the navigation:

- `BallPursuitCommand` (`kick_terms.py`) replaces the twist command each step.
  It aims at a point 12 cm behind the ball on the goal line until the duck is
  on that line behind the ball, then at a point 30 cm past the ball along the
  goal. Forward speed is `min(1.5 d, 0.3)` scaled by `cos` of the heading
  error, and yaw rate is `clip(2.0 e, +-0.8)`.
- The spawn-heading loop is off, which also zeroes the spawn-heading rewards
  (they are gated on it). The other heading costs are against the command, so
  they allow the pursuit turns.
- `reset_kick` replaces `reset_base`. It places the duck, then the ball
  0.2-0.45 m ahead within +-30 deg of its heading, and a goal direction within
  +-30 deg of its heading. The ball is placed in the same event because reading
  the heading back during reset would return the pre-reset yaw.
- Observation: ball xy relative to the root and the goal direction, in the
  heading frame, in the head-pose command slot (zero in `sprint_stand`). The
  warm start's normalizer has a std of 0.004-0.018 on that slot and is frozen
  (5.8 billion samples), so the block is scaled by 0.05 / 0.02 to land near
  +-1 after normalization.
- Rewards: ball velocity along the goal, clipped to [0, 1] m/s (weight 2), and
  ball velocity across it (weight -0.5), on top of the `sprint_stand` set.
- Scene: `scene_flat_bam_kick.xml`, the walking model plus a 3.5 cm, 45 g ball
  with rolling friction (condim 6). 2048 envs: the walking model at 4096 runs
  out of memory on this machine.

## Result

`scripts/eval_kick.py`, 256 envs, 12 s, pushes off, three seeds:

| Metric | No training (pursuit command on `sprint_stand`) | 1500 iterations |
|---|---:|---:|
| Survival | 66-71% | 99.6-100% |
| Touched | 90-95% | 98-99% |
| Ball along goal, median over all envs | 1.06-1.55 m | 1.02-1.41 m |
| Ball past 0.5 m | 57-65% | 57-68% |
| Ball across goal, mean | 0.37-0.46 m | 0.22-0.23 m |

The untrained gait already pushes the ball 2.2-2.3 m (median over survivors),
but about a third of the envs fall: stepping on the ball, or turning sharply.
Training kept the distance, removed the falls and halved the sideways error.
The video shows the ball staying at the feet as the duck walks.

## Why it does not strike

In the walking model only the two foot soles have collision geoms. The shins
and the top of the foot pass through the ball, so a swing cannot hit it; the
sole can only push it during a step. Upstream microduck_rl's `BallKick` uses
its ground-contact model (legs collide) with a 70 mm, 15 g ball and a
ball-blind actor. A real kick here would need that model, which the
`sprint_stand` gait was not trained on.

## A striking kick: upstream's policy

Upstream's published `ball_kick_right/left.onnx` run in a matching scene here
(`microduck_ballkick_flat`, `scripts/make_ballkick_scene.py`: the
ground-contact hull duck and a 70 mm, 15 g ball 9 cm ahead of the kicking
foot), unchanged. 256 envs, 4 s, three seeds: kicked 99.2-100%, peak ball speed
1.28-1.33 m/s, the ball 4 m forward and 0.5-0.8 m sideways, the duck standing
at the end 98-100%. Converted with `scripts/onnx_to_rsl_rl.py`, they are
published as [`examples/ballkick/`](../../examples/ballkick/). The dribble
above stays: it walks to the ball, the kick does not.

## Open
- Goal directions beyond +-30 deg of the spawn heading, and walking around a
  ball that is on the wrong side (the pursuit target can pass through the ball).
