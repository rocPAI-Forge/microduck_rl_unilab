# Forward roll: tumble and roulade

Two forward-roll policies came out of this work, both simulation only:

- **Roulade** (`examples/roulade/model_999.pt`): upstream microduck_rl's
  forward roll over the flat top of the head, landing on the feet. Fine-tuned
  from upstream's published policy; our own training did not reach it.
- **Tumble** (`examples/tumble/model_17997.pt`): a fast whole-body tumble that
  ends standing, trained here from the standup policy.

Evaluation for both: `scripts/eval_roll.py`, 512 envs, standing starts, 4.9 s,
three seeds. Roll complete means the rotation maximum passed 260° (for the
roulade also with the head-top latch).

## Tumble

`microduck_roll_flat` (round 1): standup warm start, paid for forward pitch
rate about the episode's lateral axis up to one full turn, then for being done
and still; all standing rewards off. After 1500 iterations every env turned a
full circle in about 0.8 s and lay curled; switching to the standup policy at
2 s got 100% standing 0.84 s later. It is a whip over the head and back, not
upstream's roulade: rotation counted whether or not the duck touched the floor
and at any speed.

`microduck_tumble_flat`: the same rewards plus stand-up terms gated on the
completed turn (standing composite, linear upright, upward velocity, a tax on
every step below standing height), 1500 iterations from round 1. Result: full
turn 100% at 0.56 s, standing 100% from about 0.8 s, peak pitch rate about
9.8 rad/s.

## Roulade: porting upstream

`roulade_terms.py` ports upstream's fifth-run design
(`microduck_roulade_env_cfg.py`, develop):

- the rotation accumulator integrates the body-frame pitch rate only while
  some geom touches the floor and the lateral axis is near horizontal;
- progress pays new maxima, at most 5 rad/s, with a tax above 7 rad/s;
- the landing terms open after 260° and only once the flat top of the head
  touched the floor between 20° and 170° (contact sensors in
  `assets/robots/microduck/roulade_task.xml`);
- half the episodes start mid-roll (50-340°, tucked, with forward spin).

| Attempt | Start | Iterations | From standing: rotation max | Head-top latch | Complete |
|---|---|---|---|---|---|
| 1 | standup policy | 3000 | 147° | 0% | 0% |
| 2 | attempt 1, plus roll-back and stall costs (not upstream) | 1500 | 148° | 0% | 0% |
| 3 | scratch, upstream recipe | 6000 (stopped) | 160° | 20-37% | 0% |

All three learned the second half: from 180-260° mid-roll starts they
completed 100% and stood up to 87%. From standing they never got over the head.
The standup warm start face-planted, lay prone at about 147° with the legs up,
fell back and did standup's prone push-up; attempt 2's costs only turned that
into lying prone. From scratch the duck learned to tuck its chin (latch 100% at
1500 iterations) but still stalled at 150-170°, and the progress reward stayed
flat at 0.50-0.58 from iteration 1000 to 6000.

## Is it the physics or the training?

- Convex-hull collision (`make_hull_robot.py`, 128 vertices) is not it: MuJoCo
  collides meshes as convex hulls anyway, and the attempt-3 policy behaves the
  same on the original mesh (rotation max 160° on both).
- Upstream's published `roulade.onnx`
  ([pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies),
  revision `236d1835`) run in our simulator through the same 61-D observation
  and 14-D action contract: 93% past 260° within 1 s, 100% within 4.9 s,
  99.6-100% standing at the end, on both collision models.

So the simulator supports the full roulade and the gap is our training. The
published policy crouches, tucks, plants the head and kicks the legs up into a
headstand before rolling over the back; none of our runs found that kick.
Upstream's manifest runs this policy for 1 s and then hands over to its walking
policy, and does not record the run that produced it, so we cannot tell
whether the develop recipe alone reaches it.

## Roulade: fine-tuning upstream's policy

`scripts/onnx_to_rsl_rl.py` turns the ONNX into an rsl_rl checkpoint (same
network, normalizer recovered from the baked divisor, output matches to 1e-6),
with the critic from attempt 3. 1000 iterations of `microduck_roulade_flat` at
learning rate 1e-4:

| Metric | Upstream policy in our sim | Fine-tuned `model_999` |
|---|---:|---:|
| Roll complete | 98.4% | 100% |
| Time to complete | 1.00 s | 0.86 s |
| Rotation maximum p90 | 969° (keeps rolling) | 353° |
| Peak pitch rate | 8.9 rad/s | 7.9 rad/s |
| Standing at the end | 81.6% | 100% |

The fine-tune keeps the motion and adds what upstream's 1 s hand-off did not
need: stop after one roll and stand.

## Open

- Our own training still does not find the headstand kick. A teacher (the
  published policy) for an imitation reward or distillation is the next thing
  to try if an independently trained roulade matters.
- Neither policy has a MotrixSim check or a hardware run. The roulade plants
  the head on the floor, and the tumble peaks near 10 rad/s.
