# Hardware bring-up

The checklist to run before a policy first moves the robot. The software side
is already checked: `scripts/deploy_sim_loop.py` drives the
[reference runtime](DEPLOY.md#reference-runtime) closed-loop in MuJoCo and
MotrixSim and shows it reproduces the training contract to 1e-6. What
simulation cannot see is the robot side: IMU axes and signs, servo IDs, zero
offsets, and motor directions. A mistake in any of them looks like a broken
policy. This page and `scripts/hardware_bringup.py` catch them first.

## Before you start

- **Servos (XL330).** Position control mode. Goal-position limits set to the
  `soft_joint_limits_rad` in `deploy_manifest.json`; the policy was trained
  with that clamp. Each servo's zero matches the simulator's zero: with every
  joint at its manifest `default_joint_pos_rad`, the robot is in the home pose.
- **Supply.** 6.5–8.2 V at the servos, the voltage range the servo model was
  trained on. Below 6.5 V, stop.
- **IMU.** Mounted on the trunk. The adapter reports the body frame the policy
  uses: x forward, y left, z up; gyro in rad/s; gravity as the unit vector
  pointing down, so an upright, still robot reads `[0, 0, -1]`.
- **Onboard software.** numpy and onnxruntime. Copy
  `src/microduck_rl_unilab/deploy/runtime.py` and the `deploy/` folder
  (`policy.onnx`, `deploy_manifest.json`). `bringup.py` is needed only for the
  checks.
- **Safety.** A harness or stand that keeps the robot from falling, and a way
  to cut torque at once.

## The adapter

The checks and the runtime talk to the robot through four calls, joints in the
manifest order (`left_hip_yaw … right_ankle`):

```python
class MyRobotIO:
    def read(self) -> SensorReading: ...        # gyro_b, gravity_b, joint_pos (rad, absolute), joint_vel
    def heading(self) -> float: ...              # yaw, rad, for the heading hold
    def write_goals(self, goals) -> None: ...    # goal positions, rad
    def set_torque(self, enabled: bool) -> None: ...

def make_io() -> MyRobotIO: ...
```

## Run the checks

```bash
uv run --no-sync scripts/hardware_bringup.py examples/sprint_stand/deploy --io my_robot.io:make_io
```

The script walks through the checks below in order, prompts for each setup,
stops at the first failure, turns torque off at the end, and writes
`bringup_<time>.json`. `--checks imu_static,imu_axes` runs a subset.

| # | Check | Setup | Passes when |
|---|---|---|---|
| 1 | `imu_static` | Torque off, upright, level, still | \|gravity\| within 5% of 1, tilt ≤ 5°, \|gyro\| ≤ 0.05 rad/s |
| 2 | `imu_axes` | Torque off, robot in your hands | Each prompted motion moves the expected axis with the expected sign (table below) |
| 3 | `default_pose` | Torque off, robot posed in the home pose | Every joint reads within 0.10 rad of its default |
| 4 | `joint_directions` | Suspended, feet off the ground; torque on | Each 0.15 rad goal step is followed by that joint only, and you confirm the motion matches the table below |
| 5 | `loop_timing` | Robot not moved | Read + inference: mean ≤ 15 ms, p99 ≤ 20 ms, leaving room for the write in the 20 ms period |
| 6 | `policy_stand` | Feet on the ground, harness on; torque on | The policy stands 5 s, the trunk never tilts past 20°, and no joint sits at its limit more than 20% of the time. Past 20° it stops and returns to the default pose |

### IMU motions

| Prompt | Expected |
|---|---|
| Tilt nose-down about 30° | gravity x > +0.3 |
| Tilt right side down about 30° | gravity y < −0.3 |
| Turn left (counter-clockwise from above) | gyro z > +0.3 rad/s |
| Roll right side down | gyro x > +0.3 rad/s |

### Joint motions

What a positive goal step looks like, from the MJCF joint axes. Look at the
robot from behind, so its left is your left.

| Joint | Positive goal |
|---|---|
| `left_hip_yaw` | left leg rotates clockwise seen from above (toe turns inward) |
| `left_hip_roll` | left foot swings outward, away from the body |
| `left_hip_pitch` | left leg swings backward |
| `left_knee` | left shin swings forward |
| `left_ankle` | left toe tips down |
| `neck_pitch` | neck tilts back, head moves backward |
| `head_pitch` | head nods down |
| `head_yaw` | head turns to its left |
| `head_roll` | head tilts toward its left |
| `right_hip_yaw` | right leg rotates clockwise seen from above (toe turns outward) |
| `right_hip_roll` | right foot swings inward, toward the body |
| `right_hip_pitch` | right leg swings forward |
| `right_knee` | right shin swings backward |
| `right_ankle` | right toe tips up |

The encoders cannot catch two faults, so the by-eye confirmation matters:

- **A reversed motor.** The goal and the reading agree, but the joint turns
  the other way from the simulator.
- **Two swapped servo IDs whose joints share a default.** The two hip yaws
  both default to 0, so the default-pose check passes and each ID reads back
  what it was sent; only the wrong leg moving gives it away.

Swapped IDs with different defaults, such as the two hip pitches
(−0.458 and +0.458 rad), fail `default_pose`.

## First motion

After all six checks pass, keep the harness on and step up one command at a
time. Each step uses the runtime `Controller` with the commands in
[`DEPLOY.md`](DEPLOY.md#commands). Stop and cut torque at the first sign of
trouble.

1. `Command.stand()`, 30 s.
2. `Command.straight(0.3)`, 5 s, then stand.
3. `Command.straight(0.6)`, 10 s, then stand. 0.6 m/s is the top of the band
   the MotrixSim gate checked; do not go faster on the first day.
4. `Command.turn_around(1)` and `Command.turn_around(-1)`, 10 s each.

Record the 50 Hz readings, the commands, and the goal positions of every run.
Those logs are what the sim-to-real comparison and the next round of noise and
delay calibration (`sim2real_notes.py`) need.
