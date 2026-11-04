"""Hardware bring-up checks for the onboard runtime.

Simulation proves the runtime reproduces the training contract
(``scripts/deploy_sim_loop.py``); it cannot see the robot side. These checks
do, before the first policy step. They run against any object that implements
``RobotIO`` and use numpy only. Order and setup are in
``docs/HARDWARE_BRINGUP.md``:

1. ``check_imu_static``: upright and still, gravity near (0, 0, -1), gyro near 0.
2. ``check_imu_axes``: the operator tilts and turns the robot as prompted; the
   gravity and gyro signs must follow the body frame (x forward, y left, z up).
3. ``check_default_pose``: the robot held in the default pose reads the
   manifest defaults, which catches zero offsets, and swapped servo IDs whose
   joints have different defaults.
4. ``check_joint_directions``: suspended, each joint is stepped by a small
   goal; its reading must follow and no other joint may move, and the
   operator confirms the motion by eye. Only the eye catches a reversed motor
   or a swapped ID between joints with equal defaults, because the encoder
   reads back whatever its own motor was sent.
5. ``check_loop_timing``: read, infer, and write fit the 20 ms period.
6. ``check_policy_stand``: on the ground with a harness, the policy stands for
   a few seconds without tilting or pinning a joint at its limit.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from microduck_rl_unilab.deploy.runtime import (
    Command,
    Controller,
    DeployManifest,
    SensorReading,
)


class RobotIO(Protocol):
    """The hardware adapter the checks drive. Joints are in manifest order."""

    def read(self) -> SensorReading: ...

    def heading(self) -> float: ...

    def write_goals(self, goals: np.ndarray) -> None: ...

    def set_torque(self, enabled: bool) -> None: ...


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


Prompt = Callable[[str], None]
Sleep = Callable[[float], None]


def _average_reading(io: RobotIO, samples: int, period_s: float, sleep: Sleep) -> SensorReading:
    readings = []
    for _ in range(samples):
        readings.append(io.read())
        sleep(period_s)
    return SensorReading(
        gyro_b=np.mean([r.gyro_b for r in readings], axis=0),
        gravity_b=np.mean([r.gravity_b for r in readings], axis=0),
        joint_pos=np.mean([r.joint_pos for r in readings], axis=0),
        joint_vel=np.mean([r.joint_vel for r in readings], axis=0),
    )


def check_imu_static(
    io: RobotIO,
    *,
    max_tilt_deg: float = 5.0,
    max_gyro_rad_s: float = 0.05,
    samples: int = 50,
    period_s: float = 0.02,
    sleep: Sleep = time.sleep,
) -> CheckResult:
    reading = _average_reading(io, samples, period_s, sleep)
    gravity = np.asarray(reading.gravity_b, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    tilt = math.degrees(math.acos(np.clip(-gravity[2] / max(norm, 1e-9), -1.0, 1.0)))
    gyro = float(np.max(np.abs(reading.gyro_b)))
    passed = abs(norm - 1.0) <= 0.05 and tilt <= max_tilt_deg and gyro <= max_gyro_rad_s
    return CheckResult(
        "imu_static",
        passed,
        f"|gravity|={norm:.3f} (want 1), tilt={tilt:.1f} deg (<= {max_tilt_deg}), "
        f"max |gyro|={gyro:.3f} rad/s (<= {max_gyro_rad_s})",
        {"gravity_b": gravity.tolist(), "gyro_b": np.asarray(reading.gyro_b).tolist()},
    )


# (prompt, signal, axis, expected sign). Body frame: x forward, y left, z up;
# gravity_b is the unit vector pointing down, expressed in that frame.
IMU_AXIS_STEPS: tuple[tuple[str, str, int, float], ...] = (
    ("Tilt the robot nose-down by about 30 degrees and hold it", "gravity", 0, 1.0),
    ("Tilt the robot so its right side is down by about 30 degrees and hold it", "gravity", 1, -1.0),
    ("Turn the robot left (counter-clockwise seen from above) steadily", "gyro", 2, 1.0),
    ("Roll the robot right side down steadily", "gyro", 0, 1.0),
)


def check_imu_axes(
    io: RobotIO,
    prompt: Prompt,
    *,
    min_gravity: float = 0.3,
    min_gyro_rad_s: float = 0.3,
    samples: int = 25,
    period_s: float = 0.02,
    sleep: Sleep = time.sleep,
) -> CheckResult:
    steps = []
    passed = True
    for message, signal, axis, sign in IMU_AXIS_STEPS:
        prompt(message)
        reading = _average_reading(io, samples, period_s, sleep)
        vector = np.asarray(reading.gravity_b if signal == "gravity" else reading.gyro_b)
        threshold = min_gravity if signal == "gravity" else min_gyro_rad_s
        value = float(vector[axis])
        dominant = int(np.argmax(np.abs(vector[:2] if signal == "gravity" else vector)))
        ok = sign * value >= threshold and dominant == axis
        passed &= ok
        steps.append(
            {"step": message, "signal": signal, "axis": "xyz"[axis], "value": value, "ok": ok}
        )
    failed = [s["step"] for s in steps if not s["ok"]]
    return CheckResult(
        "imu_axes",
        passed,
        "all axes and signs match the body frame" if passed else f"failed: {failed}",
        {"steps": steps},
    )


def check_default_pose(
    io: RobotIO,
    manifest: DeployManifest,
    *,
    tolerance_rad: float = 0.10,
    samples: int = 25,
    period_s: float = 0.02,
    sleep: Sleep = time.sleep,
) -> CheckResult:
    reading = _average_reading(io, samples, period_s, sleep)
    error = np.asarray(reading.joint_pos) - manifest.default_joint_pos
    bad = [
        f"{name} {err:+.3f}"
        for name, err in zip(manifest.joint_names, error, strict=True)
        if abs(err) > tolerance_rad
    ]
    return CheckResult(
        "default_pose",
        not bad,
        "every joint within "
        f"{tolerance_rad} rad of its default" if not bad else f"off by more than {tolerance_rad} rad: {bad}",
        {"error_rad": error.tolist()},
    )


# What a positive goal step looks like on the robot, from the MJCF joint axes
# in the trunk frame (robot facing away from the viewer, x forward, y left).
JOINT_POSITIVE_MOTION: dict[str, str] = {
    "left_hip_yaw": "left leg rotates clockwise seen from above (toe turns inward)",
    "left_hip_roll": "left foot swings outward, away from the body",
    "left_hip_pitch": "left leg swings backward",
    "left_knee": "left shin swings forward",
    "left_ankle": "left toe tips down",
    "neck_pitch": "neck tilts back, head moves backward",
    "head_pitch": "head nods down",
    "head_yaw": "head turns to its left",
    "head_roll": "head tilts toward its left",
    "right_hip_yaw": "right leg rotates clockwise seen from above (toe turns outward)",
    "right_hip_roll": "right foot swings inward, toward the body",
    "right_hip_pitch": "right leg swings forward",
    "right_knee": "right shin swings backward",
    "right_ankle": "right toe tips up",
}

Confirm = Callable[[str], bool]


def check_joint_directions(
    io: RobotIO,
    manifest: DeployManifest,
    *,
    confirm: Confirm | None = None,
    step_rad: float = 0.15,
    min_move_fraction: float = 0.5,
    max_cross_move_rad: float = 0.05,
    settle_s: float = 0.5,
    sleep: Sleep = time.sleep,
) -> CheckResult:
    """Step each joint alone; readings must follow, and ``confirm`` the motion by eye.

    The readings only prove command and encoder agree. Whether "positive" is
    the simulator's positive needs the operator: ``confirm`` gets the expected
    motion from ``JOINT_POSITIVE_MOTION`` (for a negative step, the opposite).
    """
    base = manifest.default_joint_pos.copy()
    io.set_torque(True)
    io.write_goals(base)
    sleep(settle_s)
    rows = []
    passed = True
    try:
        for index, name in enumerate(manifest.joint_names):
            low, high = manifest.joint_limits[index]
            step = step_rad if base[index] + step_rad <= high else -step_rad
            if base[index] + step < low:
                step = (high - base[index]) / 2.0
            before = np.asarray(io.read().joint_pos, dtype=np.float64)
            goals = base.copy()
            goals[index] += step
            io.write_goals(goals)
            sleep(settle_s)
            after = np.asarray(io.read().joint_pos, dtype=np.float64)
            seen = True
            if confirm is not None:
                motion = JOINT_POSITIVE_MOTION.get(name, "moves in its positive direction")
                if step < 0:
                    motion = f"the opposite of: {motion}"
                seen = bool(confirm(f"{name}: did it move like this? {motion}"))
            io.write_goals(base)
            sleep(settle_s)
            moved = after - before
            own = float(moved[index])
            others = np.delete(np.abs(moved), index)
            cross = int(np.argmax(np.abs(moved)))
            ok = (
                own * math.copysign(1.0, step) >= min_move_fraction * abs(step)
                and float(np.max(others)) <= max_cross_move_rad
                and seen
            )
            passed &= ok
            rows.append(
                {
                    "joint": name,
                    "goal_step_rad": step,
                    "moved_rad": own,
                    "largest_mover": manifest.joint_names[cross],
                    "confirmed_by_eye": seen if confirm is not None else None,
                    "ok": ok,
                }
            )
    finally:
        io.write_goals(base)
    failed = [
        f"{r['joint']} (moved {r['moved_rad']:+.3f}, largest mover {r['largest_mover']})"
        for r in rows
        if not r["ok"]
    ]
    return CheckResult(
        "joint_directions",
        passed,
        "every joint follows its goal alone"
        + (" and moved as described" if confirm is not None else " (motion not confirmed by eye)")
        if passed
        else f"failed: {failed}",
        {"joints": rows},
    )


def check_loop_timing(
    io: RobotIO,
    controller: Controller,
    *,
    cycles: int = 250,
    clock: Callable[[], float] = time.perf_counter,
    max_mean_ms: float = 15.0,
    max_p99_ms: float = 20.0,
) -> CheckResult:
    """Time read + infer without writing, so the robot does not move."""
    controller.reset()
    controller.set_command(Command.stand())
    durations = []
    for _ in range(cycles):
        start = clock()
        controller.step(io.read())
        durations.append((clock() - start) * 1000.0)
    controller.reset()
    mean, p99 = float(np.mean(durations)), float(np.percentile(durations, 99))
    return CheckResult(
        "loop_timing",
        mean <= max_mean_ms and p99 <= max_p99_ms,
        f"read+infer mean {mean:.2f} ms (<= {max_mean_ms}), p99 {p99:.2f} ms (<= {max_p99_ms}); "
        f"the write and the 20 ms period need the rest",
        {"mean_ms": mean, "p99_ms": p99},
    )


def check_policy_stand(
    io: RobotIO,
    controller: Controller,
    *,
    duration_s: float = 5.0,
    max_tilt_deg: float = 20.0,
    max_pinned_fraction: float = 0.2,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Sleep = time.sleep,
) -> CheckResult:
    """Stand under the policy; stops writing as soon as the trunk tilts too far."""
    manifest = controller.manifest
    period = manifest.control_period_s
    controller.reset()
    controller.set_command(Command.stand())
    io.set_torque(True)
    steps = max(1, round(duration_s / period))
    pinned = np.zeros(len(manifest.joint_names))
    worst_tilt = 0.0
    completed = 0
    aborted = False
    for _ in range(steps):
        start = clock()
        reading = io.read()
        gravity = np.asarray(reading.gravity_b, dtype=np.float64)
        tilt = math.degrees(
            math.acos(np.clip(-gravity[2] / max(np.linalg.norm(gravity), 1e-9), -1.0, 1.0))
        )
        worst_tilt = max(worst_tilt, tilt)
        if tilt > max_tilt_deg:
            aborted = True
            break
        goals = controller.step(reading, heading=io.heading())
        io.write_goals(goals)
        pinned += (goals <= manifest.joint_limits[:, 0] + 1e-4) | (
            goals >= manifest.joint_limits[:, 1] - 1e-4
        )
        completed += 1
        sleep(max(0.0, period - (clock() - start)))
    if aborted:
        io.write_goals(manifest.default_joint_pos)
    fraction = pinned / max(completed, 1)
    stuck = [
        name for name, f in zip(manifest.joint_names, fraction, strict=True) if f > max_pinned_fraction
    ]
    passed = not aborted and not stuck
    return CheckResult(
        "policy_stand",
        passed,
        f"stood {completed * period:.1f} s, worst tilt {worst_tilt:.1f} deg"
        + (" (aborted)" if aborted else "")
        + (f", pinned at a limit: {stuck}" if stuck else ""),
        {"worst_tilt_deg": worst_tilt, "pinned_fraction": fraction.tolist(), "aborted": aborted},
    )


__all__ = [
    "CheckResult",
    "IMU_AXIS_STEPS",
    "JOINT_POSITIVE_MOTION",
    "RobotIO",
    "check_default_pose",
    "check_imu_axes",
    "check_imu_static",
    "check_joint_directions",
    "check_loop_timing",
    "check_policy_stand",
]
