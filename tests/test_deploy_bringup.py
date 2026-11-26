"""Bring-up checks pass on a correct robot and catch each wiring fault."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from microduck_rl_unilab.deploy import Controller, DeployManifest, SensorReading
from microduck_rl_unilab.deploy.bringup import (
    IMU_AXIS_STEPS,
    JOINT_POSITIVE_MOTION,
    check_default_pose,
    check_imu_axes,
    check_imu_static,
    check_joint_directions,
    check_loop_timing,
    check_policy_stand,
)

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "examples" / "sprint_stand" / "deploy"
MANIFEST = DeployManifest.load(DEPLOY_DIR)


def _no_sleep(_: float) -> None:
    pass


class FakeRobot:
    """Servos follow goals at once; faults are applied between robot and runtime."""

    def __init__(
        self,
        *,
        swap: tuple[int, int] | None = None,
        flip: int | None = None,
        offset: tuple[int, float] | None = None,
        imu_flip_axis: int | None = None,
    ):
        self.q = MANIFEST.default_joint_pos.astype(np.float64).copy()
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.gyro = np.zeros(3)
        self.torque = False
        self.swap, self.flip, self.offset, self.imu_flip_axis = swap, flip, offset, imu_flip_axis

    def _to_robot(self, goals: np.ndarray) -> np.ndarray:
        q = np.asarray(goals, dtype=np.float64).copy()
        if self.flip is not None:
            # Motor turns the other way; the encoder still agrees with the command.
            q[self.flip] = 2 * MANIFEST.default_joint_pos[self.flip] - q[self.flip]
        if self.swap is not None:
            a, b = self.swap
            q[[a, b]] = q[[b, a]]
        return q

    def write_goals(self, goals: np.ndarray) -> None:
        self.q = self._to_robot(goals)

    def read(self) -> SensorReading:
        q = self.q.copy()
        if self.flip is not None:
            q[self.flip] = 2 * MANIFEST.default_joint_pos[self.flip] - q[self.flip]
        if self.swap is not None:
            a, b = self.swap
            q[[a, b]] = q[[b, a]]
        if self.offset is not None:
            q[self.offset[0]] += self.offset[1]
        gravity, gyro = self.gravity.copy(), self.gyro.copy()
        if self.imu_flip_axis is not None:
            gravity[self.imu_flip_axis] *= -1
            gyro[self.imu_flip_axis] *= -1
        return SensorReading(gyro, gravity, q.astype(np.float32), np.zeros(14))

    def heading(self) -> float:
        return 0.0

    def set_torque(self, enabled: bool) -> None:
        self.torque = enabled

    def apply_step(self, step: str) -> None:
        """What a human does for each IMU prompt, in the true body frame."""
        s = math.sin(math.radians(30))
        c = math.cos(math.radians(30))
        self.gravity, self.gyro = np.array([0.0, 0.0, -1.0]), np.zeros(3)
        if "nose-down" in step:
            self.gravity = np.array([s, 0.0, -c])
        elif "right side is down" in step:
            self.gravity = np.array([0.0, -s, -c])
        elif "Turn the robot left" in step:
            self.gyro = np.array([0.0, 0.0, 0.8])
        elif "Roll the robot right side down" in step:
            self.gyro = np.array([0.8, 0.0, 0.0])


def test_every_joint_has_a_described_positive_motion() -> None:
    assert set(JOINT_POSITIVE_MOTION) == set(MANIFEST.joint_names)
    assert len(IMU_AXIS_STEPS) == 4


def test_correct_robot_passes_every_check() -> None:
    robot = FakeRobot()
    assert check_imu_static(robot, sleep=_no_sleep).passed
    assert check_imu_axes(robot, robot.apply_step, sleep=_no_sleep).passed
    robot.apply_step("")
    assert check_default_pose(robot, MANIFEST, sleep=_no_sleep).passed
    assert check_joint_directions(
        robot, MANIFEST, confirm=lambda _: True, sleep=_no_sleep
    ).passed
    controller = Controller(DEPLOY_DIR)
    timing = check_loop_timing(robot, controller, cycles=20)
    assert timing.passed, timing.detail


def test_flipped_imu_axis_fails_the_axis_check() -> None:
    robot = FakeRobot(imu_flip_axis=1)
    result = check_imu_axes(robot, robot.apply_step, sleep=_no_sleep)
    assert not result.passed
    assert [s["ok"] for s in result.data["steps"]] == [True, False, True, True]


def test_tilted_robot_fails_the_static_imu_check() -> None:
    robot = FakeRobot()
    robot.apply_step("nose-down")
    assert not check_imu_static(robot, sleep=_no_sleep).passed


def test_zero_offset_fails_the_default_pose_check() -> None:
    robot = FakeRobot(offset=(3, 0.3))
    result = check_default_pose(robot, MANIFEST, sleep=_no_sleep)
    assert not result.passed and "left_knee" in result.detail


def test_swapped_ids_with_different_defaults_fail_the_default_pose() -> None:
    # left_hip_pitch -0.458 and right_hip_pitch +0.458 trade places.
    robot = FakeRobot(swap=(2, 11))
    robot.q = MANIFEST.default_joint_pos.astype(np.float64).copy()
    result = check_default_pose(robot, MANIFEST, sleep=_no_sleep)
    assert not result.passed
    assert "left_hip_pitch" in result.detail and "right_hip_pitch" in result.detail


def test_swapped_ids_with_equal_defaults_need_the_operator() -> None:
    # Both hip yaws default to 0 and a swapped ID reads back what it was sent,
    # so only the operator sees the wrong leg move.
    robot = FakeRobot(swap=(0, 9))
    robot.q = MANIFEST.default_joint_pos.astype(np.float64).copy()
    assert check_default_pose(robot, MANIFEST, sleep=_no_sleep).passed
    assert check_joint_directions(robot, MANIFEST, sleep=_no_sleep).passed
    wrong_leg = {"left_hip_yaw", "right_hip_yaw"}
    result = check_joint_directions(
        robot,
        MANIFEST,
        confirm=lambda message: message.split(":")[0] not in wrong_leg,
        sleep=_no_sleep,
    )
    failed = {row["joint"] for row in result.data["joints"] if not row["ok"]}
    assert failed == wrong_leg


def test_reversed_motor_needs_the_operator_to_catch_it() -> None:
    robot = FakeRobot(flip=3)
    by_encoder = check_joint_directions(robot, MANIFEST, sleep=_no_sleep)
    assert by_encoder.passed
    asked: list[str] = []

    def operator(message: str) -> bool:
        asked.append(message)
        return not message.startswith("left_knee")

    by_eye = check_joint_directions(robot, MANIFEST, confirm=operator, sleep=_no_sleep)
    assert not by_eye.passed
    assert "left shin swings forward" in asked[3]


def test_policy_stand_aborts_when_the_trunk_tips_over() -> None:
    robot = FakeRobot()
    robot.gravity = np.array([math.sin(math.radians(40)), 0.0, -math.cos(math.radians(40))])
    result = check_policy_stand(robot, Controller(DEPLOY_DIR), duration_s=0.2, sleep=_no_sleep)
    assert not result.passed and result.data["aborted"] is True
    np.testing.assert_allclose(robot.q, MANIFEST.default_joint_pos, atol=1e-6)
