"""Onboard runtime: manifest parsing, observation layout, clamping, heading hold."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from microduck_rl_unilab.deploy import (
    Command,
    Controller,
    DeployManifest,
    HeadingHold,
    ObservationBuilder,
    SensorReading,
)

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "examples" / "sprint_stand" / "deploy"


def _reading(manifest: DeployManifest, offset: float = 0.0) -> SensorReading:
    return SensorReading(
        gyro_b=np.array([0.1, -0.2, 0.3]),
        gravity_b=np.array([0.0, 0.0, -1.0]),
        joint_pos=manifest.default_joint_pos + offset,
        joint_vel=np.arange(14, dtype=np.float32) * 0.01,
    )


def test_manifest_reads_joint_order_defaults_and_limits() -> None:
    manifest = DeployManifest.load(DEPLOY_DIR)
    assert manifest.joint_names[0] == "left_hip_yaw"
    assert manifest.joint_names[5] == "neck_pitch"
    assert manifest.joint_names[-1] == "right_ankle"
    assert manifest.default_joint_pos[5] == pytest.approx(0.349066)
    assert manifest.joint_limits.shape == (14, 2)
    assert manifest.clamp is True
    assert manifest.control_period_s == pytest.approx(0.02)


def test_observation_follows_the_61d_segment_order() -> None:
    manifest = DeployManifest.load(DEPLOY_DIR)
    builder = ObservationBuilder(manifest)
    builder.last_action = np.full(14, 0.5, dtype=np.float32)
    obs = builder.build(_reading(manifest, offset=0.02), 0.6, -0.3)

    assert obs.shape == (61,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[0:3], [0.1, -0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(obs[6:20], 0.02, atol=1e-6)
    np.testing.assert_allclose(obs[20:34], np.arange(14) * 0.01, atol=1e-6)
    np.testing.assert_allclose(obs[34:48], 0.5)
    np.testing.assert_allclose(obs[48:51], [0.6, 0.0, -0.3], rtol=1e-6)
    np.testing.assert_array_equal(obs[51:61], 0.0)


def test_observation_rejects_wrong_sizes_and_nan() -> None:
    manifest = DeployManifest.load(DEPLOY_DIR)
    builder = ObservationBuilder(manifest)
    bad = SensorReading(np.zeros(2), np.zeros(3), np.zeros(14), np.zeros(14))
    with pytest.raises(ValueError, match="gyro_b"):
        builder.build(bad, 0.0, 0.0)
    nan = SensorReading(np.zeros(3), np.zeros(3), np.full(14, np.nan), np.zeros(14))
    with pytest.raises(ValueError, match="finite"):
        builder.build(nan, 0.0, 0.0)


def test_heading_hold_matches_the_training_loop() -> None:
    hold = HeadingHold()
    hold.reset(0.0, np.zeros(2))
    # Turned 0.1 rad right of the reference: steer back left.
    assert hold.yaw_command(-0.1, np.zeros(2)) == pytest.approx(0.25)
    # 0.5 m left of the ray at the reference heading: steer right.
    assert hold.yaw_command(0.0, np.array([0.0, 0.5])) == pytest.approx(-0.04)
    # Saturates at the trained limit and wraps across +-pi.
    assert hold.yaw_command(-1.0, np.zeros(2)) == pytest.approx(0.35)
    hold.reset(math.pi - 0.05)
    assert hold.yaw_command(-math.pi + 0.05) == pytest.approx(-0.25)


def test_turn_around_is_the_documented_half_metre_turn() -> None:
    left, right = Command.turn_around(1), Command.turn_around(-1)
    assert (left.forward_m_s, left.yaw_rate_rad_s) == (0.2, 0.4)
    assert (right.forward_m_s, right.yaw_rate_rad_s) == (0.2, -0.4)
    with pytest.raises(ValueError):
        Command.turn_around(0)


def test_controller_clamps_targets_and_feeds_back_the_raw_action() -> None:
    controller = Controller(DEPLOY_DIR)
    manifest = controller.manifest
    targets = controller.step(_reading(manifest))
    action = controller.observations.last_action

    assert targets.shape == (14,)
    assert np.all(targets >= manifest.joint_limits[:, 0] - 1e-6)
    assert np.all(targets <= manifest.joint_limits[:, 1] + 1e-6)
    unclamped = action * manifest.action_scale + manifest.default_joint_pos
    inside = (unclamped > manifest.joint_limits[:, 0]) & (unclamped < manifest.joint_limits[:, 1])
    np.testing.assert_allclose(targets[inside], unclamped[inside], atol=1e-6)

    next_obs = controller.observations.build(_reading(manifest), 0.0, 0.0)
    np.testing.assert_array_equal(next_obs[34:48], action)

    controller.reset()
    np.testing.assert_array_equal(controller.observations.last_action, 0.0)


def test_straight_command_needs_a_heading() -> None:
    controller = Controller(DEPLOY_DIR)
    with pytest.raises(ValueError, match="heading"):
        controller.set_command(Command.straight(0.6))
    controller.set_command(Command.straight(0.6), heading=0.2, position_xy=np.zeros(2))
    controller.step(_reading(controller.manifest), heading=0.2, position_xy=np.zeros(2))
    assert controller.last_obs is not None
    assert controller.last_obs[48] == pytest.approx(0.6)
    assert controller.last_obs[50] == pytest.approx(0.0, abs=1e-6)
