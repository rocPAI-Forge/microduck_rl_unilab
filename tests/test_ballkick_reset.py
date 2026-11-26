"""reset_ballkick places the ball in front of the kicking foot, in the robot's yaw frame."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_rl_unilab.tasks.microduck.kick_terms import reset_ballkick


class _Robot:
    def __init__(self, xy: np.ndarray, yaw: np.ndarray):
        n = len(yaw)
        self._pose = np.concatenate(
            (xy, np.full((n, 1), 0.12), np.stack((np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)), axis=1)),
            axis=1,
        )

    def read_reset_root_pose(self, env_ids: np.ndarray) -> np.ndarray:
        return self._pose[env_ids].copy()


class _Ball:
    def __init__(self, n: int):
        self.pose = np.zeros((n, 7))
        self.vel = np.ones((n, 6))

    def write_root_link_pose_to_sim(self, pose: np.ndarray, env_ids: np.ndarray) -> None:
        self.pose[env_ids] = pose

    def write_root_link_velocity_to_sim(self, vel: np.ndarray, env_ids: np.ndarray) -> None:
        self.vel[env_ids] = vel


class _Scene:
    def __init__(self, robot: _Robot, ball: _Ball, n: int):
        self._entities = {"robot": robot, "ball": ball}
        self.env_origins = np.zeros((n, 3))

    def __getitem__(self, name: str) -> object:
        return self._entities[name]


def _env(yaw: np.ndarray) -> tuple[SimpleNamespace, _Robot, _Ball]:
    n = len(yaw)
    xy = np.stack((np.arange(n, dtype=float), -np.arange(n, dtype=float)), axis=1)
    robot, ball = _Robot(xy, yaw), _Ball(n)
    return SimpleNamespace(num_envs=n, rng=np.random.default_rng(0), scene=_Scene(robot, ball, n)), robot, ball


@pytest.mark.parametrize("foot,side", [("right", -1.0), ("left", 1.0)])
def test_ball_sits_ahead_of_the_kicking_foot(foot: str, side: float) -> None:
    yaw = np.array([0.0, math.pi / 2, -2.5, 1.0])
    env, robot, ball = _env(yaw)
    reset_ballkick(env, np.arange(len(yaw)), foot=foot, noise_xy=0.0)
    rel = ball.pose[:, :2] - robot._pose[:, :2]
    c, s = np.cos(yaw), np.sin(yaw)
    forward = c * rel[:, 0] + s * rel[:, 1]
    left = -s * rel[:, 0] + c * rel[:, 1]
    np.testing.assert_allclose(forward, 0.09, atol=1e-6)
    np.testing.assert_allclose(left, side * 0.042, atol=1e-6)
    np.testing.assert_allclose(ball.pose[:, 2], 0.036, atol=1e-6)
    np.testing.assert_allclose(ball.vel, 0.0)
    np.testing.assert_allclose(env._kick_state.goal, yaw, atol=1e-6)


def test_unknown_foot_is_rejected() -> None:
    env, _, _ = _env(np.zeros(1))
    with pytest.raises(ValueError, match="foot"):
        reset_ballkick(env, np.arange(1), foot="both")
