"""Ball kicking terms: a pursuit twist command, reset, observation, rewards.

The duck walks with the sprint gait and kicks a free ball toward a goal
direction. Walking is not relearned: ``BallPursuitCommand`` rewrites the twist
command every step so the existing velocity tracking walks the duck behind the
ball and then through it along the goal direction. The policy sees the ball and
the goal in the (otherwise zero) head-pose command slot and is paid for ball
velocity along the goal, so it can learn to time and shape the kick.

State shared by the terms lives on ``env._kick_state`` (goal angle per env),
created on first use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import resolve_env_ids
from unilab.managers import SceneEntityCfg
from unilab.utils.rotation import np_wrap_to_pi

from microduck_rl_unilab.tasks.microduck.basketball_terms import _out, _pos, _quat_from_yaw_pitch_roll
from microduck_rl_unilab.tasks.microduck.manager_terms import MicroduckVelocityCommand, MicroduckVelocityCommandCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


def _state(env: ManagerBasedRlEnv) -> SimpleNamespace:
    state = getattr(env, "_kick_state", None)
    if state is None:
        state = SimpleNamespace(goal=np.zeros(env.num_envs))
        env._kick_state = state  # type: ignore[attr-defined]
    return state


def _goal_unit(env: ManagerBasedRlEnv) -> np.ndarray:
    goal = _state(env).goal
    return np.stack((np.cos(goal), np.sin(goal)), axis=1)


def _rotate_to_heading(v: np.ndarray, heading: np.ndarray) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    return np.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), axis=1)


# --- command ------------------------------------------------------------------------


@dataclass(kw_only=True)
class BallPursuitCommandCfg(MicroduckVelocityCommandCfg):
    """Twist command that walks the duck behind the ball, then through it."""

    ball_name: str = "ball"
    behind_distance: float = 0.12
    line_tolerance: float = 0.05
    push_through: float = 0.3
    lin_gain: float = 1.5
    yaw_gain: float = 2.0
    max_lin: float = 0.3
    max_yaw: float = 0.8

    def build(self, env: ManagerBasedRlEnv) -> BallPursuitCommand:
        return BallPursuitCommand(self, env)


class BallPursuitCommand(MicroduckVelocityCommand):
    def __init__(self, cfg: BallPursuitCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._ball = cast("Entity", env.scene[cfg.ball_name])

    def _update_command(self, env_ids: np.ndarray | None = None) -> None:
        super()._update_command(env_ids)
        cfg = cast(BallPursuitCommandCfg, self.cfg)
        ids = slice(None) if env_ids is None else env_ids
        robot_xy = np.asarray(self.robot.data.root_link_pos_w, dtype=np.float64)[ids, :2]
        heading = np.asarray(self.robot.data.heading_w, dtype=np.float64)[ids]
        ball_xy = _pos(self._ball)[ids, :2]
        goal = _goal_unit(self._env)[ids]
        rel = robot_xy - ball_xy
        along = np.sum(rel * goal, axis=1)
        across = rel[:, 0] * goal[:, 1] - rel[:, 1] * goal[:, 0]
        behind = (along < -0.5 * cfg.behind_distance) & (np.abs(across) < cfg.line_tolerance)
        target = np.where(
            behind[:, None], ball_xy + cfg.push_through * goal, ball_xy - cfg.behind_distance * goal
        )
        delta = target - robot_xy
        error = np_wrap_to_pi(np.arctan2(delta[:, 1], delta[:, 0]) - heading)
        dist = np.linalg.norm(delta, axis=1)
        dtype = self.vel_command_b.dtype
        self.vel_command_b[ids, 0] = (
            np.clip(cfg.lin_gain * dist, 0.0, cfg.max_lin) * np.clip(np.cos(error), 0.0, 1.0)
        ).astype(dtype)
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = np.clip(cfg.yaw_gain * error, -cfg.max_yaw, cfg.max_yaw).astype(dtype)


# --- reset --------------------------------------------------------------------------


def reset_kick(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    root_height: float = 0.12,
    xy_range: float = 0.5,
    tilt_deg: float = 2.5,
    ball_radius: float = 0.035,
    ball_distance: tuple[float, float] = (0.2, 0.45),
    ball_angle_deg: float = 30.0,
    goal_angle_deg: float = 30.0,
    robot_cfg: SceneEntityCfg | None = None,
    ball_cfg: SceneEntityCfg | None = None,
) -> None:
    """Duck standing at a random spot and yaw, ball in front of it within
    ``ball_angle_deg``, goal direction within ``goal_angle_deg`` of its heading.

    Replaces ``reset_base`` so the ball can be placed relative to the yaw it
    writes; reading heading back during reset would return the pre-reset yaw.
    """
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    robot = cast("Entity", env.scene[(robot_cfg or SceneEntityCfg("robot")).name])
    ball = cast("Entity", env.scene[(ball_cfg or SceneEntityCfg("ball")).name])
    n, dtype, rng = ids.size, get_global_dtype(), env.rng
    origins = np.asarray(env.scene.env_origins, dtype=np.float64)[ids]

    yaw = rng.uniform(-math.pi, math.pi, size=n)
    tilt = math.radians(tilt_deg)
    xy = origins[:, :2] + rng.uniform(-xy_range, xy_range, size=(n, 2))
    z = origins[:, 2:3] + root_height + rng.uniform(0.0, 0.01, size=(n, 1))
    quat = _quat_from_yaw_pitch_roll(yaw, rng.uniform(-tilt, tilt, size=n), rng.uniform(-tilt, tilt, size=n))
    robot.write_root_link_pose_to_sim(np.concatenate((xy, z, quat), axis=1).astype(dtype), env_ids=ids)
    robot.write_root_link_velocity_to_sim(np.zeros((n, 6), dtype=dtype), env_ids=ids)

    bearing = yaw + np.radians(rng.uniform(-ball_angle_deg, ball_angle_deg, size=n))
    dist = rng.uniform(ball_distance[0], ball_distance[1], size=n)
    ball_xy = xy + dist[:, None] * np.stack((np.cos(bearing), np.sin(bearing)), axis=1)
    ball_z = origins[:, 2:3] + ball_radius + 0.001
    ball_pose = np.concatenate((ball_xy, ball_z, np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))), axis=1)
    ball.write_root_link_pose_to_sim(ball_pose.astype(dtype), env_ids=ids)
    ball.write_root_link_velocity_to_sim(np.zeros((n, 6), dtype=dtype), env_ids=ids)

    _state(env).goal[ids] = yaw + np.radians(rng.uniform(-goal_angle_deg, goal_angle_deg, size=n))


# --- observation --------------------------------------------------------------------


def kick_ball_obs(
    env: ManagerBasedRlEnv,
    pos_scale: float = 0.05,
    dir_scale: float = 0.02,
    command_name: str | None = None,
) -> np.ndarray:
    """4-D block in the heading frame: ball xy relative to the root, goal (cos, sin).

    Fills the head-pose command slot, whose warm-start normalizer std is
    0.004-0.018 (frozen after billions of samples); the scales keep the
    normalized values near +-1. ``command_name`` is accepted and ignored.
    """
    del command_name
    robot = cast("Entity", env.scene["robot"])
    ball = cast("Entity", env.scene["ball"])
    heading = np.asarray(robot.data.heading_w, dtype=np.float64)
    rel = _rotate_to_heading(_pos(ball)[:, :2] - _pos(robot)[:, :2], heading)
    goal = _rotate_to_heading(_goal_unit(env), heading)
    return _out(np.concatenate((rel * pos_scale, goal * dir_scale), axis=1))


# --- rewards ------------------------------------------------------------------------


def kick_ball_velocity(env: ManagerBasedRlEnv, cap: float = 1.0) -> np.ndarray:
    """Ball velocity along the goal direction, clipped to [0, cap] m/s."""
    ball = cast("Entity", env.scene["ball"])
    v = np.nan_to_num(np.asarray(ball.data.root_link_lin_vel_w, dtype=np.float64)[:, :2], nan=0.0)
    return _out(np.clip(np.sum(v * _goal_unit(env), axis=1), 0.0, cap))


def kick_ball_off_goal(env: ManagerBasedRlEnv, cap: float = 1.0) -> np.ndarray:
    """|ball velocity across the goal direction|, clipped. Returns >= 0: negative weight."""
    ball = cast("Entity", env.scene["ball"])
    v = np.nan_to_num(np.asarray(ball.data.root_link_lin_vel_w, dtype=np.float64)[:, :2], nan=0.0)
    goal = _goal_unit(env)
    return _out(np.minimum(np.abs(v[:, 0] * goal[:, 1] - v[:, 1] * goal[:, 0]), cap))


__all__ = [
    "BallPursuitCommand",
    "BallPursuitCommandCfg",
    "kick_ball_obs",
    "kick_ball_off_goal",
    "kick_ball_velocity",
    "reset_kick",
]
