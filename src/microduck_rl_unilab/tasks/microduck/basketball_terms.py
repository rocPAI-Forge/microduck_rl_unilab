"""Basketball balance terms (ball entity, hold assist, rewards, reset).

NumPy port of Vottivott/microduck-playground ``tasks/mdp.py`` basketball
section (Apache-2.0): the duck stands on the apex of a free ball and follows
the twist command by rolling it. The difficulty axis is ``BallHoldAction``, a
per-env spring/damper wrench pinning the ball to its spawn point plus an
upright/centering assist on the duck, both scaled by a hold level in [0, 1]
that ``basketball_hold_curriculum`` lowers per env as it balances.

State shared by the terms lives on ``env._basketball_state`` (hold, level,
anchor), created on first use.
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
from unilab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

# Foot site (``left_foot`` / ``right_foot``) in its ankle body's frame; both
# sides share it. UniLab entities expose body poses but not site poses.
FOOT_SITE_OFFSET = (0.0, -0.0238146, -0.0140852)
_UP = np.array([0.0, 0.0, 1.0])


def _state(env: ManagerBasedRlEnv) -> SimpleNamespace:
    state = getattr(env, "_basketball_state", None)
    if state is None:
        state = SimpleNamespace(
            hold=np.ones(env.num_envs),
            level=np.zeros(env.num_envs, dtype=np.int64),
            anchor=None,
        )
        env._basketball_state = state  # type: ignore[attr-defined]
    return state


def _pos(entity: Entity) -> np.ndarray:
    return np.nan_to_num(np.asarray(entity.data.root_link_pos_w, dtype=np.float64), nan=0.0)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (N, 3) by wxyz quaternions ``q`` (N, 4)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_rotate(q * np.array([1.0, -1.0, -1.0, -1.0]), v)


def _quat_from_yaw_pitch_roll(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    return np.stack(
        (cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
         cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr),
        axis=-1,
    )


def _out(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=get_global_dtype())


# --- ball hold (difficulty axis) ------------------------------------------------


@dataclass(kw_only=True)
class BallHoldActionCfg(ActionTermCfg):
    """Zero-width action term: hold=1 pins the ball like a weld and props the
    duck upright over it; hold=0 is a free ball and no assist.

    The upstream spawn probe found the passive default pose topples off even
    a pinned ball within 1 s (a sole on a sphere has no sagittal support), so
    the duck assist fades with the same hold level as the ball spring.
    """

    entity_name: str = "ball"
    duck_name: str = "robot"
    levels: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
    kp: float = 400.0  # N/m at hold=1
    kd: float = 20.0  # N s/m
    kd_rot: float = 0.2  # N m s
    duck_kp_rot: float = 3.0  # N m/rad toward world up (gravity torque at 30 deg ~0.5 N m)
    duck_kd_rot: float = 0.08  # N m s
    duck_kp: float = 40.0  # N/m, root xy toward the ball centre
    duck_kd: float = 4.0  # N s/m

    def build(self, env: ManagerBasedRlEnv) -> BallHoldAction:
        return BallHoldAction(self, env)


class BallHoldAction(ActionTerm):
    cfg: BallHoldActionCfg

    def __init__(self, cfg: BallHoldActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._duck = cast("Entity", env.scene[cfg.duck_name])
        _, self._ball_body = self._entity.bind_body_wrench([0], torque=True, term_name="ball_hold")
        _, self._duck_body = self._duck.bind_body_wrench([0], torque=True, term_name="ball_hold")
        _state(env).hold[:] = float(cfg.levels[0])
        self._raw = np.zeros((self.num_envs, 0), dtype=get_global_dtype())

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw

    def process_actions(self, actions: np.ndarray) -> None:
        del actions
        state = _state(self._env)
        if state.anchor is None or not np.any(state.hold > 0.0):
            return
        cfg = self.cfg
        hold = state.hold[:, None]
        ball = self._entity.data
        pos = _pos(self._entity)
        v = np.nan_to_num(np.asarray(ball.root_link_lin_vel_w, dtype=np.float64), nan=0.0)
        w = np.nan_to_num(np.asarray(ball.root_link_ang_vel_w, dtype=np.float64), nan=0.0)
        force = hold * (-cfg.kp * (pos - state.anchor) - cfg.kd * v)
        force[:, 2] = hold[:, 0] * (-cfg.kd * v[:, 2])  # the floor holds it vertically
        torque = hold * (-cfg.kd_rot * w)

        duck = self._duck.data
        up = _quat_rotate(np.asarray(duck.root_link_quat_w, dtype=np.float64), np.broadcast_to(_UP, (self.num_envs, 3)))
        dw = np.nan_to_num(np.asarray(duck.root_link_ang_vel_w, dtype=np.float64), nan=0.0)
        dv = np.nan_to_num(np.asarray(duck.root_link_lin_vel_w, dtype=np.float64), nan=0.0)
        duck_torque = hold * (cfg.duck_kp_rot * np.cross(up, _UP) - cfg.duck_kd_rot * dw)
        duck_force = hold * (-cfg.duck_kp * (_pos(self._duck) - pos) - cfg.duck_kd * dv)
        duck_force[:, 2] = 0.0

        # Entity.apply_body_wrench_to_sim clears every pending wrench before
        # writing its own, so the two bodies go through the accumulating
        # backend call together. The wrench holds for the whole control step.
        backend = self._env._backend  # type: ignore[attr-defined]
        backend.apply_body_force(
            np.concatenate((self._ball_body, self._duck_body)),
            np.stack((force, duck_force), axis=1),
            torque=np.stack((torque, duck_torque), axis=1),
        )

    def apply_actions(self) -> None:
        return

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        del env_ids


def basketball_hold_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    levels: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0),
    promote_len_s: float = 6.0,
    demote_len_s: float = 1.5,
) -> float:
    """Per-env hold level: promote after an episode of ``promote_len_s``,
    demote after a fall within ``demote_len_s``. Returns the mean level."""
    state = _state(env)
    ids = resolve_env_ids(env, env_ids)
    if ids.size:
        lengths = np.asarray(env.episode_length_buf)[ids].astype(np.float64) * float(env.step_dt)
        level = state.level[ids] + (lengths >= promote_len_s) - (lengths < demote_len_s)
        state.level[ids] = np.clip(level, 0, len(levels) - 1)
        state.hold[ids] = np.asarray(levels, dtype=np.float64)[state.level[ids]]
    return float(state.level.mean())


# --- reset ------------------------------------------------------------------------


def reset_basketball(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    ball_radius: float = 0.12,
    root_height: float = 0.125,
    xy_noise: float = 0.01,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.05,
    robot_cfg: SceneEntityCfg | None = None,
    ball_cfg: SceneEntityCfg | None = None,
) -> None:
    """Ball resting at the env origin, duck standing on its apex (root at
    2R + root_height), random yaw, small tilt and joint noise."""
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    robot = cast("Entity", env.scene[(robot_cfg or SceneEntityCfg("robot")).name])
    ball = cast("Entity", env.scene[(ball_cfg or SceneEntityCfg("ball")).name])
    n, dtype = ids.size, get_global_dtype()
    origins = np.asarray(env.scene.env_origins, dtype=np.float64)[ids]
    rng = env.rng

    ball_pos = origins + np.array([0.0, 0.0, ball_radius + 0.001])
    ball_pose = np.concatenate((ball_pos, np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))), axis=1)
    ball.write_root_link_pose_to_sim(ball_pose.astype(dtype), env_ids=ids)
    ball.write_root_link_velocity_to_sim(np.zeros((n, 6), dtype=dtype), env_ids=ids)
    state = _state(env)
    if state.anchor is None:
        state.anchor = np.zeros((env.num_envs, 3))
    state.anchor[ids] = ball_pos

    xy = origins[:, :2] + rng.uniform(-xy_noise, xy_noise, size=(n, 2))
    z = origins[:, 2:3] + 2.0 * ball_radius + root_height + 0.003
    tilt = math.radians(tilt_noise_deg)
    quat = _quat_from_yaw_pitch_roll(
        rng.uniform(yaw_range[0], yaw_range[1], size=n), rng.uniform(-tilt, tilt, size=n), rng.uniform(-tilt, tilt, size=n)
    )
    robot.write_root_link_pose_to_sim(np.concatenate((xy, z, quat), axis=1).astype(dtype), env_ids=ids)
    robot.write_root_link_velocity_to_sim(np.zeros((n, 6), dtype=dtype), env_ids=ids)
    joints = np.asarray(robot.data.default_joint_pos, dtype=np.float64)[ids]
    joints = joints + rng.uniform(-joint_noise, joint_noise, size=joints.shape)
    robot.write_joint_state_to_sim(joints.astype(dtype), np.zeros_like(joints, dtype=dtype), env_ids=ids)


# --- observations -------------------------------------------------------------------


def basketball_state(
    env: ManagerBasedRlEnv, pos_scale: float = 3.0, vel_scale: float = 1.0, command_name: str | None = None
) -> np.ndarray:
    """Privileged 6-D ball block: ball centre relative to the root (base frame),
    ball xy velocity (base frame), and its rolling rate about base y.

    ``command_name`` is accepted and ignored: the term replaces a command slot
    whose inherited params carry it.
    """
    del command_name
    robot = cast("Entity", env.scene["robot"])
    ball = cast("Entity", env.scene["ball"])
    q = np.asarray(robot.data.root_link_quat_w, dtype=np.float64)
    rel = _quat_rotate_inverse(q, _pos(ball) - _pos(robot))
    v = _quat_rotate_inverse(q, np.nan_to_num(np.asarray(ball.data.root_link_lin_vel_w, dtype=np.float64), nan=0.0))
    w = _quat_rotate_inverse(q, np.nan_to_num(np.asarray(ball.data.root_link_ang_vel_w, dtype=np.float64), nan=0.0))
    return _out(np.concatenate((rel * pos_scale, v[:, :2] * vel_scale, w[:, 1:2] * 0.2), axis=1))


def zero_pad(env: ManagerBasedRlEnv, width: int = 6, command_name: str | None = None) -> np.ndarray:
    """Zero block keeping the 61-D actor contract (unused command slots);
    ``command_name`` is ignored, as in ``basketball_state``."""
    del command_name
    return np.zeros((env.num_envs, width), dtype=get_global_dtype())


# --- rewards ------------------------------------------------------------------------


def _foot_positions(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> np.ndarray:
    robot = cast("Entity", env.scene[asset_cfg.name])
    ids = asset_cfg.body_ids
    pos = np.asarray(robot.data.body_link_pos_w, dtype=np.float64)[:, ids, :]
    quat = np.asarray(robot.data.body_link_quat_w, dtype=np.float64)[:, ids, :]
    n, k = pos.shape[:2]
    offset = np.broadcast_to(np.asarray(FOOT_SITE_OFFSET), (n * k, 3))
    return pos + _quat_rotate(quat.reshape(-1, 4), offset).reshape(n, k, 3)


def basketball_upright(env: ManagerBasedRlEnv, std: float = 0.3) -> np.ndarray:
    g = np.nan_to_num(np.asarray(cast("Entity", env.scene["robot"]).data.projected_gravity_b, dtype=np.float64), nan=0.0)
    return _out(np.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) / (std * std)))


def basketball_centered(env: ManagerBasedRlEnv, std: float = 0.04) -> np.ndarray:
    """Gaussian on the horizontal root-to-ball-centre offset."""
    d = _pos(cast("Entity", env.scene["ball"]))[:, :2] - _pos(cast("Entity", env.scene["robot"]))[:, :2]
    return _out(np.exp(-(d * d).sum(axis=1) / (std * std)))


def basketball_feet_on_ball(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    ball_radius: float = 0.12,
    sole_gap: float = 0.012,
    std: float = 0.02,
) -> np.ndarray:
    """Per-foot Gaussian on (|foot - ball centre| - (R + gap)), averaged: both
    soles resting on the ball pay 1, a lifted foot pays ~0."""
    feet = _foot_positions(env, asset_cfg)
    ball = _pos(cast("Entity", env.scene["ball"]))
    d = np.linalg.norm(feet - ball[:, None, :], axis=-1) - (ball_radius + sole_gap)
    return _out(np.exp(-(d * d) / (std * std)).mean(axis=1))


def basketball_height(
    env: ManagerBasedRlEnv, ball_radius: float = 0.12, root_height: float = 0.125, std: float = 0.03
) -> np.ndarray:
    """Gaussian on the root height above the ball centre (standing on the apex)."""
    dz = _pos(cast("Entity", env.scene["robot"]))[:, 2] - _pos(cast("Entity", env.scene["ball"]))[:, 2]
    dz = dz - (ball_radius + root_height)
    return _out(np.exp(-(dz * dz) / (std * std)))


def basketball_ball_speed_l2(env: ManagerBasedRlEnv) -> np.ndarray:
    """Squared horizontal ball speed (>= 0: negative weight)."""
    v = np.nan_to_num(np.asarray(cast("Entity", env.scene["ball"]).data.root_link_lin_vel_w, dtype=np.float64), nan=0.0)
    return _out((v[:, :2] ** 2).sum(axis=1))


# --- termination --------------------------------------------------------------------


def basketball_fell(
    env: ManagerBasedRlEnv,
    ball_radius: float = 0.12,
    fall_tilt_deg: float = 55.0,
    min_height: float = 0.05,
    max_offset: float = 0.16,
) -> np.ndarray:
    """Off the ball: root low over the ball centre, off to the side, or tilted."""
    robot = cast("Entity", env.scene["robot"])
    root, ball = _pos(robot), _pos(cast("Entity", env.scene["ball"]))
    g = np.nan_to_num(np.asarray(robot.data.projected_gravity_b, dtype=np.float64), nan=0.0)
    tilt = np.arccos(np.clip(-g[:, 2], -1.0, 1.0))
    off = np.linalg.norm(root[:, :2] - ball[:, :2], axis=1)
    return (root[:, 2] - ball[:, 2] < ball_radius + min_height) | (off > max_offset) | (tilt > math.radians(fall_tilt_deg))
