# Ported from Vottivott/microduck-playground hardware/microdock-v2/training
# (mdp_butt_walk.py, microduck_butt_walk_env_cfg.py; Apache-2.0) and the
# stillness / joint-limit helpers it uses from tasks/mdp.py; adapted to the
# NumPy manager runtime.
"""Seated scooting on command ("butt walk") terms for MicroDuck.

Sign convention (upstream, kept here): ``*_l1`` and ``*_penalty`` functions
return <= 0 and take POSITIVE weights; ``joint_limit_proximity`` returns a
cost >= 0 and takes a NEGATIVE weight. Terminations return bool masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import _real
from microduck_rl_unilab.tasks.microduck.manager_terms import _asset_selection, _command
from microduck_rl_unilab.tasks.microduck.recovery_terms import _check_params, _root_z

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


_HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def root_height_above(
    env: ManagerBasedRlEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Termination: the trunk left the seated band (stood up)."""
    return _root_z(env, env.scene[asset_cfg.name]) > max_height


def height_above_l1_penalty(
    env: ManagerBasedRlEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """One-sided L1 on trunk z above ``max_height``. Returns <= 0: positive weight."""
    z = _root_z(env, env.scene[asset_cfg.name])
    return np.asarray(-np.clip(z - max_height, 0.0, None), dtype=get_global_dtype())


def command_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    axis: str = "lin_x",
    min_cmd: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Achieved / commanded rate on one axis, clipped to [-1, 1]; 0 below ``min_cmd``.

    Gaussian tracking is flat near zero motion at docking speeds, so a seated
    duck that never moves earns most of it; this pays for any motion the right
    way and nothing beyond the commanded rate.
    """
    data = env.scene[asset_cfg.name].data
    command = _command(env, "command_progress", command_name)
    if axis == "lin_x":
        c, v = command[:, 0], data.root_link_lin_vel_b[:, 0]
    elif axis == "ang_z":
        c, v = command[:, 2], data.root_link_ang_vel_b[:, 2]
    else:
        raise ValueError("command_progress axis must be 'lin_x' or 'ang_z'")
    active = np.abs(c) > min_cmd
    ratio = np.clip(v / np.where(active, c, 1.0), -1.0, 1.0)
    return np.asarray(np.nan_to_num(np.where(active, ratio, 0.0), nan=0.0), dtype=get_global_dtype())


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """exp(-|v_xy|^2 / std^2) while the twist command is (near) zero, else 0."""
    data = env.scene[asset_cfg.name].data
    command = _command(env, "stillness_at_zero_command", command_name)
    idle = (np.linalg.norm(command[:, :2], axis=1) + np.abs(command[:, 2])) < command_threshold
    speed = np.linalg.norm(np.nan_to_num(data.root_link_lin_vel_w[:, :2], nan=0.0), axis=1)
    return np.asarray(idle * np.exp(-np.square(speed / vel_std)), dtype=get_global_dtype())


class _HeadErrorTerm(ManagerTermBase):
    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg", "command_name", "max_error"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._entity = env.scene[asset_cfg.name]
        self._ids = np.asarray([int(self._entity.find_joints(f"^{n}$")[0][0]) for n in _HEAD_JOINTS], dtype=np.intp)
        self._command_name = cfg.params.get("command_name", "head_pose")

    def _error(self, env: ManagerBasedRlEnv) -> np.ndarray:
        data = self._entity.data
        command = _command(env, self.name, self._command_name)[:, : len(self._ids)]
        return np.nan_to_num(data.joint_pos[:, self._ids] - data.default_joint_pos[:, self._ids] - command, nan=0.0)


class head_pose_l1(_HeadErrorTerm):
    """Mean |neck/head error| vs the head command. Returns <= 0: positive weight.

    The Gaussian head tracking goes flat once the head is far off; without a
    linear price the scooting policy bowed its head ~85 deg to unload the seat.
    """

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(-np.mean(np.abs(self._error(env)), axis=1), dtype=get_global_dtype())


class head_off_command(_HeadErrorTerm):
    """Termination: any neck/head joint more than ``max_error`` rad off the command."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._max_error = _real(self.name, "max_error", cfg.params.get("max_error", 0.8), minimum=0.0, strict_minimum=True)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.max(np.abs(self._error(env)), axis=1) > self._max_error


class joint_limit_proximity(ManagerTermBase):
    """Sum over joints of the L1 entry into a ``margin`` band next to their limits.

    Cost >= 0: negative weight. Upstream reads the hard limits; the UniLab entity
    exposes the soft limits, which the band is measured from here.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg", "margin"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity, self._ids = _asset_selection(cfg, env, term=self.name)
        self._margin = _real(self.name, "margin", cfg.params.get("margin", 0.15), minimum=0.0)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        q = self._entity.data.joint_pos[:, self._ids]
        limits = np.asarray(self._entity.data.soft_joint_pos_limits)
        limits = limits[self._ids] if limits.ndim == 2 else limits[:, self._ids]
        low, high = limits[..., 0] + self._margin, limits[..., 1] - self._margin
        cost = np.clip(low - q, 0.0, None) + np.clip(q - high, 0.0, None)
        return np.asarray(np.sum(cost, axis=1), dtype=get_global_dtype())


__all__ = [
    "command_progress",
    "head_off_command",
    "head_pose_l1",
    "height_above_l1_penalty",
    "joint_limit_proximity",
    "root_height_above",
    "stillness_at_zero_command",
]
