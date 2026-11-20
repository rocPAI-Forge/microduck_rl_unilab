# Ported from pollen-robotics/microduck_rl (develop, tasks/mdp.py posture_*
# terms and microduck_sitstand_env_cfg.py, Apache-2.0); adapted to the NumPy
# manager runtime.
"""Commanded sit <-> stand terms for MicroDuck.

Every posture term reads the ``SitStandCommand``'s slewed ``alpha`` (0 = STAND,
1 = SIT) and tracks a target that moves with it: the leg pose blends from
``default_joint_pos`` to the sit keyframe, and the trunk height from
``stand_z`` to ``sit_z``. Tracking a slow setpoint is what makes a gentle
transition the reward optimum; with a binary target, arriving early pays the
full goal reward for every step saved, and crash-sitting wins.

Sign convention (upstream, kept here): ``*_l1`` and ``*_penalty`` functions
return <= 0 and take POSITIVE weights.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import _real
from microduck_rl_unilab.tasks.microduck.manager_terms import _asset_selection, _command
from microduck_rl_unilab.tasks.microduck.recovery_terms import (
    _check_params,
    _cos_tilt,
    _root_z,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


_HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Slewed target blend in [0, 1]; falls back to the raw sit flag."""
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return np.asarray(alpha, dtype=get_global_dtype())
    return np.asarray(_command(env, "posture", command_name)[:, 0], dtype=get_global_dtype())


def _sit_flag(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    return np.asarray(_command(env, "posture", command_name)[:, 0], dtype=get_global_dtype())


def _posture_height(
    env: ManagerBasedRlEnv, entity: Any, command_name: str, sit_z: float, stand_z: float
) -> tuple[np.ndarray, np.ndarray]:
    blend = _posture_blend(env, command_name)
    return stand_z + blend * (sit_z - stand_z), _root_z(env, entity)


def _tilt_gate(entity: Any, full_deg: float, zero_deg: float) -> np.ndarray:
    cos_full = math.cos(math.radians(full_deg))
    cos_zero = math.cos(math.radians(zero_deg))
    u = np.clip((_cos_tilt(entity) - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    return _smoothstep(u)


class _PostureTargetTerm(ManagerTermBase):
    """Leg-pose error against the blended STAND/SIT target.

    ``sit_overrides`` maps joint NAMES to their sit-keyframe angles; joints not
    listed keep their default angle in both postures.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"asset_cfg", "command_name", "sit_overrides"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        _check_params(cfg, self._allowed_params, self.name)
        self._entity, self._joint_ids = _asset_selection(cfg, env, term=self.name)
        self._command_name = cfg.params.get("command_name", "twist")
        overrides = cfg.params.get("sit_overrides") or {}
        if not isinstance(overrides, dict):
            raise TypeError(f"{self.name} sit_overrides must be a joint-name mapping")
        selected = {int(j): k for k, j in enumerate(self._joint_ids)}
        delta = np.zeros(len(self._joint_ids), dtype=get_global_dtype())
        default = np.asarray(self._entity.data.default_joint_pos[0, self._joint_ids])
        for joint_name, angle in overrides.items():
            ids, _ = self._entity.find_joints(f"^{joint_name}$")
            column = selected.get(int(ids[0]))
            if column is None:
                raise ValueError(f"{self.name} sit_overrides joint '{joint_name}' is not selected")
            delta[column] = _real(self.name, f"sit_overrides[{joint_name}]", angle) - default[column]
        self._sit_delta = delta

    def _target(self, env: ManagerBasedRlEnv) -> np.ndarray:
        blend = _posture_blend(env, self._command_name)
        default = self._entity.data.default_joint_pos[:, self._joint_ids]
        return default + blend[:, None] * self._sit_delta[None, :]

    def _pose_error(self, env: ManagerBasedRlEnv) -> np.ndarray:
        actual = self._entity.data.joint_pos[:, self._joint_ids]
        return np.asarray(actual - self._target(env), dtype=get_global_dtype())


class posture_pose_match(_PostureTargetTerm):
    """Gaussian leg-pose match against the commanded posture's moving target."""

    _allowed_params: ClassVar[frozenset[str]] = _PostureTargetTerm._allowed_params | {"std"}

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._std = _real(self.name, "std", cfg.params.get("std", 0.5), minimum=0.0, strict_minimum=True)

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        per_joint = np.exp(-np.square(self._pose_error(env) / self._std))
        return np.asarray(np.mean(per_joint, axis=1), dtype=get_global_dtype())


class posture_pose_l1(_PostureTargetTerm):
    """L1 companion to ``posture_pose_match``. Returns <= 0: positive weight."""

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        return np.asarray(-np.mean(np.abs(self._pose_error(env)), axis=1), dtype=get_global_dtype())


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    sit_z: float,
    stand_z: float,
    command_name: str = "twist",
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Gaussian on trunk z against the commanded posture's moving target height."""
    scale = _real("posture_height_gaussian", "std", std, minimum=0.0, strict_minimum=True)
    target, z = _posture_height(env, env.scene[asset_cfg.name], command_name, sit_z, stand_z)
    return np.asarray(np.exp(-np.square((z - target) / scale)), dtype=get_global_dtype())


def posture_height_l1(
    env: ManagerBasedRlEnv,
    sit_z: float,
    stand_z: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """L1 height error; makes resting in the wrong posture net-negative. Positive weight."""
    target, z = _posture_height(env, env.scene[asset_cfg.name], command_name, sit_z, stand_z)
    return np.asarray(-np.abs(z - target), dtype=get_global_dtype())


class posture_composite(_PostureTargetTerm):
    """Multiplicative goal score: height x upright x leg pose (x head vs its command).

    A deficiency in any factor collapses the term, so partial-sum rests (plank,
    flop, lean, head dangling to the floor) earn nothing.
    """

    _allowed_params: ClassVar[frozenset[str]] = _PostureTargetTerm._allowed_params | {
        "sit_z",
        "stand_z",
        "height_std",
        "upright_std",
        "pose_std",
        "head_std",
        "head_command_name",
    }

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._sit_z = _real(self.name, "sit_z", p.get("sit_z"))
        self._stand_z = _real(self.name, "stand_z", p.get("stand_z"))
        self._height_std = _real(self.name, "height_std", p.get("height_std", 0.03), minimum=0.0, strict_minimum=True)
        self._upright_std = _real(self.name, "upright_std", p.get("upright_std", 0.4), minimum=0.0, strict_minimum=True)
        self._pose_std = _real(self.name, "pose_std", p.get("pose_std", 0.4), minimum=0.0, strict_minimum=True)
        head_std = p.get("head_std")
        self._head_std = None if head_std is None else _real(self.name, "head_std", head_std, minimum=0.0, strict_minimum=True)
        self._head_command = p.get("head_command_name", "head_pose")
        self._head_ids = np.asarray(
            [int(self._entity.find_joints(f"^{name}$")[0][0]) for name in _HEAD_JOINTS], dtype=np.intp
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        entity = self._entity
        target_z, z = _posture_height(env, entity, self._command_name, self._sit_z, self._stand_z)
        height = np.exp(-np.square((z - target_z) / self._height_std))
        quat = entity.data.root_link_quat_w
        tilt_sq = 2.0 * (np.square(quat[:, 1]) + np.square(quat[:, 2]))
        upright = np.exp(-tilt_sq / self._upright_std**2)
        pose = np.exp(-np.mean(np.square(self._pose_error(env)), axis=1) / self._pose_std**2)
        score = height * upright * pose
        if self._head_std is not None:
            head_cmd = _command(env, self.name, self._head_command)
            actual = entity.data.joint_pos[:, self._head_ids] - entity.data.default_joint_pos[:, self._head_ids]
            head_err = np.mean(np.square(actual - head_cmd[:, : len(self._head_ids)]), axis=1)
            score = score * np.exp(-head_err / self._head_std**2)
        return np.asarray(score, dtype=get_global_dtype())


def posture_stillness(
    env: ManagerBasedRlEnv,
    sit_z: float,
    stand_z: float,
    command_name: str = "twist",
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Trunk stillness at the commanded height, upright, once the target ramp is done."""
    entity = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, entity, command_name, sit_z, stand_z)
    speed = np.linalg.norm(np.nan_to_num(entity.data.root_link_lin_vel_w, nan=0.0), axis=1)
    ramp_done = (np.abs(_sit_flag(env, command_name) - _posture_blend(env, command_name)) < 0.02)
    t = np.clip((band_zero - np.abs(z - target_z)) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    score = (
        np.exp(-np.square(speed / vel_std))
        * _smoothstep(t)
        * _tilt_gate(entity, tilt_full_deg, tilt_zero_deg)
        * ramp_done
    )
    return np.asarray(score, dtype=get_global_dtype())


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    max_height: float,
    command_name: str = "twist",
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Pays upward trunk speed (capped) under a STAND command below ``max_height``.

    Destination-only rewards have no gradient at zero motion; paying the rise
    itself makes any attempt positive. Zero under a SIT command.
    """
    entity = env.scene[asset_cfg.name]
    vz = np.nan_to_num(entity.data.root_link_lin_vel_w[:, 2], nan=0.0)
    rise = np.clip(vz, 0.0, np.inf if max_vz is None else max_vz)
    low = _root_z(env, entity) < max_height
    return np.asarray(rise * low * (1.0 - _sit_flag(env, command_name)), dtype=get_global_dtype())


def posture_sit_bootstrap(
    env: ManagerBasedRlEnv,
    min_height: float,
    command_name: str = "twist",
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Pays downward trunk speed (capped) under a SIT command above ``min_height``.

    The mirror of ``posture_rise_bootstrap``. A policy warm-started from a
    standing one gets no gradient toward the seat from destination rewards
    alone; paying the descent itself makes the first attempt positive. The cap
    keeps a drop from out-earning a gentle sit. Zero under a STAND command.
    """
    entity = env.scene[asset_cfg.name]
    vz = np.nan_to_num(entity.data.root_link_lin_vel_w[:, 2], nan=0.0)
    descent = np.clip(-vz, 0.0, np.inf if max_vz is None else max_vz)
    high = _root_z(env, entity) > min_height
    return np.asarray(descent * high * _sit_flag(env, command_name), dtype=get_global_dtype())


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Charges downward trunk speed beyond ``max_down_vel``. Returns <= 0: positive weight."""
    vz = np.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2], nan=0.0)
    return np.asarray(-np.clip(-vz - max_down_vel, 0.0, None), dtype=get_global_dtype())


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """Charges upward trunk speed beyond ``max_up_vel``. Returns <= 0: positive weight."""
    vz = np.nan_to_num(env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2], nan=0.0)
    return np.asarray(-np.clip(vz - max_up_vel, 0.0, None), dtype=get_global_dtype())


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> np.ndarray:
    """``cos(tilt)`` weighted by a smoothstep on trunk z: blocks tipping back while still tall."""
    entity = env.scene[asset_cfg.name]
    t = np.clip((_root_z(env, entity) - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    return np.asarray(_cos_tilt(entity) * _smoothstep(t), dtype=get_global_dtype())


__all__ = [
    "posture_composite",
    "posture_height_gaussian",
    "posture_height_l1",
    "posture_pose_l1",
    "posture_pose_match",
    "posture_rise_bootstrap",
    "posture_sit_bootstrap",
    "posture_stillness",
    "trunk_downward_velocity_penalty",
    "trunk_upward_velocity_penalty",
    "upright_while_tall",
]
