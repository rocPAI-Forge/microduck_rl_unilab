"""Sprint-specific reward terms for the MicroDuck Manager-Based task."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import UniformVelocityCommandCfg
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from unilab.utils.rotation import np_wrap_to_pi

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _positive_real(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def running_forward_progress(
    env: ManagerBasedRlEnv,
    speed_cap: float = 1.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Reward capped positive body-forward speed with a linear gradient."""
    cap = _positive_real(speed_cap, label="running_forward_progress speed_cap")
    asset = cast("Entity", env.scene[asset_cfg.name])
    velocity_x = np.nan_to_num(
        asset.data.root_link_lin_vel_b[:, 0],
        nan=0.0,
        posinf=cap,
        neginf=0.0,
    )
    return np.asarray(np.clip(velocity_x, 0.0, cap) / cap, dtype=get_global_dtype())


def running_planar_drift_cost(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    lateral_weight: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> np.ndarray:
    """Penalize body-frame lateral drift and yaw-rate command error."""
    weight = _positive_real(lateral_weight, label="running_planar_drift_cost lateral_weight")
    if not isinstance(command_name, str) or not command_name:
        raise ValueError("running_planar_drift_cost command_name must be non-empty")
    command = np.asarray(env.command_manager.get_command(command_name))
    if command.shape != (env.num_envs, 3):
        raise ValueError(
            f"running_planar_drift_cost command must have shape ({env.num_envs}, 3), "
            f"received {command.shape}"
        )
    asset = cast("Entity", env.scene[asset_cfg.name])
    lateral_error = np.nan_to_num(
        asset.data.root_link_lin_vel_b[:, 1] - command[:, 1],
        nan=0.0,
    )
    yaw_error = np.nan_to_num(
        asset.data.root_link_ang_vel_b[:, 2] - command[:, 2],
        nan=0.0,
    )
    return np.asarray(
        np.square(yaw_error) + weight * np.square(lateral_error),
        dtype=get_global_dtype(),
    )


def _bind_heading_asset(cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv, *, label: str):
    asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    if not isinstance(asset_cfg, SceneEntityCfg):
        raise TypeError(f"{label} asset_cfg must be SceneEntityCfg")
    entity = cast("Entity", env.scene[asset_cfg.name])
    heading_ref = np.asarray(entity.data.heading_w, dtype=get_global_dtype()).copy()
    return entity, heading_ref


def _refresh_heading_ref(
    env: ManagerBasedRlEnv,
    entity: "Entity",
    heading_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heading = np.asarray(entity.data.heading_w, dtype=get_global_dtype())
    fresh = env.episode_length_buf <= 1
    heading_ref[fresh] = heading[fresh]
    return heading, np_wrap_to_pi(heading - heading_ref)


class heading_hold(ManagerTermBase):
    """Reward holding each environment's spawn heading."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"std", "asset_cfg"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(f"heading_hold received unsupported parameters: {sorted(unexpected)}")
        self._std = _positive_real(cfg.params.get("std", 0.4), label="heading_hold std")
        self._entity, self._heading_ref = _bind_heading_asset(cfg, env, label="heading_hold")

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        _, error = _refresh_heading_ref(env, self._entity, self._heading_ref)
        return np.asarray(np.exp(-np.square(error) / self._std**2), dtype=get_global_dtype())


class heading_abs_cost(ManagerTermBase):
    """Linear spawn-heading error. Keeps a gradient after the Gaussian bowl dies."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - {"asset_cfg"}
        if unexpected:
            raise TypeError(f"heading_abs_cost received unsupported parameters: {sorted(unexpected)}")
        self._entity, self._heading_ref = _bind_heading_asset(cfg, env, label="heading_abs_cost")

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        _, error = _refresh_heading_ref(env, self._entity, self._heading_ref)
        return np.asarray(np.abs(error), dtype=get_global_dtype())


class running_spawn_progress(ManagerTermBase):
    """Reward capped world-frame speed along each environment's spawn heading."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        allowed = {"speed_cap", "asset_cfg"}
        unexpected = set(cfg.params) - allowed
        if unexpected:
            raise TypeError(
                f"running_spawn_progress received unsupported parameters: {sorted(unexpected)}"
            )
        self._cap = _positive_real(
            cfg.params.get("speed_cap", 1.2),
            label="running_spawn_progress speed_cap",
        )
        self._entity, self._heading_ref = _bind_heading_asset(
            cfg, env, label="running_spawn_progress"
        )

    def reset(self, env_ids: np.ndarray | slice | None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._heading_ref[ids] = self._entity.data.heading_w[ids]

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        heading, _ = _refresh_heading_ref(env, self._entity, self._heading_ref)
        velocity = np.nan_to_num(self._entity.data.root_link_lin_vel_w, nan=0.0)
        along_heading = velocity[:, 0] * np.cos(self._heading_ref) + velocity[:, 1] * np.sin(
            self._heading_ref
        )
        del heading
        return np.asarray(
            np.clip(along_heading, 0.0, self._cap) / self._cap,
            dtype=get_global_dtype(),
        )


class running_command_ranges_curriculum:
    """Advance a forward-only speed band while preserving the command type."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        command_name = cfg.params.get("command_name")
        if not isinstance(command_name, str) or not command_name:
            raise ValueError("running speed curriculum command_name must be non-empty")
        term_cfg = env.command_manager.get_term_cfg(command_name)
        if not isinstance(term_cfg, UniformVelocityCommandCfg):
            raise TypeError(
                "running speed curriculum requires UniformVelocityCommandCfg, "
                f"received {type(term_cfg).__name__}"
            )
        raw_stages = cfg.params.get("speed_stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("running speed curriculum speed_stages must be a non-empty list")
        stages: list[tuple[int, float, float]] = []
        previous_step = -1
        for index, stage in enumerate(raw_stages):
            if not isinstance(stage, dict) or set(stage) != {"step", "min_speed", "max_speed"}:
                raise ValueError(
                    "running speed curriculum stage "
                    f"{index} must contain step, min_speed, and max_speed"
                )
            step = stage["step"]
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or step < previous_step
            ):
                raise ValueError("running speed curriculum steps must be nonnegative and ordered")
            minimum = float(stage["min_speed"])
            maximum = float(stage["max_speed"])
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("running speed curriculum bounds must be finite")
            if minimum < 0.0 or minimum > maximum:
                raise ValueError(
                    f"running speed curriculum has invalid speed band {(minimum, maximum)}"
                )
            stages.append((step, minimum, maximum))
            previous_step = step
        self._term_cfg = term_cfg
        self._stages = tuple(stages)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: np.ndarray | slice,
        command_name: str,
        speed_stages: list[dict[str, Any]],
    ) -> dict[str, float]:
        del env_ids, command_name, speed_stages
        minimum, maximum = self._stages[0][1:]
        for step, stage_minimum, stage_maximum in self._stages:
            if env.common_step_counter >= step:
                minimum, maximum = stage_minimum, stage_maximum
        self._term_cfg.ranges.lin_vel_x = (minimum, maximum)
        return {"min_speed": minimum, "max_speed": maximum}


__all__ = [
    "heading_abs_cost",
    "heading_hold",
    "running_command_ranges_curriculum",
    "running_forward_progress",
    "running_planar_drift_cost",
    "running_spawn_progress",
]
