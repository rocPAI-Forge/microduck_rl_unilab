"""Forward roll terms: accumulated forward rotation, completion, settling.

The duck starts standing and is paid for rotating forward about the lateral
axis it had when the episode began, up to one full turn. The axis is fixed per
episode because heading is undefined while the duck is upside down.

State shared by the terms lives on ``env._roll_state`` (lateral and forward
axes, accumulated rotation), created on first use and maintained by
``roll_progress``; the other terms read it one step late.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.managers import ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

FULL_TURN = 2.0 * math.pi


def _state(env: ManagerBasedRlEnv) -> SimpleNamespace:
    state = getattr(env, "_roll_state", None)
    if state is None:
        n = env.num_envs
        state = SimpleNamespace(
            lateral=np.tile([0.0, 1.0, 0.0], (n, 1)),
            forward=np.tile([1.0, 0.0, 0.0], (n, 1)),
            theta=np.zeros(n),
            peak=np.zeros(n),
            pending=np.ones(n, dtype=bool),
        )
        env._roll_state = state  # type: ignore[attr-defined]
    return state


def _ang_vel_w(env: ManagerBasedRlEnv) -> np.ndarray:
    robot = cast("Entity", env.scene["robot"])
    return np.nan_to_num(np.asarray(robot.data.root_link_ang_vel_w, dtype=np.float64), nan=0.0)


def _out(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=get_global_dtype())


class roll_progress(ManagerTermBase):
    """Forward pitch rate about the episode's lateral axis, clipped to [0, cap],
    paid until the accumulated rotation reaches ``turn`` radians."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"cap", "turn"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise ValueError(f"roll_progress got unexpected params {sorted(unexpected)}")
        self._cap = float(cfg.params.get("cap", 6.0))
        self._turn = float(cfg.params.get("turn", FULL_TURN))

    def reset(self, env_ids: np.ndarray | slice | None = None) -> None:
        state = _state(self._env)
        ids = slice(None) if env_ids is None else env_ids
        state.theta[ids] = 0.0
        state.peak[ids] = 0.0
        state.pending[ids] = True

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        state = _state(env)
        if state.pending.any():
            robot = cast("Entity", env.scene["robot"])
            ids = np.flatnonzero(state.pending)
            heading = np.asarray(robot.data.heading_w, dtype=np.float64)[ids]
            c, s = np.cos(heading), np.sin(heading)
            state.forward[ids] = np.stack((c, s, np.zeros_like(c)), axis=1)
            state.lateral[ids] = np.stack((-s, c, np.zeros_like(c)), axis=1)
            state.pending[ids] = False
        pitch_rate = np.sum(_ang_vel_w(env) * state.lateral, axis=1)
        active = state.theta < self._turn
        state.theta += pitch_rate * float(env.step_dt)
        state.peak = np.maximum(state.peak, state.theta)
        return _out(np.clip(pitch_rate, 0.0, self._cap) * active)


def roll_complete(env: ManagerBasedRlEnv, margin: float = 0.3) -> np.ndarray:
    """1 once the accumulated forward rotation is within ``margin`` of a full turn."""
    return _out((_state(env).theta >= FULL_TURN - margin).astype(np.float64))


def roll_settle(env: ManagerBasedRlEnv, std: float = 1.5, margin: float = 0.3) -> np.ndarray:
    """exp(-|w|^2 / std^2) after the roll is complete, else 0: stop rolling."""
    w = _ang_vel_w(env)
    done = _state(env).theta >= FULL_TURN - margin
    return _out(done * np.exp(-np.sum(w * w, axis=1) / (std * std)))


def roll_off_axis(env: ManagerBasedRlEnv, cap: float = 6.0) -> np.ndarray:
    """|angular velocity off the episode's lateral axis|, capped. Returns >= 0:
    negative weight (keeps it a forward roll, not a cartwheel or a twist)."""
    w = _ang_vel_w(env)
    lateral = _state(env).lateral
    off = w - np.sum(w * lateral, axis=1, keepdims=True) * lateral
    return _out(np.minimum(np.linalg.norm(off, axis=1), cap))


__all__ = ["roll_complete", "roll_off_axis", "roll_progress", "roll_settle"]
