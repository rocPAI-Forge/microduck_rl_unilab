# Ported from pollen-robotics/microduck_rl (develop, Apache-2.0):
# src/mjlab_microduck/tasks/mdp.py "Roulade (forward roll)" section and
# microduck_roulade_env_cfg.py; adapted to the NumPy manager runtime.
"""Roulade (forward roll over the head, landing on the feet) terms.

One policy rolls forward over the flat top of its head and lands standing.
The task signal is a rotation accumulator integrated from the body-frame
forward pitch rate, but only while some robot geom touches the floor (an
airborne whip earns nothing) and only while the lateral axis stays near
horizontal (a roll over the shoulder earns nothing). Progress pays increments
of the accumulator's running maximum, capped in rate. Every landing term is
gated on the maximum passing ~260 deg and on a head latch: the flat top of the
head touched the floor in the first half of the roll.

State lives on ``env._roulade_state`` (accum, max, paid, head latch, and this
step's head contact and head-top-down flags). ``roulade_progress`` updates it
each step and must come before the other roulade rewards; ``reset_roulade_state``
initializes it per episode.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from unilab.dtype_config import get_global_dtype
from unilab.envs.mdp import resolve_env_ids
from unilab.managers import ManagerTermBaseCfg, SceneEntityCfg
from unilab.tasks.locomotion.common.manager_terms import SensorTermBase

from microduck_rl_unilab.tasks.microduck.recovery_terms import _root_z
from microduck_rl_unilab.tasks.microduck.standup_terms import standing_composite_score

if TYPE_CHECKING:
    from unilab.base.entity import Entity
    from unilab.managers._types import ManagerBasedRlEnv

FULL_TURN = 2.0 * math.pi
# Head-latch window on the accumulator: the head plants at ~60-120 deg.
HEAD_LATCH_LO = math.radians(20.0)
HEAD_LATCH_HI = math.radians(170.0)
# Head-top axis in jaw_soft's frame (upstream measurement at HOME); it must
# point at the floor for a contact to count as the flat top of the head.
HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
HEAD_TOP_DOWN_MIN = 0.3
# Lateral-axis world z: full credit below sin 30 deg, none above sin 60 deg.
FLAT_FULL = 0.5
FLAT_ZERO = 0.866


def _state(env: ManagerBasedRlEnv) -> SimpleNamespace:
    state = getattr(env, "_roulade_state", None)
    if state is None:
        n = env.num_envs
        state = SimpleNamespace(
            accum=np.zeros(n),
            max=np.zeros(n),
            paid=np.zeros(n),
            latch=np.zeros(n, dtype=bool),
            head_contact=np.zeros(n, dtype=bool),
            head_top_down=np.zeros(n, dtype=bool),
        )
        env._roulade_state = state  # type: ignore[attr-defined]
    return state


def _robot(env: ManagerBasedRlEnv) -> Entity:
    return cast("Entity", env.scene["robot"])


def _out(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=get_global_dtype())


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lateral_axis_z(quat: np.ndarray) -> np.ndarray:
    """World z of the body's lateral (y) axis; 0 for any pure pitch."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _completion_gate(env: ManagerBasedRlEnv, gate_lo: float, gate_hi: float) -> np.ndarray:
    """Smoothstep on the accumulator's maximum, times the head latch."""
    state = _state(env)
    return _smoothstep((state.max - gate_lo) / max(gate_hi - gate_lo, 1e-6)) * state.latch


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


# --- accumulator and progress -------------------------------------------------------


class roulade_progress(SensorTermBase):
    """Paid increments of the accumulator's maximum, up to one full roll.

    reward = min(delta of min(max, 2 pi), max_paid_rate * dt) / (dt * 2 pi).
    Rotation faster than ``max_paid_rate`` forfeits the excess, so a whip
    collects less than a controlled roll. Also updates the shared state.
    """

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"support_sensor", "head_sensor", "max_paid_rate", "target_angle"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise ValueError(f"{self.name} got unexpected params {sorted(unexpected)}")
        self._support = self._bind((str(cfg.params.get("support_sensor", "robot_ground_contact")),))
        self._head = self._bind((str(cfg.params.get("head_sensor", "head_ground_contact")),))
        self._max_rate = float(cfg.params.get("max_paid_rate", 5.0))
        self._target = float(cfg.params.get("target_angle", FULL_TURN))
        ids, _ = _robot(env).find_bodies("^jaw_soft$")
        self._head_body = int(ids[0])

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        state = _state(env)
        robot = _robot(env)
        dt = float(env.step_dt)
        quat = np.nan_to_num(np.asarray(robot.data.root_link_quat_w, dtype=np.float64), nan=0.0)
        omega = np.nan_to_num(np.asarray(robot.data.root_link_ang_vel_b, dtype=np.float64)[:, 1], nan=0.0)
        supported = np.asarray(self._support.read()).reshape(env.num_envs, -1).max(axis=1) > 0
        flat = _smoothstep((FLAT_ZERO - np.abs(_lateral_axis_z(quat))) / (FLAT_ZERO - FLAT_FULL))
        state.accum = state.accum + omega * dt * supported * flat
        state.max = np.maximum(state.max, state.accum)

        head_q = np.nan_to_num(np.asarray(robot.data.body_link_quat_w, dtype=np.float64)[:, self._head_body], nan=0.0)
        top_world = _quat_rotate(head_q, np.broadcast_to(np.asarray(HEAD_TOP_AXIS), (env.num_envs, 3)))
        state.head_top_down = top_world[:, 2] < -HEAD_TOP_DOWN_MIN
        state.head_contact = np.asarray(self._head.read()).reshape(env.num_envs, -1).max(axis=1) > 0
        in_window = (state.accum > HEAD_LATCH_LO) & (state.accum < HEAD_LATCH_HI)
        state.latch = state.latch | (state.head_contact & in_window & state.head_top_down)

        new_paid = np.minimum(state.max, self._target)
        delta = np.clip(new_paid - np.minimum(state.paid, self._target), 0.0, self._max_rate * dt)
        state.paid = np.maximum(state.paid, new_paid)
        return _out(delta / (dt * self._target))


# --- reset --------------------------------------------------------------------------


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: np.ndarray | None,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = math.radians(5.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(340.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple[float, float] = (0.0, 3.0),
    tuck_overrides: dict[str, float] | None = None,
    tuck_factor_range: tuple[float, float] = (0.3, 1.0),
    joint_noise_std: float = 0.08,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """Standing start, or a mid-roll start (reverse curriculum).

    Standing: upright with small pitch/roll noise, HOME joints, accumulator 0.
    Mid-roll: pitched ``midroll_pitch_min..max`` into the roll, legs and neck
    lerped from HOME toward ``tuck_overrides`` (keyed by joint name), joint
    noise, optional forward angular velocity; the accumulator starts at the
    spawn pitch and the head latch is set (that phase is already behind it).
    Keeps the xy and yaw staged by ``reset_base``.
    """
    ids = resolve_env_ids(env, env_ids)
    if ids.size == 0:
        return
    robot = cast("Entity", env.scene[(asset_cfg or SceneEntityCfg("robot")).name])
    n, rng, dtype = ids.size, env.rng, get_global_dtype()
    state = _state(env)

    total = standing_prob + midroll_prob
    is_mid = rng.random(n) < (midroll_prob / max(total, 1e-6))
    yaw = rng.uniform(-math.pi, math.pi, size=n)
    pitch = rng.uniform(-standing_tilt_max, standing_tilt_max, size=n)
    mid_pitch = rng.uniform(midroll_pitch_min, midroll_pitch_max, size=n)
    pitch = np.where(is_mid, mid_pitch, pitch)
    roll_noise = max(standing_tilt_max, math.radians(5.0))
    roll = rng.uniform(-roll_noise, roll_noise, size=n)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    quat = np.stack(
        (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy),
        axis=1,
    )
    z = np.where(
        is_mid, rng.uniform(midroll_z_min, midroll_z_max, size=n), rng.uniform(standing_z_min, standing_z_max, size=n)
    )
    pose = robot.read_reset_root_pose(env_ids=ids)
    pose[:, 2] = z
    pose[:, 3:7] = quat
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)

    vel = np.zeros((n, 6))
    lo, hi = midroll_omega_range
    if hi > 0.0:
        omega_b = np.zeros((n, 3))
        omega_b[:, 1] = rng.uniform(lo, hi, size=n) * is_mid
        vel[:, 3:] = _quat_rotate(quat, omega_b)
    robot.write_root_link_velocity_to_sim(vel.astype(dtype), env_ids=ids)

    mid_ids = ids[is_mid]
    if mid_ids.size > 0:
        joints = np.array(robot.data.default_joint_pos[mid_ids], dtype=np.float64, copy=True)
        u = rng.uniform(tuck_factor_range[0], tuck_factor_range[1], size=mid_ids.size)
        for joint_name, angle in (tuck_overrides or {}).items():
            col = int(robot.find_joints(f"^{joint_name}$")[0][0])
            joints[:, col] = joints[:, col] + u * (float(angle) - joints[:, col])
        if joint_noise_std > 0.0:
            joints += rng.normal(0.0, joint_noise_std, size=joints.shape)
        robot.write_joint_state_to_sim(joints.astype(dtype), np.zeros_like(joints, dtype=dtype), env_ids=mid_ids)

    spawn = np.where(is_mid, mid_pitch, 0.0)
    state.accum[ids] = spawn
    state.max[ids] = spawn
    state.paid[ids] = spawn
    state.latch[ids] = is_mid


# --- shaping and landing rewards ------------------------------------------------------


def roulade_overspeed_penalty(env: ManagerBasedRlEnv, omega_max: float = 7.0) -> np.ndarray:
    """max(0, |w_y| - omega_max)^2. Returns >= 0: negative weight."""
    w = np.nan_to_num(np.asarray(_robot(env).data.root_link_ang_vel_b, dtype=np.float64)[:, 1], nan=0.0)
    return _out(np.square(np.clip(np.abs(w) - omega_max, 0.0, None)))


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
) -> np.ndarray:
    """Head on the floor x mid-roll window x forward-rate factor x (0.3 + 0.7 top-down).

    The rate factor keeps a face-planted duck resting on its head from earning it.
    """
    state = _state(env)
    w = np.nan_to_num(np.asarray(_robot(env).data.root_link_ang_vel_b, dtype=np.float64)[:, 1], nan=0.0)
    in_window = (state.accum > angle_lo) & (state.accum < angle_hi)
    top = 0.3 + 0.7 * state.head_top_down
    return _out(state.head_contact * in_window * np.clip(w / rate_norm, 0.0, 1.0) * top)


class roulade_landing_composite(standing_composite_score):
    """Standing composite score x completion gate: the landing annuity."""

    _allowed_params: ClassVar[frozenset[str]] = standing_composite_score._allowed_params | {"gate_lo", "gate_hi"}

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gate_lo = float(cfg.params.get("gate_lo", math.radians(260.0)))
        self._gate_hi = float(cfg.params.get("gate_hi", math.radians(330.0)))

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        score = super().__call__(env, **params)
        return _out(score * _completion_gate(env, self._gate_lo, self._gate_hi))


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv, gate_lo: float = math.radians(260.0), gate_hi: float = math.radians(330.0)
) -> np.ndarray:
    """Linear cos(tilt), clipped at 0, x completion gate."""
    q = np.asarray(_robot(env).data.root_link_quat_w, dtype=np.float64)
    upright = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return _out(np.clip(upright, 0.0, None) * _completion_gate(env, gate_lo, gate_hi))


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float = 0.115,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
) -> np.ndarray:
    """Broad height Gaussian x completion gate."""
    z = _root_z(env, _robot(env))
    return _out(np.exp(-np.square((z - target_height) / std)) * _completion_gate(env, gate_lo, gate_hi))


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float = 0.115,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
) -> np.ndarray:
    """Tight upright x height Gaussians x completion gate: the last mile."""
    robot = _robot(env)
    q = np.asarray(robot.data.root_link_quat_w, dtype=np.float64)
    tilt_sq = 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    z = _root_z(env, robot)
    score = np.exp(-tilt_sq / (upright_std * upright_std)) * np.exp(-np.square((z - target_height) / height_std))
    return _out(score * _completion_gate(env, gate_lo, gate_hi))


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float = 0.115,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
) -> np.ndarray:
    """-max(0, target - z) x completion gate. Returns <= 0: positive weight.

    Makes lying in a heap after the roll cost something; the gate keeps the
    roll itself untaxed.
    """
    z = _root_z(env, _robot(env))
    return _out(-np.clip(target_height - z, 0.0, None) * _completion_gate(env, gate_lo, gate_hi))


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
) -> np.ndarray:
    """Upward root velocity below ``max_height`` x late-roll gate: bootstraps the rise."""
    robot = _robot(env)
    vz = np.nan_to_num(np.asarray(robot.data.root_link_lin_vel_w, dtype=np.float64)[:, 2], nan=0.0)
    low = _root_z(env, robot) < max_height
    return _out(np.clip(vz, 0.0, None) * low * _completion_gate(env, gate_lo, gate_hi))


def roulade_regress_penalty(
    env: ManagerBasedRlEnv,
    window_lo: float = math.radians(30.0),
    window_hi: float = math.radians(260.0),
    cap: float = math.pi,
) -> np.ndarray:
    """(max - accum), capped, while the maximum is mid-roll. Returns >= 0: negative weight.

    Not upstream. Progress only pays new maxima, so falling back from a
    face-plant to the standing pose was free; this makes it cost.
    """
    state = _state(env)
    mid = (state.max > window_lo) & (state.max < window_hi)
    return _out(np.clip(state.max - state.accum, 0.0, cap) * mid)


def roulade_stall_penalty(
    env: ManagerBasedRlEnv,
    window_lo: float = math.radians(30.0),
    window_hi: float = math.radians(260.0),
) -> np.ndarray:
    """1 while the maximum is mid-roll and short of completion. Returns >= 0:
    negative weight. Not upstream: with ``roulade_regress_penalty`` alone,
    lying on the face at the maximum would cost nothing."""
    state = _state(env)
    return _out(((state.max > window_lo) & (state.max < window_hi)).astype(np.float64))


def roulade_flatness_penalty(env: ManagerBasedRlEnv) -> np.ndarray:
    """(lateral-axis world z)^2. Returns >= 0: negative weight."""
    q = np.asarray(_robot(env).data.root_link_quat_w, dtype=np.float64)
    return _out(np.square(_lateral_axis_z(q)))


def roulade_sagittal_penalty(env: ManagerBasedRlEnv) -> np.ndarray:
    """Body w_x^2 + w_z^2 (w_y is the roll axis). Returns >= 0: negative weight."""
    w = np.asarray(_robot(env).data.root_link_ang_vel_b, dtype=np.float64)
    return _out(w[:, 0] ** 2 + w[:, 2] ** 2)


def roulade_lateral_velocity_penalty(env: ManagerBasedRlEnv) -> np.ndarray:
    """Body lateral velocity^2. Returns >= 0: negative weight."""
    v = np.asarray(_robot(env).data.root_link_lin_vel_b, dtype=np.float64)
    return _out(v[:, 1] ** 2)


__all__ = [
    "reset_roulade_state",
    "roulade_flatness_penalty",
    "roulade_head_pivot",
    "roulade_height_after_roll",
    "roulade_landing_composite",
    "roulade_landing_sharp",
    "roulade_lateral_velocity_penalty",
    "roulade_overspeed_penalty",
    "roulade_progress",
    "roulade_regress_penalty",
    "roulade_rise_velocity",
    "roulade_sagittal_penalty",
    "roulade_stall_penalty",
    "roulade_stand_tax",
]
