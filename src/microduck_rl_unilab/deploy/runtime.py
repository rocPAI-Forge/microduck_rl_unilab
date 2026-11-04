"""Onboard runtime for an exported MicroDuck policy.

Depends on numpy and onnxruntime only, so it can run on the robot. One
``Controller.step`` per 20 ms control period turns sensor readings and a
command into clamped servo goal positions, following ``docs/DEPLOY.md``:

1. ``ObservationBuilder`` assembles the 61-D observation from the gyro, the
   gravity direction, joint positions and velocities, the previous action, and
   the twist command.
2. ``PolicyRunner`` runs ``policy.onnx`` (the normalizer is inside the graph)
   and maps the action to ``q_des = action * scale + default``, clamped to the
   soft joint limits.
3. ``HeadingHold`` is the straight-line controller the policy was trained and
   scored with; ``Command`` covers stand, walk, straight, and turn-around.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

OBS_DIM = 61
NUM_JOINTS = 14
TURN_AROUND_SPEED_M_S = 0.2
TURN_AROUND_YAW_RAD_S = 0.4
_EXPECTED_SEGMENTS = (
    ("gyro", 3),
    ("gravity", 3),
    ("joint_pos", NUM_JOINTS),
    ("joint_vel", NUM_JOINTS),
    ("last_action", NUM_JOINTS),
    ("twist_cmd", 3),
    ("head_pose_cmd", 4),
    ("body_pose_cmd", 6),
)


@dataclass(frozen=True)
class DeployManifest:
    """The parts of ``deploy_manifest.json`` the runtime needs."""

    directory: Path
    onnx_path: Path
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    joint_limits: np.ndarray
    action_scale: float
    clamp: bool
    control_period_s: float

    @classmethod
    def load(cls, directory: str | Path) -> DeployManifest:
        directory = Path(directory)
        raw: dict[str, Any] = json.loads((directory / "deploy_manifest.json").read_text())
        segments = tuple((s["name"], int(s["dim"])) for s in raw["observation_segments"])
        if segments != _EXPECTED_SEGMENTS or int(raw["observation_dim"]) != OBS_DIM:
            raise ValueError(f"unsupported observation layout: {segments}")
        names = tuple(raw["action_joint_order"])
        if len(names) != NUM_JOINTS or int(raw["action_dim"]) != NUM_JOINTS:
            raise ValueError(f"expected {NUM_JOINTS} joints, got {len(names)}")
        default = np.asarray([raw["default_joint_pos_rad"][n] for n in names], dtype=np.float32)
        limits = np.asarray([raw["soft_joint_limits_rad"][n] for n in names], dtype=np.float32)
        return cls(
            directory=directory,
            onnx_path=directory / raw["onnx"],
            joint_names=names,
            default_joint_pos=default,
            joint_limits=limits,
            action_scale=float(raw["action_scale"]),
            clamp=bool(raw["goal_position_clamp"]),
            control_period_s=float(raw["control_period_s"]),
        )


@dataclass(frozen=True)
class SensorReading:
    """One control period of sensor data, joints in ``DeployManifest.joint_names`` order.

    ``gyro_b`` is the trunk angular velocity in the body frame (rad/s) and
    ``gravity_b`` the unit gravity direction in the body frame, both from the
    IMU. Joint positions are absolute servo angles in radians, the same frame
    as ``default_joint_pos``.
    """

    gyro_b: np.ndarray
    gravity_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


@dataclass(frozen=True)
class Command:
    """Forward speed (m/s) and yaw rate (rad/s); lateral is always 0."""

    forward_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    hold_heading: bool = False

    @classmethod
    def stand(cls) -> Command:
        return cls()

    @classmethod
    def walk(cls, forward_m_s: float, yaw_rate_rad_s: float = 0.0) -> Command:
        return cls(forward_m_s, yaw_rate_rad_s)

    @classmethod
    def straight(cls, forward_m_s: float) -> Command:
        """Walk along the heading and track held when the command started."""
        return cls(forward_m_s, 0.0, hold_heading=True)

    @classmethod
    def turn_around(cls, direction: int = 1) -> Command:
        """A 0.5 m radius turn; the policy has no pivot. ``direction`` +1 is left."""
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (left) or -1 (right)")
        return cls(TURN_AROUND_SPEED_M_S, direction * TURN_AROUND_YAW_RAD_S)


@dataclass
class HeadingHold:
    """``yaw = clip(2.5 * heading_error - 0.08 * cross_track, -0.35, 0.35)``.

    Heading error and cross-track are measured from the heading and position
    captured by ``reset``; cross-track is positive to the left of that ray.
    """

    heading_stiffness: float = 2.5
    cross_track_stiffness: float = 0.08
    yaw_limit: float = 0.35
    _heading_ref: float | None = field(default=None, init=False)
    _origin_xy: np.ndarray | None = field(default=None, init=False)

    def reset(self, heading: float, position_xy: np.ndarray | None = None) -> None:
        self._heading_ref = float(heading)
        self._origin_xy = None if position_xy is None else np.asarray(position_xy, dtype=np.float64)

    @property
    def active(self) -> bool:
        return self._heading_ref is not None

    def yaw_command(self, heading: float, position_xy: np.ndarray | None = None) -> float:
        if self._heading_ref is None:
            raise RuntimeError("HeadingHold.reset must be called first")
        error = math.atan2(
            math.sin(self._heading_ref - heading), math.cos(self._heading_ref - heading)
        )
        cross_track = 0.0
        if self._origin_xy is not None and position_xy is not None:
            displacement = np.asarray(position_xy, dtype=np.float64) - self._origin_xy
            lateral = np.array([-math.sin(self._heading_ref), math.cos(self._heading_ref)])
            cross_track = float(displacement @ lateral)
        yaw = self.heading_stiffness * error - self.cross_track_stiffness * cross_track
        return float(np.clip(yaw, -self.yaw_limit, self.yaw_limit))


class ObservationBuilder:
    """Assemble the 61-D observation; keeps the previous raw action."""

    def __init__(self, manifest: DeployManifest):
        self._default = manifest.default_joint_pos
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def build(self, reading: SensorReading, forward_m_s: float, yaw_rate_rad_s: float) -> np.ndarray:
        parts = (
            _vector(reading.gyro_b, 3, "gyro_b"),
            _vector(reading.gravity_b, 3, "gravity_b"),
            _vector(reading.joint_pos, NUM_JOINTS, "joint_pos") - self._default,
            _vector(reading.joint_vel, NUM_JOINTS, "joint_vel"),
            self.last_action,
            np.asarray([forward_m_s, 0.0, yaw_rate_rad_s], dtype=np.float32),
            np.zeros(4 + 6, dtype=np.float32),
        )
        return np.concatenate(parts).astype(np.float32, copy=False)


class PolicyRunner:
    """Run ``policy.onnx`` and turn its action into clamped goal positions."""

    def __init__(self, manifest: DeployManifest, providers: list[str] | None = None):
        import onnxruntime as ort

        self._manifest = manifest
        self._session = ort.InferenceSession(
            str(manifest.onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def act(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {obs.shape}")
        (action,) = self._session.run(None, {self._input: obs[None, :]})
        return np.asarray(action[0], dtype=np.float32)

    def targets(self, action: np.ndarray) -> np.ndarray:
        manifest = self._manifest
        q_des = action * manifest.action_scale + manifest.default_joint_pos
        if manifest.clamp:
            q_des = np.clip(q_des, manifest.joint_limits[:, 0], manifest.joint_limits[:, 1])
        return q_des.astype(np.float32, copy=False)


class Controller:
    """One call per control period: readings and a command in, goal positions out."""

    def __init__(self, deploy_dir: str | Path, providers: list[str] | None = None):
        self.manifest = DeployManifest.load(deploy_dir)
        self.observations = ObservationBuilder(self.manifest)
        self.policy = PolicyRunner(self.manifest, providers)
        self.heading_hold = HeadingHold()
        self._command = Command.stand()
        self.last_obs: np.ndarray | None = None

    def reset(self) -> None:
        self.observations.reset()
        self.heading_hold = HeadingHold()
        self._command = Command.stand()

    def set_command(
        self,
        command: Command,
        heading: float | None = None,
        position_xy: np.ndarray | None = None,
    ) -> None:
        """Switch command; a straight command captures ``heading`` as its reference."""
        if command.hold_heading:
            if heading is None:
                raise ValueError("a straight command needs the current heading")
            self.heading_hold.reset(heading, position_xy)
        self._command = command

    def step(
        self,
        reading: SensorReading,
        heading: float | None = None,
        position_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        command = self._command
        yaw = command.yaw_rate_rad_s
        if command.hold_heading:
            if heading is None:
                raise ValueError("a straight command needs the current heading every step")
            yaw = self.heading_hold.yaw_command(heading, position_xy)
        obs = self.observations.build(reading, command.forward_m_s, yaw)
        action = self.policy.act(obs)
        self.observations.last_action = action
        self.last_obs = obs
        return self.policy.targets(action)


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have {size} values, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


__all__ = [
    "Command",
    "Controller",
    "DeployManifest",
    "HeadingHold",
    "ObservationBuilder",
    "PolicyRunner",
    "SensorReading",
]
