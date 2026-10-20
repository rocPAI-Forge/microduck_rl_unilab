"""Focused reward tests for MicroDuck stateful manager terms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.envs.mdp import UniformPoseCommandCfg, UniformVelocityCommand, UniformVelocityCommandCfg
from unilab.managers import RewardTermCfg, SceneEntityCfg
from unilab.managers._types import ManagerBasedRlEnv
from microduck_rl_unilab.tasks.microduck.manager_terms import (
    MicroduckVelocityCommandCfg,
    body_pose_tracking,
    flight_phase,
    foot_air_time_biped,
    gait_phase_contact,
    head_lateral_cost,
    sagittal_step,
    fallen_cost,
    trunk_pitch_cost,
    yaw_rate_cost,
)


class _SensorScene:
    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self.values = values
        self.bound_names: tuple[str, ...] | None = None

    def bind_sensor_data(self, names: tuple[str, ...]):
        self.bound_names = names
        arrays = [self.values[name] for name in names]
        return SimpleNamespace(
            dimensions=tuple(array.shape[1] for array in arrays),
            backend_type="fake",
            read=lambda: np.concatenate(arrays, axis=1),
        )


class _RobotScene(_SensorScene):
    def __init__(self, values: dict[str, np.ndarray], robot: Any) -> None:
        super().__init__(values)
        self._robot = robot

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self._robot


def _env(contact: np.ndarray, command: np.ndarray) -> tuple[ManagerBasedRlEnv, _SensorScene]:
    scene = _SensorScene(
        {
            "left_foot_contact": contact[:, 0:1],
            "right_foot_contact": contact[:, 1:2],
        }
    )
    command_manager = SimpleNamespace(get_command=lambda name: command)
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=contact.shape[0],
            step_dt=0.1,
            scene=scene,
            command_manager=command_manager,
        ),
    )
    return env, scene


def _term(term_type: type, env: ManagerBasedRlEnv, **params: Any):
    return term_type(
        RewardTermCfg(func=term_type, weight=1.0, params=params),
        env,
    )


def test_foot_air_time_biped_rewards_single_stance_only() -> None:
    env, scene = _env(
        np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        np.asarray([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
    )
    term = _term(
        foot_air_time_biped,
        env,
        threshold=0.3,
        command_threshold=0.01,
        command_name="twist",
    )

    np.testing.assert_allclose(term(env), [0.1, 0.0])
    assert scene.bound_names == ("left_foot_contact", "right_foot_contact")
    term.reset(np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(term._air_time[0], 0.0)
    np.testing.assert_array_equal(term._contact_time[0], 0.0)


def test_flight_phase_penalizes_only_double_air() -> None:
    env, _ = _env(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
    )
    term = _term(flight_phase, env)
    np.testing.assert_array_equal(term(env), [1.0, 0.0])


def test_gait_phase_contact_requires_the_commanded_single_stance() -> None:
    contact = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    velocity = np.asarray(
        [
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    phase = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env, _ = _env(contact, velocity)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: phase if name == "gait_phase" else velocity
    )
    term = _term(
        gait_phase_contact,
        env,
        phase_command_name="gait_phase",
        velocity_command_name="twist",
        command_threshold=0.01,
    )

    np.testing.assert_array_equal(term(env), [1.0, 1.0, 0.25, 0.25, 0.0])


def test_head_lateral_cost_ignores_pitch_and_penalizes_yaw_velocity() -> None:
    entity = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(
            joint_pos=np.asarray([[0.2, 0.0]], dtype=np.float32),
            default_joint_pos=np.zeros((1, 2), dtype=np.float32),
            joint_vel=np.asarray([[0.0, 1.5]], dtype=np.float32),
        ),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=1,
            scene={"robot": entity},
            command_manager=SimpleNamespace(
                get_command=lambda name: np.zeros((1, 4), dtype=np.float32)
            ),
        ),
    )
    term = _term(
        head_lateral_cost,
        env,
        command_name="head_pose",
        command_indices=[2, 3],
        pos_std=0.1,
        vel_std=1.5,
        asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1]),
    )

    np.testing.assert_allclose(term(env), [2.5])


def test_sagittal_step_rewards_swing_foot_reach_in_single_stance() -> None:
    contact = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    command = np.asarray(
        [
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    feet = np.zeros((6, 2, 3), dtype=np.float32)
    feet[0, 1, 0] = 0.08
    feet[1, 0, 1] = 0.04
    feet[4, 1, 0] = 0.08
    feet[5, 1, 0] = -0.03
    env, _ = _env(contact, command)
    env.scene = _RobotScene(
        env.scene.values,
        SimpleNamespace(
            body_names=["ankle_left", "ankle_right"],
            data=SimpleNamespace(
                heading_w=np.asarray([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                root_link_pos_w=np.zeros((6, 3), dtype=np.float32),
                root_link_quat_w=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (6, 1)),
                body_link_pos_w=feet,
            ),
        ),
    )
    term = _term(
        sagittal_step,
        env,
        command_name="twist",
        command_threshold=0.01,
        target_length=0.08,
        asset_cfg=SceneEntityCfg("robot", body_names=("ankle_left", "ankle_right")),
    )

    np.testing.assert_allclose(term(env), [1.0, 0.5, 0.0, 0.0, 0.0, 0.0], rtol=1e-6, atol=1e-6)


def test_sagittal_step_drops_out_in_a_nose_dive() -> None:
    contact = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    command = np.asarray([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
    feet = np.zeros((2, 2, 3), dtype=np.float32)
    feet[:, 1, 0] = 0.08
    half = np.radians(45.0) / 2.0
    env, _ = _env(contact, command)
    env.scene = _RobotScene(
        env.scene.values,
        SimpleNamespace(
            body_names=["ankle_left", "ankle_right"],
            data=SimpleNamespace(
                heading_w=np.zeros(2, dtype=np.float32),
                root_link_pos_w=np.zeros((2, 3), dtype=np.float32),
                root_link_quat_w=np.asarray(
                    [
                        [np.cos(half), 0.0, np.sin(half), 0.0],
                        [np.cos(half), 0.0, -np.sin(half), 0.0],
                    ],
                    dtype=np.float32,
                ),
                body_link_pos_w=feet,
            ),
        ),
    )
    term = _term(
        sagittal_step,
        env,
        command_name="twist",
        command_threshold=0.01,
        target_length=0.08,
        pitch_gate_std=0.25,
        asset_cfg=SceneEntityCfg("robot", body_names=("ankle_left", "ankle_right")),
    )

    np.testing.assert_array_less(term(env), 1e-3)


def test_fallen_cost_is_one_only_past_the_tilt_limit() -> None:
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        projected_gravity_b=np.asarray(
                            [[0.0, 0.0, -1.0], [0.0, 0.0, -0.1]],
                            dtype=np.float32,
                        )
                    )
                )
            },
        ),
    )
    term = _term(
        fallen_cost,
        env,
        limit_angle=1.2217304763960306,
        asset_cfg=SceneEntityCfg("robot"),
    )

    np.testing.assert_allclose(term(env), [0.0, 1.0])


def test_trunk_pitch_cost_ignores_roll() -> None:
    pitch_half = np.radians(20.0) / 2.0
    roll_half = np.radians(30.0) / 2.0
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_quat_w=np.asarray(
                            [
                                [np.cos(pitch_half), 0.0, np.sin(pitch_half), 0.0],
                                [np.cos(roll_half), np.sin(roll_half), 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            },
        ),
    )
    term = _term(trunk_pitch_cost, env, asset_cfg=SceneEntityCfg("robot"))

    expected_pitch = np.square(np.radians(20.0))
    np.testing.assert_allclose(term(env), [expected_pitch, 0.0], rtol=1e-5, atol=1e-5)


def test_yaw_rate_cost_tracks_commanded_yaw() -> None:
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_ang_vel_b=np.asarray([[0.0, 0.0, 0.3], [0.0, 0.0, -0.1]], dtype=np.float32)
                    )
                )
            },
            command_manager=SimpleNamespace(
                get_command=lambda name: np.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]], dtype=np.float32)
            ),
        ),
    )
    term = _term(yaw_rate_cost, env, command_name="twist", asset_cfg=SceneEntityCfg("robot"))

    np.testing.assert_allclose(term(env), [0.04, 0.0], rtol=1e-6, atol=1e-6)


def test_turn_in_place_bucket_overrides_independent_standing_sample(monkeypatch) -> None:
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            step_dt=0.02,
            rng=np.random.default_rng(4),
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_lin_vel_b=np.zeros((2, 3), dtype=np.float32),
                        root_link_ang_vel_b=np.zeros((2, 3), dtype=np.float32),
                        heading_w=np.zeros(2, dtype=np.float32),
                        root_link_pos_w=np.zeros((2, 3), dtype=np.float32),
                    )
                )
            },
        ),
    )
    cfg = MicroduckVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0, 1.0),
        rel_standing_envs=1.0,
        turn_in_place_fraction=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    term = cfg.build(env)

    def sample_standing(self, env_ids):
        self.vel_command_b[env_ids] = (0.2, 0.1, 0.0)
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = True

    monkeypatch.setattr(UniformVelocityCommand, "_resample_command", sample_standing)
    ids = np.asarray([0, 1], dtype=np.int32)
    term._resample_command(ids)
    term._update_command()

    np.testing.assert_array_equal(term.is_standing_env, False)
    np.testing.assert_array_equal(term.command[:, :2], 0.0)
    assert np.all(np.abs(term.command[:, 2]) >= 0.4)
    np.testing.assert_array_equal(term.vel_command_w, term.command)


def test_contact_terms_read_first_component_from_vector_force_sensors() -> None:
    scene = _SensorScene(
        {
            "left_foot_contact": np.asarray([[0.0, 4.0, 5.0], [1.0, 0.0, 0.0]]),
            "right_foot_contact": np.asarray([[0.0, 6.0, 7.0], [0.0, 8.0, 9.0]]),
        }
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=2, step_dt=0.1, scene=scene),
    )

    term = _term(flight_phase, env)

    np.testing.assert_array_equal(term(env), [1.0, 0.0])


def test_contact_terms_fail_closed_when_sensor_is_missing() -> None:
    env, scene = _env(
        np.ones((1, 2), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )
    del scene.values["right_foot_contact"]
    with pytest.raises(KeyError, match="right_foot_contact"):
        _term(flight_phase, env)


def test_uniform_pose_command_resamples_from_live_ranges() -> None:
    """Step-staged command curricula mutate the live term cfg; the next
    resample must sample from the widened ranges without rebuilding the term."""
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=8, rng=np.random.default_rng(0)))
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=[[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]],
    )
    term = cfg.build(env)
    env_ids = np.arange(8)
    term.reset(env_ids)
    assert float(np.abs(term.command).max()) <= 0.07

    cfg.ranges = [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]]
    term.reset(env_ids)
    assert float(np.abs(term.command[:, 2]).max()) > 0.07
    assert float(np.abs(term.command).max()) <= 1.4


def test_uniform_pose_command_rejects_range_width_change() -> None:
    env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=2, rng=np.random.default_rng(0)))
    cfg = UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=[[-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015]],
    )
    term = cfg.build(env)
    cfg.ranges = [[-1.0, 1.0]] * 3
    with pytest.raises(ValueError, match="width changed"):
        term.reset(np.arange(2))


class _PoseScene:
    def __init__(self, position: np.ndarray, quat: np.ndarray) -> None:
        self.env_origins = np.zeros((position.shape[0], 3), dtype=np.float32)
        self._robot = SimpleNamespace(
            data=SimpleNamespace(root_link_pos_w=position, root_link_quat_w=quat)
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self._robot


def _pose_env(position: np.ndarray, quat: np.ndarray, command: np.ndarray) -> ManagerBasedRlEnv:
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=position.shape[0],
            scene=_PoseScene(position, quat),
            command_manager=SimpleNamespace(get_command=lambda name: command),
        ),
    )


def test_body_pose_tracking_matches_upstream_6d_gaussian_mean() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # env0 is exactly at the nominal pose with a zero command; env1 is one
    # xy_std off in x and one z_std off in z: two axes at exp(-1), four at 1.
    position = np.asarray([[0.0, 0.0, 0.095], [0.05, 0.0, 0.095 + 0.02]], dtype=np.float32)
    command = np.zeros((2, 6), dtype=np.float32)
    env = _pose_env(position, identity, command)
    expected = (4.0 + 2.0 * np.exp(-1.0)) / 6.0
    np.testing.assert_allclose(body_pose_tracking(env), [1.0, expected], rtol=1e-6)

    # One angle_std of roll error: quat for roll=15deg about x.
    half = np.deg2rad(15.0) / 2.0
    rolled = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [np.cos(half), np.sin(half), 0.0, 0.0]], dtype=np.float32
    )
    position = np.asarray([[0.0, 0.0, 0.095], [0.0, 0.0, 0.095]], dtype=np.float32)
    env = _pose_env(position, rolled, command)
    expected = (5.0 + np.exp(-1.0)) / 6.0
    np.testing.assert_allclose(body_pose_tracking(env), [1.0, expected], rtol=1e-5)


def test_body_pose_tracking_uses_env_origins_and_command_targets() -> None:
    # A root parked at the env origin plus the commanded xyz delta scores 1.
    position = np.asarray([[2.01, -1.01, 0.105]], dtype=np.float32)
    quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    command = np.asarray([[0.01, -0.01, 0.01, 0.0, 0.0, 0.0]], dtype=np.float32)
    env = _pose_env(position, quat, command)
    scene = cast(Any, env.scene)
    scene.env_origins = np.asarray([[2.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(body_pose_tracking(env), [1.0], rtol=1e-6)

    bad_env = _pose_env(position, quat, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="width 6"):
        body_pose_tracking(bad_env)
