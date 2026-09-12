"""Contracts for the forward-only MicroduckSprintFlat recipe."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.cli import package_root
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.envs.mdp import UniformVelocityCommand
from unilab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
from unilab.managers._types import ManagerBasedRlEnv

from microduck_rl_unilab.tasks.microduck.sprint_terms import (
    cross_track_abs_cost,
    head_facing_abs_cost,
    head_facing_hold,
    head_upright_hold,
    heading_abs_cost,
    heading_hold,
    joint_band_cost,
    running_alternate_steps,
    running_command_ranges_curriculum,
    running_contact_symmetry_cost,
    running_forward_progress,
    running_planar_drift_cost,
    running_straightness_cost,
    running_spawn_progress,
    running_leg_alternation,
    running_stride_length,
    running_trunk_bow,
)
from microduck_rl_unilab.tasks.microduck.manager_terms import (
    MicroduckVelocityCommand,
    MicroduckVelocityCommandCfg,
)
from microduck_rl_unilab.tasks.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_CRITIC_OBS_DIM,
    MICRODUCK_NUM_ACTION,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def _materialize_owner() -> tuple[Any, ManagerBasedRlEnvCfg]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()
    return cfg, env_cfg


def test_sprint_owner_specializes_velocity_contract() -> None:
    cfg, env_cfg = _materialize_owner()
    assert cfg.training.task_name == "MicroduckSprintFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.max_iterations == 11750
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(0.02)

    assert env_cfg.max_episode_seconds == pytest.approx(12.0)
    twist = env_cfg.commands["twist"]
    assert twist.resampling_time_range == [12.0, 12.0]
    assert twist.ranges.lin_vel_x == [0.2, 0.45]
    assert twist.ranges.lin_vel_y == [-0.02, 0.02]
    assert twist.ranges.ang_vel_z == [-0.05, 0.05]
    assert twist.rel_standing_envs == pytest.approx(0.03)
    assert twist.turn_in_place_fraction == pytest.approx(0.0)

    assert env_cfg.events["push_robot"] is None
    for name in (
        "head_pose_bias_weight",
        "standing_envs",
        "head_pose_range",
        "base_com_range",
        "head_com_range",
    ):
        assert env_cfg.curriculum[name] is None
    stages = env_cfg.curriculum["running_speed_range"].params["speed_stages"]
    assert stages[0] == {"step": 0, "min_speed": 0.2, "max_speed": 0.45}
    assert stages[-1] == {"step": 264000, "min_speed": 1.65, "max_speed": 2.2}

    rewards = env_cfg.rewards
    assert rewards["forward_progress"].func is running_forward_progress
    assert rewards["forward_progress"].weight == pytest.approx(5.0)
    assert rewards["forward_progress"].params["speed_cap"] == pytest.approx(2.4)
    assert rewards["spawn_progress"].func is running_spawn_progress
    assert rewards["spawn_progress"].weight == pytest.approx(1.5)
    assert rewards["heading_abs"].func is heading_abs_cost
    assert rewards["heading_abs"].weight == pytest.approx(-0.25)
    assert rewards["planar_drift"].func is running_planar_drift_cost
    assert rewards["heading_hold"].func is heading_hold
    assert rewards["heading_hold"].weight == pytest.approx(1.5)
    assert rewards["tracking_ang_vel"].weight == pytest.approx(0.5)
    assert rewards["leg_pose"].weight == pytest.approx(0.15)
    assert rewards["upright"].weight == pytest.approx(0.75)
    assert rewards["air_time"].weight == pytest.approx(0.0)
    assert rewards["leg_alternation"].func is running_leg_alternation
    assert rewards["leg_alternation"].weight == pytest.approx(2.5)
    assert rewards["leg_alternation"].params["asset_cfg"].joint_names == [
        "left_hip_pitch",
        "right_hip_pitch",
    ]
    assert rewards["stride_length"].func is running_stride_length
    assert rewards["stride_length"].weight == pytest.approx(1.5)
    assert rewards["stride_length"].params["stride_cap"] == pytest.approx(0.10)
    assert rewards["alternate_steps"].func is running_alternate_steps
    assert rewards["alternate_steps"].weight == pytest.approx(1.0)
    assert rewards["contact_symmetry"].func is running_contact_symmetry_cost
    assert rewards["contact_symmetry"].weight == pytest.approx(-1.0)
    assert "trunk_bow" not in rewards
    assert rewards["head_pose_tracking"].weight == pytest.approx(0.0)
    assert rewards["action_rate"].weight == pytest.approx(-0.02)
    # The head contract shapes speed discovery, not just robustification.
    assert rewards["head_upright"].func is head_upright_hold
    assert rewards["head_upright"].weight == pytest.approx(2.0)
    assert rewards["head_pitch_band"].func is joint_band_cost
    assert rewards["head_pitch_band"].weight == pytest.approx(-1.0)
    assert rewards["head_facing_abs"].func is head_facing_abs_cost
    assert rewards["head_facing_abs"].weight == pytest.approx(-1.0)
    assert rewards["head_roll_band"].func is joint_band_cost
    assert rewards["head_roll_band"].weight == pytest.approx(-2.0)
    assert rewards["head_roll_band"].params["band_deg"] == [[-10.0, 10.0]]
    assert rewards["head_roll_band"].params["asset_cfg"].joint_names == ["head_roll"]


def test_sprint_robust_owner_pins_speed_and_restores_disturbances() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_robust_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()

    assert cfg.training.task_name == "MicroduckSprintFlat"
    assert cfg.algo.max_iterations == 100
    twist = env_cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == [1.65, 2.2]
    assert twist.ranges.lin_vel_y == [-0.01, 0.01]
    assert twist.ranges.ang_vel_z == [-0.02, 0.02]
    head_pose = env_cfg.commands["head_pose"]
    assert head_pose.ranges == [[0.0, 0.0]] * 4
    assert head_pose.resampling_time_range == [12.0, 12.0]
    assert env_cfg.curriculum["running_speed_range"] is None
    assert env_cfg.curriculum["action_rate_weight"] is None
    assert env_cfg.rewards["forward_progress"].func is running_forward_progress
    assert env_cfg.rewards["forward_progress"].weight == pytest.approx(5.0)
    assert env_cfg.rewards["spawn_progress"].func is running_spawn_progress
    assert env_cfg.rewards["spawn_progress"].weight == pytest.approx(1.5)
    assert env_cfg.rewards["planar_drift"].weight == pytest.approx(-0.10)
    assert env_cfg.rewards["heading_hold"].weight == pytest.approx(2.0)
    assert env_cfg.rewards["heading_abs"].func is heading_abs_cost
    assert env_cfg.rewards["heading_abs"].weight == pytest.approx(-0.2)
    assert env_cfg.rewards["tracking_ang_vel"].weight == pytest.approx(0.8)
    assert env_cfg.rewards["head_pose_tracking"].func is head_facing_hold
    assert env_cfg.rewards["head_pose_tracking"].weight == pytest.approx(3.5)
    assert env_cfg.rewards["head_pose_tracking"].params["yaw_std"] == pytest.approx(0.15)
    assert env_cfg.rewards["head_pose_tracking"].params["roll_std"] == pytest.approx(0.12)
    upright = env_cfg.rewards["head_upright"]
    assert upright.func is head_upright_hold
    assert upright.weight == pytest.approx(3.0)
    assert upright.params["free_tilt_deg"] == pytest.approx(45.0)
    assert upright.params["zero_tilt_deg"] == pytest.approx(120.0)
    assert upright.params["asset_cfg"].body_names == ["jaw_soft"]
    facing_abs = env_cfg.rewards["head_facing_abs"]
    assert facing_abs.func is head_facing_abs_cost
    assert facing_abs.weight == pytest.approx(-2.0)
    assert facing_abs.params["asset_cfg"].joint_names == ["head_yaw", "head_roll"]
    band = env_cfg.rewards["head_pitch_band"]
    assert band.func is joint_band_cost
    assert band.weight == pytest.approx(-1.5)
    assert band.params["band_deg"] == [[-50.0, 45.0], [-50.0, 50.0]]
    assert band.params["asset_cfg"].joint_names == ["neck_pitch", "head_pitch"]
    assert env_cfg.rewards["head_pose_bias"].weight == pytest.approx(0.0)
    assert env_cfg.rewards["head_joint_vel"].weight == pytest.approx(-0.02)
    assert env_cfg.rewards["head_joint_vel"].params["asset_cfg"].joint_names == [
        "head_yaw",
        "head_roll",
    ]
    assert env_cfg.rewards["action_rate"].weight == pytest.approx(-0.10)
    push = env_cfg.events["push_robot"]
    assert push is not None
    assert push.mode == "interval"
    assert push.params["velocity_range"]["x"] == [-0.03, 0.03]
    pose = env_cfg.events["reset_base"].params["pose_range"]
    assert pose["roll"][1] == pytest.approx(math.radians(1.0))
    assert pose["pitch"][1] == pytest.approx(math.radians(1.0))


def test_sprint_rollfix_owner_ramps_only_head_roll_band() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_rollfix_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()

    assert env_cfg.curriculum["head_upright_weight"] is None
    assert env_cfg.curriculum["head_pitch_band_weight"] is None
    assert env_cfg.curriculum["head_facing_abs_weight"] is None
    assert env_cfg.curriculum["head_pose_tracking_weight"] is None
    stages = env_cfg.curriculum["head_roll_band_weight"].params["stages"]
    assert stages[0] == {"step": 0, "weight": -0.5}
    assert stages[-1] == {"step": 36000, "weight": -4.0}
    assert env_cfg.rewards["head_upright"].weight == pytest.approx(3.0)
    assert env_cfg.rewards["head_pitch_band"].weight == pytest.approx(-1.5)
    assert env_cfg.rewards["head_facing_abs"].weight == pytest.approx(-2.0)
    assert env_cfg.rewards["head_pose_tracking"].weight == pytest.approx(3.5)
    assert env_cfg.rewards["head_roll_band"].func is joint_band_cost
    assert env_cfg.rewards["head_roll_band"].weight == pytest.approx(-0.5)
    assert env_cfg.rewards["head_roll_band"].params["band_deg"] == [[-10.0, 10.0]]
    assert env_cfg.commands["twist"].ranges.lin_vel_x == [1.65, 2.2]


def test_sprint_straightfix_owner_extends_and_remembers_the_line() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightfix_flat/mujoco"]
        )
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(
            cfg, root_dir=ROOT_DIR, algo_name="ppo"
        ).build_task_env_cfg_override(),
    )
    env_cfg.validate()

    assert env_cfg.max_episode_seconds == pytest.approx(30.0)
    twist = env_cfg.commands["twist"]
    assert twist.resampling_time_range == [30.0, 30.0]
    assert twist.ranges.lin_vel_y == [0.0, 0.0]
    assert twist.ranges.ang_vel_z == [0.0, 0.0]
    assert twist.hold_spawn_heading is True
    assert twist.spawn_heading_stiffness == pytest.approx(2.5)
    assert twist.spawn_cross_track_stiffness == pytest.approx(0.08)
    assert twist.spawn_heading_yaw_rate_limit == pytest.approx(0.35)
    assert env_cfg.events["push_robot"] is None
    assert env_cfg.curriculum["head_roll_band_weight"] is None
    assert env_cfg.curriculum["heading_abs_weight"] is None
    assert env_cfg.rewards["forward_progress"].weight == pytest.approx(5.0)
    assert env_cfg.rewards["spawn_progress"].weight == pytest.approx(3.0)
    assert env_cfg.rewards["straightness"].func is running_straightness_cost
    assert env_cfg.rewards["straightness"].params["command_name"] == "twist"
    assert env_cfg.rewards["straightness"].weight == pytest.approx(-0.30)
    assert env_cfg.rewards["cross_track"].func is cross_track_abs_cost
    assert env_cfg.rewards["cross_track"].weight == pytest.approx(-0.40)
    assert env_cfg.rewards["heading_abs"].weight == pytest.approx(-0.25)
    assert env_cfg.rewards["tracking_ang_vel"].weight == pytest.approx(12.0)
    assert env_cfg.rewards["head_roll_band"].weight == pytest.approx(-4.0)


def test_sprint_steerfix_owner_restores_bidirectional_yaw_lessons() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_steerfix_flat/mujoco"]
        )
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSprintFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(
            cfg, root_dir=ROOT_DIR, algo_name="ppo"
        ).build_task_env_cfg_override(),
    )
    env_cfg.validate()

    twist = env_cfg.commands["twist"]
    assert twist.hold_spawn_heading is False
    assert twist.resampling_time_range == [4.0, 8.0]
    assert twist.ranges.ang_vel_z == [-0.35, 0.35]
    assert env_cfg.events["push_robot"] is None
    assert env_cfg.curriculum["head_roll_band_weight"] is None
    assert env_cfg.rewards["spawn_progress"] is None
    assert env_cfg.rewards["heading_abs"] is None
    assert env_cfg.rewards["heading_hold"] is None
    assert env_cfg.rewards["tracking_ang_vel"].weight == pytest.approx(12.0)
    assert env_cfg.rewards["planar_drift"].weight == pytest.approx(-0.20)
    assert env_cfg.rewards["head_roll_band"].weight == pytest.approx(-4.0)


def test_spawn_heading_capture_is_deferred_until_committed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        UniformVelocityCommand,
        "_update_command",
        lambda self, env_ids=None: None,
    )
    term = object.__new__(MicroduckVelocityCommand)
    term._hold_spawn_heading = True
    term._spawn_heading_stiffness = 2.5
    term._spawn_cross_track_stiffness = 0.1
    term._spawn_heading_yaw_rate_limit = 1.2
    term._spawn_heading = np.zeros(2, dtype=np.float32)
    term._spawn_position_xy = np.zeros((2, 2), dtype=np.float32)
    term._capture_spawn_heading = np.asarray([True, False])
    term.robot = SimpleNamespace(
        data=SimpleNamespace(
            heading_w=np.asarray([1.0, -1.0], dtype=np.float32),
            root_link_pos_w=np.zeros((2, 3), dtype=np.float32),
        )
    )
    term.vel_command_b = np.zeros((2, 3), dtype=np.float32)

    term._update_command(np.asarray([0], dtype=np.intp))
    np.testing.assert_allclose(term._spawn_heading, [1.0, 0.0])
    np.testing.assert_allclose(term.vel_command_b[0, 2], 0.0)

    term.robot.data.heading_w[0] = 1.2
    term._update_command(None)
    np.testing.assert_allclose(term.vel_command_b[0, 2], -0.5, atol=1.0e-6)

    term.robot.data.heading_w[0] = 1.0
    term.robot.data.root_link_pos_w[0, :2] = 2.0 * np.asarray(
        [-math.sin(1.0), math.cos(1.0)]
    )
    term._update_command(None)
    np.testing.assert_allclose(term.vel_command_b[0, 2], -0.2, atol=1.0e-6)


def test_linear_straightness_and_cross_track_keep_small_errors_visible() -> None:
    entity = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=np.asarray(
                [[1.3, 0.04, 0.0], [1.3, 0.0, 0.0]], dtype=np.float32
            ),
            root_link_ang_vel_b=np.asarray(
                [[0.0, 0.0, 0.04], [0.0, 0.0, 0.0]], dtype=np.float32
            ),
            root_link_pos_w=np.asarray(
                [[1.0, 0.15, 0.13], [1.0, 0.0, 0.13]], dtype=np.float32
            ),
            heading_w=np.zeros(2, dtype=np.float32),
        )
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            scene={"robot": entity},
            episode_length_buf=np.full(2, 10, dtype=np.int64),
        ),
    )
    straightness = running_straightness_cost(
        env, lateral_scale=0.2, yaw_rate_scale=0.2
    )
    np.testing.assert_allclose(straightness, [0.2, 0.0], atol=1e-6)

    env.command_manager = SimpleNamespace(
        get_command=lambda name: np.asarray(
            [[0.0, 0.04, 0.04], [0.0, 0.0, 0.0]], dtype=np.float32
        )
    )
    residual = running_straightness_cost(
        env,
        lateral_scale=0.2,
        yaw_rate_scale=0.2,
        command_name="twist",
    )
    np.testing.assert_allclose(residual, [0.0, 0.0], atol=1e-6)

    cross_track = cross_track_abs_cost(
        RewardTermCfg(
            func=cross_track_abs_cost,
            weight=-1.0,
            params={"distance_cap": 0.30},
        ),
        env,
    )
    cross_track._origin_xy[:] = 0.0
    np.testing.assert_allclose(cross_track(env), [0.5, 0.0], atol=1e-6)


def test_sprint_registry_is_mujoco_only() -> None:
    registry.ensure_registries()
    assert registry.list_registered_envs()["MicroduckSprintFlat"] == {
        "config_factory": "ManagerBasedRlEnvCfg",
        "available_backends": ["mujoco"],
    }


class _FakeEntity:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_link_lin_vel_b=np.asarray(
                [[-1.0, 0.1, 0.0], [0.7, -0.2, 0.0], [3.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            root_link_lin_vel_w=np.asarray(
                [[1.4, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, 1.4, 0.0]],
                dtype=np.float32,
            ),
            root_link_ang_vel_b=np.asarray(
                [[0.0, 0.0, 0.2], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            heading_w=np.asarray([0.0, 0.0, 0.5 * np.pi], dtype=np.float32),
        )


def _fake_env() -> ManagerBasedRlEnv:
    entity = _FakeEntity()
    command = np.asarray(
        [[0.4, 0.0, 0.0], [0.8, -0.1, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=3,
            scene={"robot": entity},
            command_manager=SimpleNamespace(get_command=lambda name: command),
            episode_length_buf=np.asarray([2, 2, 2], dtype=np.int64),
        ),
    )


def test_sprint_rewards_match_reference_math_and_wrap_heading() -> None:
    env = _fake_env()
    np.testing.assert_allclose(running_forward_progress(env, speed_cap=1.4), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        running_planar_drift_cost(env, lateral_weight=4.0),
        [0.08, 0.05, 0.0],
        atol=1e-6,
    )

    cast(Any, env.scene["robot"]).data.heading_w[:] = [0.0, 0.2, np.pi - 0.1]
    term = heading_hold(
        RewardTermCfg(
            func=heading_hold,
            weight=1.0,
            params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    cast(Any, env.scene["robot"]).data.heading_w[:] = [0.4, -0.2, -np.pi + 0.1]
    expected = np.exp(-np.square([0.4, -0.4, 0.2]) / 0.4**2)
    np.testing.assert_allclose(term(env), expected, rtol=1e-5)


def test_spawn_aligned_progress_penalizes_world_yaw() -> None:
    env = _fake_env()
    spawn = running_spawn_progress(
        RewardTermCfg(
            func=running_spawn_progress,
            weight=1.0,
            params={"speed_cap": 1.4, "asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    np.testing.assert_allclose(spawn(env), [1.0, 0.0, 1.0], atol=1e-6)
    abs_term = heading_abs_cost(
        RewardTermCfg(
            func=heading_abs_cost,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    cast(Any, env.scene["robot"]).data.heading_w[:] = [0.4, -0.2, 0.5 * np.pi + 0.3]
    np.testing.assert_allclose(abs_term(env), [0.4, 0.2, 0.3], atol=1e-6)


def test_head_facing_hold_allows_pitch_and_penalizes_yaw_roll() -> None:
    default = np.zeros((5, 4), dtype=np.float32)
    actual = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.4, -0.5, 0.0, 0.0],
            [-0.4, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    entity = SimpleNamespace(
        num_joints=4,
        data=SimpleNamespace(joint_pos=actual, default_joint_pos=default),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=5,
            scene={"robot": entity},
            command_manager=SimpleNamespace(get_command=lambda name: np.zeros((5, 4), dtype=np.float32)),
            episode_length_buf=np.ones(5, dtype=np.int64),
        ),
    )
    term = head_facing_hold(
        RewardTermCfg(
            func=head_facing_hold,
            weight=1.0,
            params={
                "command_name": "head_pose",
                "extra_slack": 0.45,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        env,
    )
    values = term(env)
    assert values[1] == pytest.approx(values[0])
    assert values[2] == pytest.approx(values[0])
    assert values[3] < values[0] - 0.2
    assert values[4] < values[0] - 0.2


def test_head_facing_abs_cost_still_grows_past_the_joint_stop() -> None:
    default = np.zeros((3, 2), dtype=np.float32)
    actual = np.asarray(
        [[0.0, 0.0], [0.1, -0.3], [0.0, 0.4363]],
        dtype=np.float32,
    )
    entity = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(joint_pos=actual, default_joint_pos=default),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=3, scene={"robot": entity}),
    )
    term = head_facing_abs_cost(
        RewardTermCfg(
            func=head_facing_abs_cost,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    np.testing.assert_allclose(term(env), [0.0, 0.2, 0.21815], atol=1e-5)
    # A Gaussian bowl is numerically flat at the 25 deg roll stop; this cost is not.
    saturated = term(env)[2]
    cast(Any, entity).data.joint_pos = np.asarray(
        [[0.0, 0.0], [0.1, -0.3], [0.0, 0.5]], dtype=np.float32
    )
    assert term(env)[2] > saturated


def test_joint_band_cost_only_charges_for_leaving_the_band() -> None:
    actual = np.radians(
        np.asarray(
            [
                [20.0, 20.0],  # home, inside both bands
                [-30.0, -45.0],  # exactly on the lower edges
                [-90.0, -75.0],  # the folded neck / counter-twist pose
                [70.0, 20.0],  # above the upper edge on one joint
            ],
            dtype=np.float64,
        )
    ).astype(np.float32)
    entity = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(joint_pos=actual, default_joint_pos=np.zeros_like(actual)),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=4, scene={"robot": entity}),
    )
    term = joint_band_cost(
        RewardTermCfg(
            func=joint_band_cost,
            weight=-1.0,
            params={
                "band_deg": [[-30.0, 60.0], [-45.0, 60.0]],
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        env,
    )
    expected = [
        0.0,
        0.0,
        math.radians(0.5 * (60.0 + 30.0)),
        math.radians(0.5 * 10.0),
    ]
    np.testing.assert_allclose(term(env), expected, atol=1e-6)


def test_head_upright_hold_frees_the_nod_and_punishes_inversion() -> None:
    root_half = math.sqrt(0.5)
    quats = np.asarray(
        [
            [root_half, 0.0, -root_half, 0.0],  # top of head at world up
            [0.92388, 0.0, -0.38268, 0.0],  # 45 deg forward nod, still free
            [1.0, 0.0, 0.0, 0.0],  # head-top horizontal
            [root_half, 0.0, root_half, 0.0],  # upside down
        ],
        dtype=np.float32,
    )
    entity = SimpleNamespace(
        num_bodies=1,
        data=SimpleNamespace(body_link_quat_w=quats[:, None, :]),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=4, scene={"robot": entity}),
    )
    term = head_upright_hold(
        RewardTermCfg(
            func=head_upright_hold,
            weight=1.0,
            params={
                "free_tilt_deg": 45.0,
                "zero_tilt_deg": 120.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=("jaw_soft",)),
            },
        ),
        env,
    )
    np.testing.assert_allclose(term(env), [1.0, 1.0, 0.414214, 0.0], atol=1e-4)


class _FakeFootView:
    backend_type = "fake"
    dimensions = (1, 1)

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def read(self) -> np.ndarray:
        return self._values


def _gait_env(*, contact: np.ndarray, feet_x: np.ndarray | None = None) -> ManagerBasedRlEnv:
    num_envs = contact.shape[0]
    identity = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_envs, 1))
    if feet_x is None:
        feet_x = np.zeros((num_envs, 2), dtype=np.float32)
    feet = np.zeros((num_envs, 2, 3), dtype=np.float32)
    feet[:, :, 0] = feet_x
    robot = SimpleNamespace(
        num_bodies=2,
        num_joints=2,
        data=SimpleNamespace(
            body_link_pos_w=feet,
            root_link_pos_w=np.zeros((num_envs, 3), dtype=np.float32),
            root_link_quat_w=identity,
            joint_pos=np.zeros((num_envs, 2), dtype=np.float32),
        ),
    )

    class _Scene:
        def __getitem__(self, name: str) -> Any:
            del name
            return robot

        def bind_sensor_data(self, names: tuple[str, ...]) -> _FakeFootView:
            del names
            return _FakeFootView(contact.astype(np.float32))

    command = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (num_envs, 1))
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            step_dt=0.02,
            scene=_Scene(),
            command_manager=SimpleNamespace(get_command=lambda name: command),
        ),
    )


def test_running_alternate_steps_scores_single_stance_and_flight() -> None:
    contact = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env = _gait_env(contact=contact)
    term = running_alternate_steps(
        RewardTermCfg(func=running_alternate_steps, weight=1.0, params={}),
        env,
    )
    np.testing.assert_allclose(term(env), [0.0, 1.0, 1.0, 0.6], atol=1e-5)


def test_running_contact_symmetry_cost_charges_a_limp() -> None:
    contact = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    env = _gait_env(contact=contact)
    term = running_contact_symmetry_cost(
        RewardTermCfg(
            func=running_contact_symmetry_cost,
            weight=-1.0,
            params={"tau_s": 0.02},
        ),
        env,
    )
    values = term(env)
    assert values[0] > values[1]


def _run_gait_steps(
    term: Any,
    env: ManagerBasedRlEnv,
    *,
    write: Any,
    steps: int = 100,
) -> float:
    """Drive a stateful gait term and return the mean reward of its second half."""
    values = []
    for step in range(steps):
        write(step)
        values.append(float(term(env)[0]))
    return float(np.mean(values[steps // 2 :]))


def test_running_stride_length_pays_for_scissoring_not_a_parked_split() -> None:
    # Column zero holds a constant 18 cm split; column one scissors by the same
    # amount. The old magnitude-based term scored both identically.
    contact = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    env = _gait_env(contact=contact)
    robot = env.scene["robot"]
    term = running_stride_length(
        RewardTermCfg(
            func=running_stride_length,
            weight=1.0,
            params={"stride_cap": 0.09, "tau_s": 0.2, "asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )

    def write(step: int) -> None:
        swing = 0.09 if step % 6 < 3 else -0.09
        robot.data.body_link_pos_w[0, 0, 0] = 0.09
        robot.data.body_link_pos_w[0, 1, 0] = -0.09
        robot.data.body_link_pos_w[1, 0, 0] = swing
        robot.data.body_link_pos_w[1, 1, 0] = -swing

    parked = []
    scissor = []
    for step in range(100):
        write(step)
        rewards = term(env)
        parked.append(float(rewards[0]))
        scissor.append(float(rewards[1]))
    assert np.mean(parked[50:]) < 0.05
    assert np.mean(scissor[50:]) > 0.7


def test_running_leg_alternation_rejects_a_parked_thigh_split() -> None:
    contact = np.asarray([[1.0, 0.0]], dtype=np.float32)
    split = math.radians(50.0)

    def build() -> Any:
        env = _gait_env(contact=contact)
        term = running_leg_alternation(
            RewardTermCfg(
                func=running_leg_alternation,
                weight=1.0,
                params={
                    "swing_cap_deg": 50.0,
                    "tau_s": 0.2,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            env,
        )
        return env, term

    # Mirrored frames: forward is negative on the left and positive on the
    # right, so a parked split holds both hips at the same forward angle.
    env, term = build()
    robot = env.scene["robot"]

    def write_parked(step: int) -> None:
        del step
        robot.data.joint_pos[0, 0] = -0.5 * split
        robot.data.joint_pos[0, 1] = -0.5 * split

    parked = _run_gait_steps(term, env, write=write_parked)

    env, term = build()
    robot = env.scene["robot"]

    def write_alternating(step: int) -> None:
        forward = 0.5 * split if step % 6 < 3 else -0.5 * split
        robot.data.joint_pos[0, 0] = -forward
        robot.data.joint_pos[0, 1] = -forward

    alternating = _run_gait_steps(term, env, write=write_alternating)

    assert parked < 0.05
    assert alternating > 0.7


def test_running_trunk_bow_rewards_forward_lean() -> None:
    from unilab.utils.rotation import np_quat_from_euler_xyz

    upright = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    bowed = np_quat_from_euler_xyz(
        np.asarray([0.0]), np.asarray([0.35]), np.asarray([0.0])
    ).astype(np.float32)
    entity = SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=np.vstack([upright, bowed]))
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(num_envs=2, scene={"robot": entity}),
    )
    term = running_trunk_bow(
        RewardTermCfg(
            func=running_trunk_bow,
            weight=1.0,
            params={"free_tilt_deg": 12.0, "max_tilt_deg": 40.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        env,
    )
    values = term(env)
    assert values[0] == pytest.approx(0.0, abs=1e-5)
    assert values[1] > 0.5


def test_running_speed_curriculum_mutates_only_forward_band() -> None:
    term_cfg = MicroduckVelocityCommandCfg(
        entity_name="robot",
        ranges=MicroduckVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.45),
            lin_vel_y=(-0.02, 0.02),
            ang_vel_z=(-0.05, 0.05),
        ),
        resampling_time_range=(12.0, 12.0),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            common_step_counter=24,
            command_manager=SimpleNamespace(get_term_cfg=lambda name: term_cfg),
        ),
    )
    stages = [
        {"step": 0, "min_speed": 0.2, "max_speed": 0.45},
        {"step": 24, "min_speed": 0.3, "max_speed": 0.55},
    ]
    cfg = CurriculumTermCfg(
        func=running_command_ranges_curriculum,
        params={"command_name": "twist", "speed_stages": stages},
    )
    term = running_command_ranges_curriculum(cfg, env)
    assert term(env, slice(None), "twist", stages) == {
        "min_speed": 0.3,
        "max_speed": 0.55,
    }
    assert term_cfg.ranges.lin_vel_x == (0.3, 0.55)
    assert term_cfg.ranges.lin_vel_y == (-0.02, 0.02)
    assert term_cfg.ranges.ang_vel_z == (-0.05, 0.05)


@pytest.mark.slow
def test_sprint_owner_builds_and_steps_real_mujoco_env() -> None:
    pytest.importorskip("mujoco")
    cfg, _ = _materialize_owner()
    env = cast(
        Any,
        registry.make(
            "MicroduckSprintFlat",
            sim_backend="mujoco",
            num_envs=1,
            env_cfg_override=BackendAdapter(
                cfg,
                root_dir=ROOT_DIR,
                algo_name="ppo",
            ).build_task_env_cfg_override(),
        ),
    )
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs["obs"].shape == (1, MICRODUCK_ACTOR_OBS_DIM)
        assert obs["critic"].shape == (1, MICRODUCK_CRITIC_OBS_DIM)
        state = env.step(np.zeros((1, MICRODUCK_NUM_ACTION), dtype=np.float32))
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert env.curriculum_manager.active_terms == [
            "action_rate_weight",
            "running_speed_range",
        ]
        assert "push_robot" not in env.event_manager.active_terms
    finally:
        env.close()
