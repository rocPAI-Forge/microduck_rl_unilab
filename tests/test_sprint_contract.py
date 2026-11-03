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
    action_clip_margin_curriculum,
    action_target_band_cost,
    action_target_overshoot_cost,
    cross_track_abs_cost,
    head_facing_abs_cost,
    head_tilt_excess_cost,
    head_upright_hold,
    heading_hold,
    joint_band_cost,
    running_command_ranges_curriculum,
    running_forward_progress,
    running_leg_alternation,
    running_planar_drift_cost,
    running_spawn_progress,
    running_straightness_cost,
    running_stride_length,
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
    assert rewards["planar_drift"].func is running_planar_drift_cost
    assert rewards["heading_hold"].func is heading_hold
    assert rewards["heading_hold"].weight == pytest.approx(1.5)
    assert rewards["tracking_ang_vel"].weight == pytest.approx(0.5)
    assert rewards["leg_pose"].weight == pytest.approx(0.15)
    assert rewards["upright"].weight == pytest.approx(0.75)
    assert rewards["air_time"].weight == pytest.approx(0.0)
    assert rewards["head_pose_tracking"].weight == pytest.approx(0.0)
    assert rewards["action_rate"].weight == pytest.approx(-0.02)


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
    assert env_cfg.curriculum["running_speed_range"] is None
    assert env_cfg.curriculum["action_rate_weight"] is None
    assert env_cfg.rewards["action_rate"].weight == pytest.approx(-0.10)
    push = env_cfg.events["push_robot"]
    assert push is not None
    assert push.mode == "interval"
    assert push.params["velocity_range"]["x"] == [-0.03, 0.03]
    pose = env_cfg.events["reset_base"].params["pose_range"]
    assert pose["roll"][1] == pytest.approx(math.radians(1.0))
    assert pose["pitch"][1] == pytest.approx(math.radians(1.0))


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
            root_link_ang_vel_b=np.asarray(
                [[0.0, 0.0, 0.2], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            heading_w=np.asarray([0.0, 0.2, np.pi - 0.1], dtype=np.float32),
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


def test_sprint_rollfix_owner_declares_head_and_gait_guards() -> None:
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
    rewards = env_cfg.rewards
    assert rewards["head_upright"].func is head_upright_hold
    assert rewards["head_upright"].weight == pytest.approx(3.0)
    assert rewards["head_roll_band"].func is joint_band_cost
    assert rewards["head_roll_band"].weight == pytest.approx(-0.5)
    assert rewards["head_roll_band"].params["band_deg"] == [[-10.0, 10.0]]
    assert rewards["leg_alternation"].func is running_leg_alternation
    assert rewards["stride_length"].func is running_stride_length
    stages = env_cfg.curriculum["head_roll_band_weight"].params["stages"]
    assert stages[-1] == {"step": 36000, "weight": -4.0}
    assert env_cfg.curriculum["head_upright_weight"] is None


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


def test_sprint_straightspeed_owner_pays_for_ray_speed_without_opening_the_line() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightspeed_flat/mujoco"]
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
    assert twist.hold_spawn_heading is True
    assert twist.spawn_heading_stiffness == pytest.approx(2.5)
    assert twist.spawn_cross_track_stiffness == pytest.approx(0.08)
    assert twist.ranges.lin_vel_x == [1.65, 2.20]
    assert env_cfg.events["push_robot"] is None
    rewards = env_cfg.rewards
    assert rewards["forward_progress"].weight == pytest.approx(5.0)
    assert rewards["forward_progress"].params["speed_cap"] == pytest.approx(2.4)
    assert rewards["spawn_progress"].func is running_spawn_progress
    assert rewards["spawn_progress"].weight == pytest.approx(3.0)
    assert rewards["spawn_progress"].params["speed_cap"] == pytest.approx(1.6)
    assert rewards["heading_abs"].weight == pytest.approx(-0.25)
    assert rewards["straightness"].weight == pytest.approx(-0.30)
    assert rewards["cross_track"].weight == pytest.approx(-0.40)
    assert rewards["tracking_ang_vel"].weight == pytest.approx(12.0)
    assert rewards["head_roll_band"].weight == pytest.approx(-4.0)
    spawn_stages = env_cfg.curriculum["spawn_progress_weight"].params["stages"]
    heading_stages = env_cfg.curriculum["heading_abs_weight"].params["stages"]
    assert spawn_stages[0] == {"step": 0, "weight": 3.0}
    assert spawn_stages[-1] == {"step": 48000, "weight": 5.0}
    assert heading_stages[0] == {"step": 0, "weight": -0.25}
    assert heading_stages[-1] == {"step": 48000, "weight": -0.75}


def test_sprint_straighthold_owner_pins_weights_and_prices_falls() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straighthold_flat/mujoco"]
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

    assert env_cfg.max_episode_seconds == pytest.approx(60.0)
    twist = env_cfg.commands["twist"]
    assert twist.hold_spawn_heading is True
    assert twist.resampling_time_range == [60.0, 60.0]
    push = env_cfg.events["push_robot"]
    assert push is not None
    assert push.params["velocity_range"]["x"] == [-0.05, 0.05]
    assert env_cfg.curriculum["spawn_progress_weight"] is None
    assert env_cfg.curriculum["heading_abs_weight"] is None
    rewards = env_cfg.rewards
    assert rewards["tracking_lin_vel"].weight == pytest.approx(1.0)
    assert rewards["tracking_lin_vel"].params["std"] == pytest.approx(1.0)
    assert rewards["spawn_progress"].weight == pytest.approx(5.0)
    assert rewards["spawn_progress"].params["speed_cap"] == pytest.approx(1.6)
    assert rewards["heading_abs"].weight == pytest.approx(-0.75)
    assert rewards["cross_track"].weight == pytest.approx(-0.8)
    assert rewards["straightness"].weight == pytest.approx(-0.5)
    assert rewards["fall"].weight == pytest.approx(-300.0)
    assert rewards["head_tilt_excess"].func is head_tilt_excess_cost
    assert rewards["head_tilt_excess"].weight == pytest.approx(0.0)
    assert rewards["head_roll_band"].weight == pytest.approx(-4.0)


def test_sprint_straightcmd_owner_pays_for_tracking_not_progress() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightcmd_flat/mujoco"]
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
    assert twist.hold_spawn_heading is True
    assert twist.ranges.lin_vel_x == [0.3, 1.2]
    assert twist.resampling_time_range == [6.0, 12.0]
    assert env_cfg.max_episode_seconds == pytest.approx(60.0)
    assert env_cfg.events["push_robot"].params["velocity_range"]["x"] == [-0.10, 0.10]
    assert env_cfg.events["base_com"].params["com_range"]["x"] == [-0.008, 0.008]
    assert env_cfg.events["head_com"].params["com_range"]["x"] == [-0.005, 0.005]
    assert env_cfg.events["reset_base"].params["pose_range"]["roll"] == pytest.approx(
        [-0.0262, 0.0262]
    )
    rewards = env_cfg.rewards
    assert rewards["forward_progress"] is None
    assert rewards["spawn_progress"] is None
    assert twist.high_speed_fraction == pytest.approx(0.0)
    assert rewards["tracking_lin_vel"].weight == pytest.approx(4.0)
    assert rewards["tracking_lin_vel"].params["std"] == pytest.approx(0.6)
    assert rewards["tracking_lin_vel_fine"].weight == pytest.approx(10.0)
    assert rewards["tracking_lin_vel_fine"].params["std"] == pytest.approx(0.25)
    assert rewards["head_tilt_excess"].weight == pytest.approx(-40.0)
    assert rewards["fall"].weight == pytest.approx(-300.0)
    assert rewards["heading_abs"].weight == pytest.approx(-0.75)
    assert rewards["cross_track"].weight == pytest.approx(-0.8)


def test_sprint_straightrange_owner_ramps_the_target_overshoot_cost() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightrange_flat/mujoco"]
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
    assert twist.high_speed_fraction == pytest.approx(0.4)
    assert list(twist.high_speed_range) == [0.9, 1.2]
    term = env_cfg.rewards["action_target_overshoot"]
    assert term.func is action_target_overshoot_cost
    assert term.params["action_name"] == "joint_pos"
    stages = env_cfg.curriculum["action_target_overshoot_weight"].params["stages"]
    assert stages[0] == {"step": 0, "weight": -0.002}
    assert stages[-1] == {"step": 36000, "weight": -0.06}
    assert env_cfg.rewards["head_pitch_band"].weight == pytest.approx(-1.5)


def test_sprint_straightclip_owner_clamps_targets_and_pins_the_cost() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightclip_flat/mujoco"]
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

    assert env_cfg.actions["joint_pos"].clip_target_to_joint_limits is True
    assert env_cfg.actions["joint_pos"].clip_target_margin == pytest.approx(0.5)
    assert env_cfg.curriculum["action_target_overshoot_weight"] is None
    margin = env_cfg.curriculum["action_clip_margin"]
    assert margin.func is action_clip_margin_curriculum
    assert margin.params["stages"][0] == {"step": 0, "margin": 0.5}
    assert margin.params["stages"][-1] == {"step": 48000, "margin": 0.0}
    assert env_cfg.rewards["action_target_overshoot"].weight == pytest.approx(-0.3)
    assert env_cfg.rewards["head_pitch_band"].weight == pytest.approx(-1.5)


def test_sprint_straightneck_owner_ramps_only_the_neck_band() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straightneck_flat/mujoco"]
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
    assert twist.ranges.lin_vel_x == [0.3, 1.2]
    assert twist.high_speed_fraction == pytest.approx(0.4)
    assert list(twist.high_speed_range) == [0.9, 1.2]
    assert env_cfg.events["reset_base"].params["pose_range"]["roll"] == pytest.approx(
        [-0.0436, 0.0436]
    )
    stages = env_cfg.curriculum["head_pitch_target_band_weight"].params["stages"]
    assert stages[0] == {"step": 0, "weight": -1.5}
    assert stages[-1] == {"step": 48000, "weight": -12.0}
    target_band = env_cfg.rewards["head_pitch_target_band"]
    assert target_band.func is action_target_band_cost
    assert target_band.params["joint_names"] == ["neck_pitch", "head_pitch"]
    assert target_band.params["band_deg"] == [[-50.0, 45.0], [-50.0, 50.0]]
    assert env_cfg.curriculum["action_target_overshoot_weight"] is None
    assert env_cfg.actions["joint_pos"].clip_target_to_joint_limits is True
    assert env_cfg.actions["joint_pos"].clip_target_margin == pytest.approx(0.0)
    assert env_cfg.curriculum["action_clip_margin"] is None
    rewards = env_cfg.rewards
    assert rewards["action_target_overshoot"].weight == pytest.approx(-0.3)
    assert rewards["head_pitch_band"].func is joint_band_cost
    assert rewards["head_pitch_band"].params["band_deg"] == [[-50.0, 45.0], [-50.0, 50.0]]
    assert rewards["tracking_lin_vel_fine"].weight == pytest.approx(10.0)
    assert rewards["head_tilt_excess"].weight == pytest.approx(-40.0)


def test_sprint_straighthead_owner_caps_the_band_and_pins_the_head() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            "config", overrides=["task=microduck_sprint_straighthead_flat/mujoco"]
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
    assert twist.ranges.lin_vel_x == [0.3, 1.0]
    assert list(twist.high_speed_range) == [0.75, 1.0]
    assert twist.high_speed_fraction == pytest.approx(0.4)
    action = env_cfg.actions["joint_pos"]
    assert action.clip_target_to_joint_limits is True
    assert action.clip_target_margin == pytest.approx(0.0)
    assert env_cfg.curriculum["head_pitch_target_band_weight"] is None
    assert env_cfg.curriculum["action_clip_margin"] is None
    band = env_cfg.rewards["head_pitch_target_band"]
    assert band.weight == pytest.approx(-12.0)
    assert band.params["band_deg"] == [[-15.0, 45.0], [-50.0, 50.0]]


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


def test_timed_resample_keeps_the_spawn_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UniformVelocityCommand,
        "_resample_command",
        lambda self, env_ids: None,
    )
    term = object.__new__(MicroduckVelocityCommand)
    term._hold_spawn_heading = True
    term._turn_fraction = 0.0
    term._high_speed_range = None
    term._capture_spawn_heading = np.zeros(3, dtype=np.bool_)
    # Row 0 was just reset; rows 1 and 2 hit their resampling timer mid-episode.
    term.command_counter = np.asarray([0, 1, 4], dtype=np.int64)

    term._resample_command(np.asarray([0, 1, 2], dtype=np.intp))

    np.testing.assert_array_equal(term._capture_spawn_heading, [True, False, False])


def test_high_speed_fraction_redraws_forward_speed_from_the_top_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        UniformVelocityCommand,
        "_resample_command",
        lambda self, env_ids: None,
    )
    num_envs = 2000
    term = object.__new__(MicroduckVelocityCommand)
    term._hold_spawn_heading = False
    term._turn_fraction = 0.0
    term._high_speed_fraction = 0.4
    term._high_speed_range = (0.9, 1.2)
    term._env = SimpleNamespace(rng=np.random.default_rng(0))
    term.vel_command_b = np.zeros((num_envs, 3), dtype=np.float32)
    term.vel_command_b[:, 0] = 0.5
    term.vel_command_w = term.vel_command_b.copy()
    term.is_standing_env = np.zeros(num_envs, dtype=np.bool_)
    term.is_standing_env[:10] = True

    term._resample_command(np.arange(num_envs, dtype=np.intp))

    speeds = term.vel_command_b[:, 0]
    fast = speeds != 0.5
    assert not fast[:10].any()
    assert 0.35 < fast.mean() < 0.45
    assert speeds[fast].min() >= 0.9 and speeds[fast].max() <= 1.2
    np.testing.assert_array_equal(term.vel_command_w[:, 0], speeds)


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
            root_link_pos_w=np.zeros((2, 3), dtype=np.float32),
            heading_w=np.zeros(2, dtype=np.float32),
        )
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            episode_length_buf=np.asarray([10, 10], dtype=np.int64),
            scene={"robot": entity},
            command_manager=SimpleNamespace(
                get_command=lambda name: np.asarray(
                    [[1.3, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float32
                )
            ),
        ),
    )
    straightness = running_straightness_cost(
        env,
        command_name="twist",
        lateral_scale=0.20,
        yaw_rate_scale=0.20,
        asset_cfg=SceneEntityCfg("robot"),
    )
    term = cross_track_abs_cost(
        RewardTermCfg(
            func=cross_track_abs_cost,
            weight=-1.0,
            params={
                "distance_cap": 0.40,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        env,
    )
    entity.data.root_link_pos_w[0, :2] = np.asarray([1.0, 0.15], dtype=np.float32)
    entity.data.root_link_pos_w[1, :2] = np.asarray([1.0, 0.0], dtype=np.float32)
    cross_track = term(env)
    assert straightness[0] > straightness[1]
    assert cross_track[0] > cross_track[1]


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


def test_action_target_overshoot_cost_reads_the_target_not_the_joint() -> None:
    term = SimpleNamespace(
        processed_action=np.asarray(
            [[0.2, -0.1], [-3.0, 0.4], [0.0, 2.5]], dtype=np.float32
        ),
        joint_ids=np.asarray([2, 0], dtype=np.intp),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            soft_joint_pos_limits=np.asarray(
                [[-0.5, 0.5], [-9.0, 9.0], [-1.0, 1.0]], dtype=np.float32
            )
        )
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: term),
            scene={"robot": robot},
        ),
    )
    cost = action_target_overshoot_cost(env, action_name="joint_pos")
    # Column 0 drives joint 2 (limit +-1.0), column 1 drives joint 0 (+-0.5).
    np.testing.assert_allclose(cost, [0.0, 2.0, 2.0], atol=1e-6)

    # With a clamping action, the cost reads the request before the clamp.
    term.requested_target = term.processed_action
    term.processed_action = np.zeros_like(term.requested_target)
    np.testing.assert_allclose(
        action_target_overshoot_cost(env, action_name="joint_pos"), [0.0, 2.0, 2.0], atol=1e-6
    )


def test_action_clip_margin_curriculum_steps_the_margin_down() -> None:
    margins: list[float] = []
    term = SimpleNamespace(set_target_clip_margin=margins.append)
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: term),
        common_step_counter=0,
    )
    stages = [
        {"step": 0, "margin": 0.5},
        {"step": 100, "margin": 0.2},
        {"step": 200, "margin": 0.0},
    ]
    curriculum = action_clip_margin_curriculum(
        CurriculumTermCfg(func=action_clip_margin_curriculum, params={"stages": stages}),
        cast(ManagerBasedRlEnv, env),
    )
    for step in (0, 150, 250):
        env.common_step_counter = step
        curriculum(cast(ManagerBasedRlEnv, env), slice(None), stages=stages)
    assert margins == [0.5, 0.2, 0.0]


def test_action_target_band_cost_reads_the_request_behind_the_clamp() -> None:
    term = SimpleNamespace(
        requested_target=np.radians(
            np.asarray([[-120.0, 0.0], [-40.0, 60.0]], dtype=np.float64)
        ).astype(np.float32),
        joint_ids=np.asarray([1, 0], dtype=np.intp),
    )
    robot = SimpleNamespace(joint_names=["head_pitch", "neck_pitch"])
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: term),
            scene={"robot": robot},
        ),
    )
    cost = action_target_band_cost(
        RewardTermCfg(
            func=action_target_band_cost,
            weight=-1.0,
            params={
                "action_name": "joint_pos",
                "joint_names": ["neck_pitch", "head_pitch"],
                "band_deg": [[-50.0, 45.0], [-50.0, 50.0]],
            },
        ),
        env,
    )
    # Column 0 drives neck_pitch: -120 is 70 deg below its band; head_pitch at
    # 60 is 10 deg above its band. Row means of the overshoot, in radians.
    np.testing.assert_allclose(
        cost(env), np.radians([35.0, 5.0]), atol=1e-5
    )


def test_head_upright_hold_frees_the_nod_and_punishes_inversion() -> None:
    root_half = math.sqrt(0.5)
    quats = np.asarray(
        [
            [root_half, 0.0, -root_half, 0.0],
            [0.92388, 0.0, -0.38268, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [root_half, 0.0, root_half, 0.0],
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

    excess = head_tilt_excess_cost(
        RewardTermCfg(
            func=head_tilt_excess_cost,
            weight=-1.0,
            params={
                "free_tilt_deg": 60.0,
                "zero_tilt_deg": 95.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=("jaw_soft",)),
            },
        ),
        env,
    )
    np.testing.assert_allclose(excess(env), [0.0, 0.0, 0.851487, 1.0], atol=1e-4)


def test_running_stride_length_pays_for_scissoring_not_a_parked_split() -> None:
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

    env, term = build()
    robot = env.scene["robot"]

    def write_parked(step: int) -> None:
        del step
        robot.data.joint_pos[0, 0] = -0.5 * split
        robot.data.joint_pos[0, 1] = -0.5 * split

    parked = []
    for step in range(100):
        write_parked(step)
        parked.append(float(term(env)[0]))

    env, term = build()
    robot = env.scene["robot"]
    alternating = []
    for step in range(100):
        forward = 0.5 * split if step % 6 < 3 else -0.5 * split
        robot.data.joint_pos[0, 0] = -forward
        robot.data.joint_pos[0, 1] = -forward
        alternating.append(float(term(env)[0]))

    assert float(np.mean(parked[50:])) < 0.05
    assert float(np.mean(alternating[50:])) > 0.7
