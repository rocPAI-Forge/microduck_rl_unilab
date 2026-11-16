"""Commanded sit <-> stand: posture terms and the MuJoCo owner."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.cli import package_root
from unilab.managers._types import ManagerBasedRlEnv

from microduck_rl_unilab.tasks.microduck import sitstand_terms as st

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


class _Scene(dict):
    def __init__(self, robot: SimpleNamespace, n: int):
        super().__init__(robot=robot)
        self.env_origins = np.zeros((n, 3))


def _env(
    *,
    z: list[float],
    vz: list[float],
    flag: list[float],
    alpha: list[float],
    tilt_rad: float = 0.0,
) -> ManagerBasedRlEnv:
    n = len(z)
    half = tilt_rad / 2.0
    quat = np.tile([math.cos(half), math.sin(half), 0.0, 0.0], (n, 1))
    data = SimpleNamespace(
        root_link_pos_w=np.stack([np.zeros(n), np.zeros(n), np.asarray(z)], axis=1),
        root_link_lin_vel_w=np.stack([np.zeros(n), np.zeros(n), np.asarray(vz)], axis=1),
        root_link_quat_w=quat,
        projected_gravity_b=np.tile([0.0, math.sin(tilt_rad), -math.cos(tilt_rad)], (n, 1)),
    )
    command = np.stack([np.asarray(flag), np.zeros(n), np.zeros(n)], axis=1)
    term = SimpleNamespace(alpha=np.asarray(alpha))
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=n,
            scene=_Scene(SimpleNamespace(data=data), n),
            command_manager=SimpleNamespace(get_command=lambda name: command, get_term=lambda name: term),
        ),
    )


def test_height_terms_track_the_slewed_target_not_the_flag() -> None:
    # Mid-ramp (alpha 0.5): the target is halfway, so a robot already seated is penalized.
    env = _env(z=[0.0875, 0.060], vz=[0.0, 0.0], flag=[1.0, 1.0], alpha=[0.5, 0.5])
    gauss = st.posture_height_gaussian(env, sit_z=0.060, stand_z=0.115, std=0.015)
    l1 = st.posture_height_l1(env, sit_z=0.060, stand_z=0.115)
    assert gauss[0] == pytest.approx(1.0)
    assert gauss[1] < 0.05
    np.testing.assert_allclose(l1, [0.0, -0.0275], atol=1e-6)


def test_speed_caps_and_rise_bootstrap_signs() -> None:
    env = _env(z=[0.08, 0.08, 0.13], vz=[-0.15, 0.2, 0.2], flag=[0.0, 0.0, 0.0], alpha=[0.0, 0.0, 0.0])
    np.testing.assert_allclose(st.trunk_downward_velocity_penalty(env, 0.05), [-0.10, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(st.trunk_upward_velocity_penalty(env, 0.08), [0.0, -0.12, -0.12], atol=1e-6)
    # Capped at max_vz, only below max_height, only under a STAND command.
    np.testing.assert_allclose(st.posture_rise_bootstrap(env, 0.125, max_vz=0.08), [0.0, 0.08, 0.0], atol=1e-6)
    sit = _env(z=[0.08], vz=[0.2], flag=[1.0], alpha=[1.0])
    assert st.posture_rise_bootstrap(sit, 0.125, max_vz=0.08)[0] == 0.0


def test_stillness_needs_the_ramp_done_and_an_upright_rest() -> None:
    kwargs = dict(sit_z=0.060, stand_z=0.115)
    done = _env(z=[0.060], vz=[0.0], flag=[1.0], alpha=[1.0])
    ramping = _env(z=[0.060], vz=[0.0], flag=[1.0], alpha=[0.9])
    flopped = _env(z=[0.060], vz=[0.0], flag=[1.0], alpha=[1.0], tilt_rad=math.radians(80))
    assert st.posture_stillness(done, **kwargs)[0] == pytest.approx(1.0)
    assert st.posture_stillness(ramping, **kwargs)[0] == 0.0
    assert st.posture_stillness(flopped, **kwargs)[0] == 0.0


def test_upright_while_tall_fades_out_toward_the_seat() -> None:
    env = _env(z=[0.11, 0.07], vz=[0.0, 0.0], flag=[0.0, 0.0], alpha=[0.0, 0.0])
    np.testing.assert_allclose(st.upright_while_tall(env, height_low=0.075, height_high=0.10), [1.0, 0.0])


def test_sitstand_mujoco_owner_composes_the_posture_layer() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sitstand_flat/mujoco"])
    registry.ensure_registries()
    env_cfg = registry.materialize_env_config("MicroduckSitStandFlat")
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override(),
    )
    env_cfg.validate()

    assert cfg.training.sim_backend == "mujoco"
    twist = env_cfg.commands["twist"]
    assert type(twist).__name__ == "SitStandCommandCfg"
    assert (twist.sit_prob, twist.ramp_s) == (0.5, 2.0)
    rewards = env_cfg.rewards
    assert rewards["posture_composite"].func is st.posture_composite
    assert rewards["posture_pose_legs"].params["sit_overrides"]["left_knee"] == pytest.approx(1.35)
    for name in ("descent_speed", "posture_height_l1", "posture_pose_l1", "gentle_motion"):
        assert rewards[name].weight >= 0.0, name
    for name in ("pose_stand_legs", "standing_composite", "body_pose_tracking"):
        assert name not in rewards or rewards[name] is None
    ground = env_cfg.events["set_ground_state"].params
    assert (ground["sitting_prob"], ground["standing_prob"], ground["face_down_prob"]) == (0.5, 0.5, 0.0)
