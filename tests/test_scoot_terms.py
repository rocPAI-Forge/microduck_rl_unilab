"""Seated scooting ("butt walk") terms and owner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.cli import package_root
from unilab.managers._types import ManagerBasedRlEnv

from microduck_rl_unilab.tasks.microduck import scoot_terms as sc

CONF_DIR = package_root() / "conf" / "ppo"


class _Scene(dict):
    def __init__(self, robot: SimpleNamespace, n: int):
        super().__init__(robot=robot)
        self.env_origins = np.zeros((n, 3))


def _env(*, z: list[float], vx: list[float], wz: list[float], cmd: list[list[float]]) -> ManagerBasedRlEnv:
    n = len(z)
    data = SimpleNamespace(
        root_link_pos_w=np.stack([np.zeros(n), np.zeros(n), np.asarray(z)], axis=1),
        root_link_lin_vel_b=np.stack([np.asarray(vx), np.zeros(n), np.zeros(n)], axis=1),
        root_link_lin_vel_w=np.stack([np.asarray(vx), np.zeros(n), np.zeros(n)], axis=1),
        root_link_ang_vel_b=np.stack([np.zeros(n), np.zeros(n), np.asarray(wz)], axis=1),
    )
    command = np.asarray(cmd, dtype=np.float32)
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=n,
            scene=_Scene(SimpleNamespace(data=data), n),
            command_manager=SimpleNamespace(get_command=lambda name: command),
        ),
    )


def test_progress_is_capped_signed_and_off_below_threshold() -> None:
    env = _env(
        z=[0.065] * 4,
        vx=[-0.1, -0.02, 0.03, 0.05],
        wz=[0.0] * 4,
        cmd=[[-0.05, 0, 0], [-0.05, 0, 0], [-0.05, 0, 0], [0.005, 0, 0]],
    )
    np.testing.assert_allclose(sc.command_progress(env, axis="lin_x", min_cmd=0.01), [1.0, 0.4, -0.6, 0.0], atol=1e-6)
    with pytest.raises(ValueError):
        sc.command_progress(env, axis="lin_y")


def test_stillness_only_at_the_zero_command() -> None:
    env = _env(z=[0.065, 0.065], vx=[0.0, 0.0], wz=[0.0, 0.0], cmd=[[0, 0, 0], [-0.05, 0, 0]])
    np.testing.assert_allclose(sc.stillness_at_zero_command(env, vel_std=0.03), [1.0, 0.0])


def test_seated_band_cost_and_stood_up_termination() -> None:
    env = _env(z=[0.07, 0.09, 0.10], vx=[0.0] * 3, wz=[0.0] * 3, cmd=[[0, 0, 0]] * 3)
    np.testing.assert_allclose(sc.height_above_l1_penalty(env, 0.08), [0.0, -0.01, -0.02], atol=1e-6)
    np.testing.assert_array_equal(sc.root_height_above(env, 0.095), [False, False, True])


def test_scoot_owner_starts_seated_and_tracks_a_backward_biased_twist() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_scoot_flat/mujoco"])
    twist = cfg.env.commands.twist
    assert list(twist.ranges.lin_vel_x) == [-0.10, 0.06]
    assert twist.rel_standing_envs == pytest.approx(0.2)
    ground = cfg.env.events.set_ground_state.params
    assert (ground.sitting_prob, ground.standing_prob) == (1.0, 0.0)
    assert cfg.env.terminations.stood_up.params.max_height == pytest.approx(0.095)
    assert cfg.reward.progress_lin.weight == pytest.approx(2.0)
    assert cfg.reward.hip_limit_proximity.weight < 0
    for name in ("seated_height_cap", "head_pose_l1", "descent_speed", "gentle_motion"):
        assert cfg.reward[name].weight > 0, name
    assert "np_f970_collision" in cfg.env.events.seat_friction.params.asset_cfg.geom_names
    assert max(s["step"] for s in cfg.env.curriculum.head_pose_range.params.stages) < 1400 * 24
