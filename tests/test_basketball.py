"""Basketball balance scene, terms, and owner."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

from microduck_rl_unilab.conf_searchpath import register_conf_search_path
from microduck_rl_unilab.tasks.microduck import basketball_terms as bb

ROOT = Path(__file__).resolve().parents[1]


def _make_scene() -> None:
    spec = importlib.util.spec_from_file_location("make_basketball_scene", ROOT / "scripts" / "make_basketball_scene.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.write_scene()


def test_hold_curriculum_promotes_long_episodes_and_demotes_quick_falls() -> None:
    levels = (1.0, 0.5, 0.0)
    env = SimpleNamespace(num_envs=3, step_dt=0.02, episode_length_buf=np.array([400, 100, 10]))
    bb._state(env).level[:] = [0, 1, 0]
    mean = bb.basketball_hold_curriculum(env, np.arange(3), levels=levels, promote_len_s=6.0, demote_len_s=1.5)
    assert bb._state(env).level.tolist() == [1, 1, 0]
    assert bb._state(env).hold.tolist() == [0.5, 0.5, 1.0]
    assert mean == np.mean([1, 1, 0])


def _env(hold: float, num_envs: int = 8):
    _make_scene()
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_basketball_flat/mujoco"])
    with open_dict(cfg):
        cfg.env.actions.ball_hold.levels = [hold]
        cfg.env.curriculum.basketball_hold = None
        cfg.env.events.push_robot = None
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="ppo").build_task_env_cfg_override()
    env = registry.make("MicroduckBasketballFlat", sim_backend="mujoco", num_envs=num_envs, env_cfg_override=override)
    env.reset()
    return env


def test_reset_stands_the_duck_on_the_apex_and_hold_pins_the_ball() -> None:
    env = _env(hold=1.0)
    robot, ball = env.scene["robot"], env.scene["ball"]
    obs = env.obs_buf
    assert np.asarray(obs["obs"]).shape[1] == 61
    assert not np.any(np.asarray(obs["obs"])[:, 51:61])
    rel_z = np.asarray(obs["critic"])[:, 57] / 3.0
    np.testing.assert_allclose(rel_z, -(0.12 + 0.125), atol=0.01)
    spawn = np.asarray(ball.data.root_link_pos_w).copy()
    for _ in range(100):
        state = env.step(np.zeros((env.num_envs, 14), dtype=np.float32))
        assert not np.any(np.asarray(state.terminated))
    assert np.linalg.norm(np.asarray(ball.data.root_link_pos_w)[:, :2] - spawn[:, :2], axis=1).max() < 0.01
    feet = bb.basketball_feet_on_ball(env, env.reward_manager.get_term_cfg("feet_on_ball").params["asset_cfg"])
    assert np.all(feet > 0.3)
    assert np.all(np.asarray(robot.data.root_link_pos_w)[:, 2] > 0.3)


def test_a_free_ball_drops_the_passive_duck() -> None:
    env = _env(hold=0.0)
    fell = np.zeros(env.num_envs, dtype=bool)
    for _ in range(100):
        fell |= np.asarray(env.step(np.zeros((env.num_envs, 14), dtype=np.float32)).terminated)
    assert fell.all()
