"""Stilt morphology generator and the stilt owner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.cli import package_root

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def _script():
    spec = importlib.util.spec_from_file_location("make_stilt_robot", ROOT / "scripts" / "make_stilt_robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stilt_profile_morphs_from_platform_to_peg() -> None:
    script = _script()
    assert script.profile(0.0)["tip_width_mm"] == pytest.approx(22.0)
    assert script.profile(1.0)["tip_length_mm"] == pytest.approx(12.0)
    assert script.profile(0.5)["tip_width_mm"] == pytest.approx(17.0)
    assert script.stilt_tag(7.5, 0.5) == "h7p5_b050"
    with pytest.raises(ValueError):
        script.stilt_mesh(0.5, 0.0)


def test_stilt_mesh_spans_the_requested_height() -> None:
    vertices, faces = _script().stilt_mesh(10.0, 0.5)
    v = np.asarray(vertices).reshape(-1, 3)
    assert v[:, 2].min() == pytest.approx(-0.10) and v[:, 2].max() == pytest.approx(0.0)
    assert len(faces) % 3 == 0 and max(faces) < len(v)


def test_generated_stage_moves_contact_to_the_stilt_tip() -> None:
    paths = _script().write_stage(5.0, 0.5)
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    stilt = model.geom("left_foot_collision")
    sole = model.geom("left_original_sole_disabled")
    assert (stilt.contype[0], stilt.conaffinity[0]) == (1, 1)
    assert (sole.contype[0], sole.conaffinity[0]) == (0, 0)
    tip = data.site("left_foot").xpos
    original = data.site("left_original_foot_site").xpos
    assert original[2] - tip[2] == pytest.approx(0.05, abs=1e-6)
    keyframe = paths["task"].read_text()
    assert "0 0 0.1700" in keyframe


def test_stilt_owner_builds_on_the_walking_task() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_stilt_flat/mujoco"])
    assert cfg.training.task_name == "MicroduckVelocityFlat"
    assert cfg.env.events.push_robot is None
    assert cfg.reward.upright.weight == pytest.approx(3.0)
    assert cfg.env.commands.twist.rel_standing_envs == pytest.approx(0.25)
    assert [s["weight"] for s in cfg.env.curriculum.action_rate_weight.params.stages] == [-0.02, -0.05, -0.10, -0.20]
