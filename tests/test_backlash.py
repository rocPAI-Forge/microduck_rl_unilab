"""Backlash robot: generated files, keyframe layout, and the unchanged contract."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.cli import package_root

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "assets" / "robots" / "microduck"
CONF_DIR = package_root() / "conf" / "ppo"


def _script():
    spec = importlib.util.spec_from_file_location("add_backlash", ROOT / "scripts" / "add_backlash.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checked_in_backlash_files_match_the_script() -> None:
    script = _script()
    robot, added = script.add_backlash(
        (ROBOT_DIR / "microduck_bam.xml").read_text(),
        total_deg=2.0,
        damping=0.01,
        armature=0.001,
        frictionloss=0.0,
    )
    assert len(added) == 14
    assert (ROBOT_DIR / "microduck_bam_backlash.xml").read_text() == robot
    task = script.interleave_keyframe((ROBOT_DIR / "locomotion_task.xml").read_text(), 14)
    assert (ROBOT_DIR / "locomotion_task_backlash.xml").read_text() == task


def test_backlash_keyframe_puts_a_zero_after_every_servo_joint() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROBOT_DIR / "scene_flat_bam_backlash.xml"))
    text = (ROBOT_DIR / "locomotion_task_backlash.xml").read_text()
    qpos = np.asarray(re.search(r'qpos="([^"]*)"', text).group(1).split(), dtype=float)
    plain = np.asarray(
        re.search(r'qpos="([^"]*)"', (ROBOT_DIR / "locomotion_task.xml").read_text()).group(1).split(),
        dtype=float,
    )
    assert model.nq == qpos.size == 7 + 28
    assert model.nu == 14
    for joint in range(1, model.njnt):
        name = model.joint(joint).name
        value = qpos[model.jnt_qposadr[joint]]
        if name.startswith("passive_"):
            assert value == 0.0
            np.testing.assert_allclose(np.degrees(model.jnt_range[joint]), [-1.0, 1.0])
    servo = [qpos[model.jnt_qposadr[j]] for j in range(1, model.njnt) if not model.joint(j).name.startswith("passive_")]
    np.testing.assert_allclose(servo, plain[7:])


def test_backlash_owner_keeps_the_14_servo_joints_as_the_entity() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_sprint_straightspeed_flat/mujoco_backlash"])
    scene = cfg.env.scene
    assert scene.model_file.endswith("scene_flat_bam_backlash.xml")
    assert list(scene.fragment_files) == ["assets/robots/microduck/locomotion_task_backlash.xml"]
    joints = list(scene.entities.robot.joint_names)
    assert len(joints) == 14 and not any(name.startswith("passive_") for name in joints)
