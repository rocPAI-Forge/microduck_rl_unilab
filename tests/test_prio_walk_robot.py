"""The priority walking robot lets foot friction below 1.0 govern floor contacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "microduck"


def _script():
    spec = importlib.util.spec_from_file_location("make_prio_walk_robot", REPO_ROOT / "scripts" / "make_prio_walk_robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load(scene: Path) -> mujoco.MjModel:
    task = (ROBOT_DIR / "locomotion_task.xml").resolve()
    wrapper = scene.with_name(f".{scene.stem}_with_task.xml")
    wrapper.write_text(f'<mujoco><include file="{scene.name}"/><include file="{task}"/></mujoco>')
    try:
        return mujoco.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink()


def test_low_foot_friction_takes_effect_only_with_priority() -> None:
    prio = _load(_script().write_variant())
    plain = _load(ROBOT_DIR / "scene_flat_bam.xml")
    floor = prio.geom("floor").id
    feet = [prio.geom(name).id for name in ("left_foot_collision", "right_foot_collision")]
    assert all(prio.geom_priority[g] > prio.geom_priority[floor] for g in feet)
    for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "body_mass"):
        assert (getattr(prio, field) == getattr(plain, field)).all(), field
    for model, expected in ((plain, 1.0), (prio, 0.1)):
        model.geom_friction[feet, 0] = 0.1
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[2] -= 0.005
        mujoco.mj_forward(model, data)
        frictions = [
            data.contact[i].friction[0]
            for i in range(data.ncon)
            if floor in (data.contact[i].geom1, data.contact[i].geom2)
        ]
        assert frictions and np.allclose(frictions, expected)
