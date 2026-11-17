"""Convex-hull collision variant of the ground-contact robot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("make_hull_robot", ROOT / "scripts" / "make_hull_robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hull_robot_keeps_physics_and_contacts() -> None:
    script = _script()
    scene = script.write_variant(128)
    original = script.load_with_task(script.ROBOT_DIR / script.SCENE)
    hull = script.load_with_task(scene)
    floor, self_pairs = script.check(original, hull)

    assert set(floor) == {original.geom(g).name for g in script.collision_mesh_geoms(original)}
    for name, (low, high) in floor.items():
        # The capped hull lies inside the exact one: never closer, at most 2 mm farther.
        assert low > -0.01, name
        assert high < 2.0, name
    for pair, (_, disagree, gap) in self_pairs.items():
        assert disagree == 0 and gap < 2.0, pair
    for gid in script.collision_mesh_geoms(hull):
        assert hull.mesh_vertnum[hull.geom_dataid[gid]] <= 128


def test_priority_variant_lets_robot_friction_govern_floor_contacts() -> None:
    script = _script()
    plain = script.load_with_task(script.write_variant(128))
    prio = script.load_with_task(script.write_variant(128, friction_priority=True))
    floor = prio.geom("floor").id
    robot = [g for g in range(prio.ngeom) if g != floor and script.is_collision(prio, g)]
    assert robot and all(prio.geom_priority[g] > prio.geom_priority[floor] for g in robot)
    for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "body_mass"):
        assert (getattr(prio, field) == getattr(plain, field)).all(), field
    # A low robot friction survives contact combination with the 1.0 floor.
    for model, expected in ((plain, 1.0), (prio, 0.1)):
        model.geom_friction[robot, 0] = 0.1
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[2] = 0.02
        mujoco.mj_forward(model, data)
        frictions = [data.contact[i].friction[0] for i in range(data.ncon) if floor in (data.contact[i].geom1, data.contact[i].geom2)]
        assert frictions and np.allclose(frictions, expected)
