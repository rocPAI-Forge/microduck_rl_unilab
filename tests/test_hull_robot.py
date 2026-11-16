"""Convex-hull collision variant of the ground-contact robot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
