#!/usr/bin/env python3
"""Write a ground-contact MicroDuck variant whose collision meshes are small convex hulls.

UniLab's MuJoCo backend clones the physics model once per environment as soon
as any model field (friction, armature, mass, CoM) is randomized per
environment. For the ground-contact robot that clone is ~30 MB, almost all of
it the vertices, faces, and triangle BVH of nine 5k-10k vertex collision
meshes. MuJoCo collides mesh geoms through their convex hull only, so each
collision mesh is replaced by its hull capped at ``--maxhullvert`` vertices
(MuJoCo's own hull simplification); the clone then drops to about 1 MB, which
lets ground-contact tasks run thousands of environments on CPU.

The capped hull sits inside the exact hull; ``check`` reports how far the exact
hull pokes out of it, which is the only geometric change. Bodies carry explicit
``<inertial>`` elements, so mass and inertia do not depend on the meshes, and
visual geoms keep the original meshes. The check also confirms bodies,
joints, actuators, sensors, and contact parameters are unchanged and that the
robot settles into the same standing pose.

Writes into ``assets/robots/microduck/generated/`` (git-ignored):
    microduck_groundcontact_bam_hull.xml
    scene_flat_groundcontact_bam_hull.xml

Usage:
    uv run --no-sync scripts/make_hull_robot.py [--maxhullvert 128]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "microduck"
OUT_DIR = ROBOT_DIR / "generated"
ROBOT = "microduck_groundcontact_bam.xml"
SCENE = "scene_flat_groundcontact_bam.xml"
TASK = ROBOT_DIR / "locomotion_task.xml"


def _graph(model: mujoco.MjModel, mesh_id: int) -> tuple[int, int, int]:
    adr = int(model.mesh_graphadr[mesh_id])
    if adr < 0:
        raise ValueError(f"mesh {model.mesh(mesh_id).name!r} has no convex hull")
    return adr, int(model.mesh_graph[adr]), int(model.mesh_graph[adr + 1])


def hull_vertex_ids(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    """Indices of a compiled mesh's hull vertices within that mesh's vertex list."""
    adr, numvert, _ = _graph(model, mesh_id)
    return model.mesh_graph[adr + 2 + numvert : adr + 2 + 2 * numvert]


def hull_vertices(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    """Hull vertices of a compiled mesh, in the compiled mesh frame."""
    return model.mesh_vert[model.mesh_vertadr[mesh_id] + hull_vertex_ids(model, mesh_id)].astype(np.float64)


def hull_faces(model: mujoco.MjModel, mesh_id: int) -> np.ndarray:
    """Hull triangles as (numface, 3, 3) vertex positions in the compiled mesh frame."""
    adr, numvert, numface = _graph(model, mesh_id)
    start = adr + 2 + 3 * numvert + 3 * numface
    ids = model.mesh_graph[start : start + 3 * numface].reshape(numface, 3)
    return model.mesh_vert[model.mesh_vertadr[mesh_id] + ids].astype(np.float64)


def is_collision(model: mujoco.MjModel, geom_id: int) -> bool:
    return bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])


def collision_mesh_geoms(model: mujoco.MjModel) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and is_collision(model, g)]


def build_robot(maxhullvert: int) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / ROBOT))
    model = spec.compile()
    geoms = collision_mesh_geoms(model)
    if any(not model.geom(g).name for g in geoms):
        raise ValueError("collision mesh geoms must be named so they can be matched")
    mesh_ids = sorted({int(model.geom_dataid[g]) for g in geoms})

    capped_spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / ROBOT))
    for mesh in capped_spec.meshes:
        mesh.maxhullvert = maxhullvert
    capped = capped_spec.compile()

    hull_names = {}
    for mesh_id in mesh_ids:
        hull = spec.add_mesh()
        hull.name = f"{model.mesh(mesh_id).name}_hull"
        # The mesh frame follows the hull, so take the capped hull's vertex indices
        # and read them in the uncapped frame the geoms are placed in.
        ids = model.mesh_vertadr[mesh_id] + hull_vertex_ids(capped, mesh_id)
        hull.uservert = model.mesh_vert[ids].astype(np.float64).ravel().tolist()
        hull_names[mesh_id] = hull.name
    for gid in geoms:
        geom = spec.geom(model.geom(gid).name)
        geom.meshname = hull_names[int(model.geom_dataid[gid])]
        # Hull vertices are expressed in the compiled frame of the original mesh;
        # pinning the geom to that frame keeps the world geometry in place.
        geom.pos = model.geom_pos[gid].copy()
        geom.quat = model.geom_quat[gid].copy()
    used = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in list(spec.meshes):
        if (mesh.name or Path(mesh.file).stem) not in used:
            spec.delete(mesh)
    return spec


def load_with_task(scene: Path) -> mujoco.MjModel:
    """Compile a scene plus the task fragment (sensors and the home keyframe)."""
    # Includes resolve against the top-level file, so the wrapper sits beside the scene.
    wrapper = scene.with_name(f".{scene.stem}_with_task.xml")
    wrapper.write_text(f'<mujoco><include file="{scene.name}"/><include file="{TASK}"/></mujoco>')
    try:
        return mujoco.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink()


SELF_PAIRS = (
    ("trunk_base_self_collision", "left_leg_self_collision"),
    ("trunk_base_self_collision", "right_leg_self_collision"),
    ("left_leg_self_collision", "right_leg_self_collision"),
)


def _random_pose(model: mujoco.MjModel, rng: np.random.Generator, free_base: bool) -> np.ndarray:
    qpos = model.key("home").qpos.copy()
    if free_base:
        qpos[2] = rng.uniform(0.02, 0.14)
        axis = rng.normal(size=3)
        angle = rng.uniform(0, np.pi)
        qpos[3:7] = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis / np.linalg.norm(axis)]
    lo, hi = model.jnt_range[1:].T
    qpos[7:] = rng.uniform(lo, hi)
    return qpos


def floor_distance_gap_mm(original: mujoco.MjModel, hull: mujoco.MjModel, poses: int = 2000, seed: int = 0,
                          cutoff: float = 0.05) -> dict[str, tuple[float, float]]:
    """Per collision mesh: (min, max) of capped minus exact floor distance over random poses, in mm."""
    data_o, data_h = mujoco.MjData(original), mujoco.MjData(hull)
    rng = np.random.default_rng(seed)
    names = [original.geom(g).name for g in collision_mesh_geoms(original)]
    gaps: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(poses):
        qpos = _random_pose(original, rng, free_base=True)
        for model, data in ((original, data_o), (hull, data_h)):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
        for name in names:
            exact = mujoco.mj_geomDistance(original, data_o, original.geom("floor").id, original.geom(name).id, cutoff, None)
            capped = mujoco.mj_geomDistance(hull, data_h, hull.geom("floor").id, hull.geom(name).id, cutoff, None)
            if min(exact, capped) < cutoff:
                gaps[name].append(1e3 * (capped - exact))
    return {k: (min(v), max(v)) for k, v in gaps.items() if v}


def self_contact_agreement(original: mujoco.MjModel, hull: mujoco.MjModel, poses: int = 2000,
                           seed: int = 1) -> dict[str, tuple[int, int, float]]:
    """Per self-collision pair: (poses in contact with exact meshes, poses that disagree, max depth gap in mm).

    Uses the contacts the collision pipeline generates, which is what the
    self-collision sensors read; ``mj_geomDistance`` is unreliable for
    mesh-mesh pairs.
    """
    data_o, data_h = mujoco.MjData(original), mujoco.MjData(hull)
    rng = np.random.default_rng(seed)
    stats = {f"{a} / {b}": [0, 0, 0.0] for a, b in SELF_PAIRS}

    def depth(model: mujoco.MjModel, data: mujoco.MjData, a: str, b: str) -> float | None:
        pair = {model.geom(a).id, model.geom(b).id}
        dists = [data.contact[i].dist for i in range(data.ncon) if {data.contact[i].geom1, data.contact[i].geom2} == pair]
        return min(dists) if dists else None

    for _ in range(poses):
        qpos = _random_pose(original, rng, free_base=False)
        for model, data in ((original, data_o), (hull, data_h)):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
        for a, b in SELF_PAIRS:
            exact, capped = depth(original, data_o, a, b), depth(hull, data_h, a, b)
            entry = stats[f"{a} / {b}"]
            entry[0] += exact is not None
            entry[1] += (exact is None) != (capped is None)
            if exact is not None and capped is not None:
                entry[2] = max(entry[2], 1e3 * abs(capped - exact))
    return {k: (v[0], v[1], v[2]) for k, v in stats.items()}


def check(original: mujoco.MjModel, hull: mujoco.MjModel) -> tuple[dict, dict]:
    for field in ("nbody", "njnt", "nu", "nsensor", "nkey"):
        assert getattr(hull, field) == getattr(original, field), field
    # Tolerances absorb the float printing of the XML round trip.
    np.testing.assert_allclose(hull.body_inertia, original.body_inertia, rtol=1e-5, err_msg="body_inertia")
    for field in ("body_mass", "body_ipos", "body_iquat", "jnt_range", "actuator_ctrlrange",
                  "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity"):
        np.testing.assert_allclose(getattr(hull, field), getattr(original, field), rtol=1e-5, atol=1e-6, err_msg=field)
    return floor_distance_gap_mm(original, hull), self_contact_agreement(original, hull)


def write_variant(maxhullvert: int = 128) -> Path:
    """Write the hull robot and its scene; return the scene path."""
    OUT_DIR.mkdir(exist_ok=True)
    spec = build_robot(maxhullvert)
    spec.compile()
    # Visual mesh files resolve against the original robot directory.
    robot = OUT_DIR / ROBOT.replace(".xml", "_hull.xml")
    robot.write_text(re.sub(r'meshdir="[^"]*"', 'meshdir="../assets"', spec.to_xml()))
    scene_text = (ROBOT_DIR / SCENE).read_text()
    include = f'<include file="{ROBOT}"/>'
    if scene_text.count(include) != 1:
        raise ValueError(f"{SCENE} must include {ROBOT} once")
    scene = OUT_DIR / SCENE.replace(".xml", "_hull.xml")
    scene.write_text(scene_text.replace(include, f'<include file="{robot.name}"/>'))
    return scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maxhullvert", type=int, default=128)
    args = parser.parse_args()

    scene = write_variant(args.maxhullvert)
    original, hulled = load_with_task(ROBOT_DIR / SCENE), load_with_task(scene)
    floor, self_pairs = check(original, hulled)
    print("capped minus exact floor distance over random poses (mm):")
    for name, (low, high) in floor.items():
        print(f"  {name:30s} {low:+.3f} .. {high:+.3f}")
    print("self-collision contacts over random joint poses:")
    for pair, (touching, disagree, gap) in self_pairs.items():
        print(f"  {pair:52s} touching {touching:4d}, disagree {disagree:3d}, max depth gap {gap:.2f} mm")
    print(f"{scene.relative_to(REPO_ROOT)}: collision meshes capped at {args.maxhullvert} hull vertices, "
          f"model buffer {original.nbuffer / 1e6:.1f} MB -> {hulled.nbuffer / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
