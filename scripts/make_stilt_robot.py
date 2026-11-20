#!/usr/bin/env python3
"""Write a MicroDuck BAM robot on stilts, its scene, and its keyframe fragment.

Port of ``stilt_constants.py`` from Vottivott/microduck-playground (Apache-2.0).
Each stilt is a convex loft from a rounded-rectangle root under the sole to a
tip at ``height_cm`` below it; ``blend`` morphs the profile from a 22 x 32 mm
platform (0) to a 12 mm round peg (1). The stilt body takes over the
``<side>_foot_collision`` geom and ``<side>_foot`` site names, so the contact
sensors and foot terms read the stilt tip; the original sole stays visible
with its collision off. The ``home`` keyframe is raised by the stilt length.

Writes into ``assets/robots/microduck/generated/`` (git-ignored):
``microduck_bam_stilt_<tag>.xml``, ``scene_flat_bam_stilt_<tag>.xml``, and
``locomotion_task_stilt_<tag>.xml``, with ``<tag> = h<cm>_b<blend*100>``.

Usage:
    uv run --no-sync scripts/make_stilt_robot.py --height-cm 2 --blend 0
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mujoco

ROBOT_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "microduck"
OUT_DIR = ROBOT_DIR / "generated"
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 0.8, 300.0
MASS_BASE_KG, MASS_PER_CM_KG = 0.012, 0.001


def stilt_tag(height_cm: float, blend: float) -> str:
    return f"h{height_cm:g}_b{round(blend * 100):03d}".replace(".", "p")


def profile(blend: float) -> dict[str, float]:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")

    def lerp(a: float, b: float) -> float:
        return a + blend * (b - a)

    return {
        "tip_width_mm": lerp(22.0, 12.0),
        "tip_length_mm": lerp(32.0, 12.0),
        "tip_radius_mm": lerp(3.5, 6.0),
        "root_width_mm": lerp(23.0, 22.0),
        "root_length_mm": lerp(41.0, 22.0),
        "root_radius_mm": lerp(4.5, 11.0),
    }


def _ring(width: float, length: float, radius: float, samples: int = 8) -> list[tuple[float, float]]:
    radius = min(radius, width / 2.0, length / 2.0)
    corners = (
        (width / 2 - radius, length / 2 - radius, 0.0),
        (-width / 2 + radius, length / 2 - radius, math.pi / 2),
        (-width / 2 + radius, -length / 2 + radius, math.pi),
        (width / 2 - radius, -length / 2 + radius, 3 * math.pi / 2),
    )
    return [
        (cx + radius * math.cos(a0 + (math.pi / 2) * k / samples), cy + radius * math.sin(a0 + (math.pi / 2) * k / samples))
        for cx, cy, a0 in corners
        for k in range(samples)
    ]


def stilt_mesh(height_cm: float, blend: float) -> tuple[list[float], list[int]]:
    """Watertight convex loft in the foot-site frame: top at z=0, tip at z=-height."""
    if not MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM:
        raise ValueError(f"height_cm must be in [{MIN_HEIGHT_CM}, {MAX_HEIGHT_CM}]")
    d = profile(blend)
    mm = 0.001
    bottom = _ring(d["tip_width_mm"] * mm, d["tip_length_mm"] * mm, d["tip_radius_mm"] * mm)
    top = _ring(d["root_width_mm"] * mm, d["root_length_mm"] * mm, d["root_radius_mm"] * mm)
    n = len(bottom)
    h = height_cm * 0.01
    vertices = [(x, y, -h) for x, y in bottom] + [(x, y, 0.0) for x, y in top] + [(0.0, 0.0, -h), (0.0, 0.0, 0.0)]
    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i), (bc, j, i), (tc, n + i, n + j)]
    return [c for v in vertices for c in v], [k for f in faces for k in f]


def build_robot(height_cm: float, blend: float, mass_kg: float | None = None) -> mujoco.MjSpec:
    mass_kg = MASS_BASE_KG + MASS_PER_CM_KG * height_cm if mass_kg is None else mass_kg
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "microduck_bam.xml"))
    vertices, faces = stilt_mesh(height_cm, blend)
    spec.add_mesh(name="stilt_mesh", uservert=vertices, userface=faces)
    for side in ("left", "right"):
        sole = spec.geom(f"{side}_foot_collision")
        contype, conaffinity = sole.contype, sole.conaffinity
        sole.name = f"{side}_original_sole_disabled"
        sole.contype, sole.conaffinity = 0, 0
        site = spec.site(f"{side}_foot")
        site.name = f"{side}_original_foot_site"
        body = spec.body(f"ankle_{side}").add_body(name=f"stilt_{side}", pos=site.pos.copy(), quat=site.quat.copy())
        geom = body.add_geom(
            name=f"{side}_foot_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="stilt_mesh",
            mass=mass_kg,
        )
        geom.contype, geom.conaffinity = contype, conaffinity
        geom.friction = sole.friction.copy()
        geom.condim = sole.condim
        geom.rgba = (0.45, 0.34, 0.85, 1.0)
        body.add_site(name=f"{side}_foot", pos=(0.0, 0.0, -height_cm * 0.01))
    return spec


def write_stage(height_cm: float, blend: float, mass_kg: float | None = None) -> dict[str, Path]:
    OUT_DIR.mkdir(exist_ok=True)
    tag = stilt_tag(height_cm, blend)
    spec = build_robot(height_cm, blend, mass_kg)
    spec.compile()
    robot_xml = spec.to_xml()
    # Mesh files resolve against the original robot directory.
    robot_xml = re.sub(r'meshdir="[^"]*"', 'meshdir="../assets"', robot_xml)
    robot = OUT_DIR / f"microduck_bam_stilt_{tag}.xml"
    robot.write_text(robot_xml)

    scene = (ROBOT_DIR / "scene_flat_bam.xml").read_text()
    if scene.count('<include file="microduck_bam.xml"/>') != 1:
        raise ValueError("scene_flat_bam.xml must include microduck_bam.xml once")
    scene_path = OUT_DIR / f"scene_flat_bam_stilt_{tag}.xml"
    scene_path.write_text(scene.replace('<include file="microduck_bam.xml"/>', f'<include file="{robot.name}"/>'))

    task = (ROBOT_DIR / "locomotion_task.xml").read_text()
    match = re.search(r'(<key name="home" qpos="\s*)(\S+ \S+ )(\S+)', task)
    if match is None:
        raise ValueError("home keyframe not found")
    z = float(match.group(3)) + height_cm * 0.01
    task = task[: match.start(3)] + f"{z:.4f}" + task[match.end(3):]
    task_path = OUT_DIR / f"locomotion_task_stilt_{tag}.xml"
    task_path.write_text(task)
    return {"robot": robot, "scene": scene_path, "task": task_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height-cm", type=float, required=True)
    parser.add_argument("--blend", type=float, required=True)
    parser.add_argument("--mass-kg", type=float, default=None)
    args = parser.parse_args()
    for kind, path in write_stage(args.height_cm, args.blend, args.mass_kg).items():
        print(f"{kind}: {path.relative_to(ROBOT_DIR.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
