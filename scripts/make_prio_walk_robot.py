#!/usr/bin/env python3
"""Write the walking robot with its collision geoms at contact priority 1.

The floor in every MicroDuck scene has MuJoCo's default friction 1.0 at
priority 0, the robot's geoms too, and MuJoCo combines equal-priority geoms by
the larger friction. So foot friction randomized below 1.0 has never taken
effect in the walking tasks (``docs/audits/friction_combination.md``, roadmap
item 12). At priority 1 the robot's own friction governs floor contacts; the
floor and robot share solref/solimp/condim, so nothing else changes. This is
the walking-model counterpart of ``make_hull_robot.py``'s ``_prio`` variant.

Writes into ``assets/robots/microduck/generated/`` (git-ignored):
    microduck_bam_prio.xml, scene_flat_bam_prio.xml

Usage:
    uv run --no-sync scripts/make_prio_walk_robot.py
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "microduck"
OUT_DIR = ROBOT_DIR / "generated"
ROBOT = "microduck_bam.xml"
SCENE = "scene_flat_bam.xml"


def write_variant() -> Path:
    """Write the priority robot and its scene; return the scene path."""
    OUT_DIR.mkdir(exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / ROBOT))
    model = spec.compile()
    for gid in range(model.ngeom):
        if model.geom_contype[gid] or model.geom_conaffinity[gid]:
            name = model.geom(gid).name
            if not name:
                raise ValueError("collision geoms must be named to raise their priority")
            spec.geom(name).priority = 1
    spec.compile()
    robot = OUT_DIR / ROBOT.replace(".xml", "_prio.xml")
    # Mesh files resolve against the original robot directory.
    robot.write_text(re.sub(r'meshdir="[^"]*"', 'meshdir="../assets"', spec.to_xml()))
    scene_text = (ROBOT_DIR / SCENE).read_text()
    include = f'<include file="{ROBOT}"/>'
    if scene_text.count(include) != 1:
        raise ValueError(f"{SCENE} must include {ROBOT} once")
    scene = OUT_DIR / SCENE.replace(".xml", "_prio.xml")
    scene.write_text(scene_text.replace(include, f'<include file="{robot.name}"/>'))
    return scene


def main() -> int:
    scene = write_variant()
    print(f"{scene.relative_to(REPO_ROOT)}: robot collision geoms at contact priority 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
