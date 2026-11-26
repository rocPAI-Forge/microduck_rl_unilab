#!/usr/bin/env python3
"""Write the basketball scene: the hull ground-contact duck plus a free ball.

After Vottivott/microduck-playground ``microduck_basketball_env_cfg.py``
(Apache-2.0): a size-7 basketball (radius 0.12 m, 0.62 kg, rubber friction
1.2 / 0.01 / 0.001, condim 4 so a spinning ball under the feet is not free)
on a free joint. The robot is the ``_prio`` convex-hull variant from
``make_hull_robot.py`` (4096 envs fit; robot friction governs its contacts);
the ball also has priority 1, so robot-ball contacts take the larger friction.

The task fragment is ``locomotion_task.xml`` with the ``home`` keyframe
extended by the ball's free joint (the ball body follows the robot in the
worldbody) and the duck standing on the apex.

Writes into ``assets/robots/microduck/generated/`` (git-ignored):
    scene_basketball.xml, locomotion_task_basketball.xml

Usage:
    uv run --no-sync scripts/make_basketball_scene.py
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "microduck"
OUT_DIR = ROBOT_DIR / "generated"

BALL_RADIUS = 0.12
BALL_MASS = 0.62
BALL_FRICTION = "1.2 0.01 0.001"
# Root above the sole plane when standing (velocity spawn height).
ROOT_HEIGHT = 0.125
SPAWN_CLEARANCE = 0.003

BALL_BODY = f"""
    <body name="ball" pos="0 0 {BALL_RADIUS}">
      <freejoint name="ball_free"/>
      <geom name="ball_sphere" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}"
        friction="{BALL_FRICTION}" condim="4" priority="1" rgba="0.91 0.42 0.12 1"/>
    </body>
"""


def _hull_script():
    spec = importlib.util.spec_from_file_location("make_hull_robot", REPO_ROOT / "scripts" / "make_hull_robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_scene() -> dict[str, Path]:
    hull_scene = _hull_script().write_variant(128, friction_priority=True)
    scene_text = hull_scene.read_text()
    if scene_text.count("</worldbody>") != 1:
        raise ValueError(f"{hull_scene.name} must have one worldbody")
    scene = OUT_DIR / "scene_basketball.xml"
    scene.write_text(scene_text.replace("</worldbody>", BALL_BODY + "  </worldbody>"))

    task = (ROBOT_DIR / "locomotion_task.xml").read_text()
    match = re.search(r'(<key name="home" qpos=")([^"]*)(")', task)
    if match is None:
        raise ValueError("home keyframe not found")
    values = match.group(2).split()
    values[2] = f"{2 * BALL_RADIUS + ROOT_HEIGHT + SPAWN_CLEARANCE:.4f}"
    qpos = " ".join(values) + f" 0 0 {BALL_RADIUS + 0.001:.4f} 1 0 0 0"
    task_path = OUT_DIR / "locomotion_task_basketball.xml"
    task_path.write_text(task[: match.start(2)] + qpos + task[match.end(2):])
    return {"scene": scene, "task": task_path}


def main() -> int:
    paths = write_scene()
    wrapper = paths["scene"].with_name(".scene_basketball_with_task.xml")
    wrapper.write_text(f'<mujoco><include file="{paths["scene"].name}"/><include file="{paths["task"]}"/></mujoco>')
    try:
        model = mujoco.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink()
    ball = model.body("ball").id
    print(f"{paths['scene'].relative_to(REPO_ROOT)}: nq {model.nq}, ball mass {model.body_mass[ball]:.2f} kg, "
          f"model buffer {model.nbuffer / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
