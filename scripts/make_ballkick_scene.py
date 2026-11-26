#!/usr/bin/env python3
"""Write the ball-kick scene: the hull ground-contact duck plus a small free ball.

After pollen-robotics/microduck_rl ``microduck_ball_kick_env_cfg.py`` and
``robot/microduck/ball.xml`` (develop, Apache-2.0): a 70 mm, 15 g hollow ball
(inertia 2/3 m r^2, friction 0.5 / 0.005 / 0.0001 so a kicked ball rolls
away). Unlike ``scene_flat_bam_kick.xml`` (walking model, soles only), the legs
and the top of the feet collide here, so the duck can strike the ball. The
robot is the plain convex-hull variant from ``make_hull_robot.py`` (same
contact friction as the standup training).

The task fragment is ``locomotion_task.xml`` with the ``home`` keyframe
extended by the ball's free joint.

Writes into ``assets/robots/microduck/generated/`` (git-ignored):
    scene_ballkick.xml, locomotion_task_ballkick.xml

Usage:
    uv run --no-sync scripts/make_ballkick_scene.py
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "microduck"
OUT_DIR = ROBOT_DIR / "generated"

BALL_RADIUS = 0.035
BALL_MASS = 0.015
BALL_INERTIA = 2.0 / 3.0 * BALL_MASS * BALL_RADIUS**2

BALL_BODY = f"""
    <body name="ball" pos="0.3 0 {BALL_RADIUS}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{BALL_MASS}" diaginertia="{BALL_INERTIA:.4e} {BALL_INERTIA:.4e} {BALL_INERTIA:.4e}"/>
      <geom name="ball_sphere" type="sphere" size="{BALL_RADIUS}" friction="0.5 0.005 0.0001" rgba="1 0.55 0 1"/>
    </body>
"""


def _hull_script():
    spec = importlib.util.spec_from_file_location("make_hull_robot", REPO_ROOT / "scripts" / "make_hull_robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_scene() -> dict[str, Path]:
    hull_scene = _hull_script().write_variant(128)
    scene_text = hull_scene.read_text()
    if scene_text.count("</worldbody>") != 1:
        raise ValueError(f"{hull_scene.name} must have one worldbody")
    scene = OUT_DIR / "scene_ballkick.xml"
    scene.write_text(scene_text.replace("</worldbody>", BALL_BODY + "  </worldbody>"))

    task = (ROBOT_DIR / "locomotion_task.xml").read_text()
    match = re.search(r'(<key name="home" qpos=")([^"]*)(")', task)
    if match is None:
        raise ValueError("home keyframe not found")
    qpos = " ".join(match.group(2).split()) + f" 0.3 0 {BALL_RADIUS + 0.001:.4f} 1 0 0 0"
    task_path = OUT_DIR / "locomotion_task_ballkick.xml"
    task_path.write_text(task[: match.start(2)] + qpos + task[match.end(2):])
    return {"scene": scene, "task": task_path}


def main() -> int:
    paths = write_scene()
    wrapper = paths["scene"].with_name(".scene_ballkick_with_task.xml")
    wrapper.write_text(f'<mujoco><include file="{paths["scene"].name}"/><include file="{paths["task"]}"/></mujoco>')
    try:
        model = mujoco.MjModel.from_xml_path(str(wrapper))
    finally:
        wrapper.unlink()
    ball = model.body("ball").id
    print(f"{paths['scene'].relative_to(REPO_ROOT)}: nq {model.nq}, ball mass {model.body_mass[ball]:.3f} kg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
