#!/usr/bin/env python3
"""Write backlash variants of the MicroDuck BAM robot, scene, and keyframe.

Port of upstream ``add_backlash.py`` from pollen-robotics/microduck_rl
(Apache-2.0, ``src/mjlab_microduck/robot/microduck/add_backlash.py``). After
every actuated servo joint (``class="chosen_actuator"``) it inserts an
unactuated hinge on the same body and axis:

    <joint axis="0 0 1" name="left_hip_yaw" ... class="chosen_actuator"/>
    <joint axis="0 0 1" name="passive_left_hip_yaw_backlash" type="hinge" class="backlash"/>

The link rotation is servo output plus play, and the play is free within
+-backlash/2. BAM drives the servo joint and the encoder reads it, so the
policy observation is unchanged; the task entity lists the 14 servo joints by
name, so the passive joints stay out of every observation, action, and reward.

Writes, next to the inputs:

* ``microduck_bam_backlash.xml``: robot with the passive joints,
* ``scene_flat_bam_backlash.xml``: ``scene_flat_bam.xml`` including it,
* ``locomotion_task_backlash.xml``: the task fragment with a ``home`` keyframe
  that has a zero after every servo joint.

Usage:
    uv run --no-sync scripts/add_backlash.py --backlash-deg 2.0
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

ROBOT_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "microduck"
JOINT_RE = re.compile(r"^(\s*)<joint\b[^>]*/>\s*$")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
QPOS_RE = re.compile(r'(<key name="home" qpos=")([^"]*)(")', re.S)


def _backlash_default(half_range: float, damping: float, armature: float, frictionloss: float, total_deg: float) -> str:
    # With a range this small the default solref (0.02, 1) lets the joint
    # overshoot its limits about 2x under load; 0.01 is 2 x sim_dt (0.005 s),
    # the stiffest stable setting, and solimp makes the tooth contact nearly rigid.
    return (
        f"  <!-- Backlash from scripts/add_backlash.py: {total_deg:g} deg total play"
        f" (symmetric +/-{total_deg / 2:g} deg) -->\n"
        f"  <default>\n"
        f'    <default class="backlash">\n'
        f'      <joint damping="{damping:g}" frictionloss="{frictionloss:g}"'
        f' armature="{armature:g}" limited="true"'
        f' range="{-half_range:.17g} {half_range:.17g}"'
        f' solreflimit="0.01 1" solimplimit="0.95 0.999 0.0001 0.5 2"/>\n'
        f"    </default>\n"
        f"  </default>\n"
    )


def add_backlash(robot_xml: str, *, total_deg: float, damping: float, armature: float, frictionloss: float) -> tuple[str, list[str]]:
    if 'class="backlash"' in robot_xml:
        raise ValueError("robot already contains backlash joints")
    half_range = math.radians(total_deg) / 2.0
    out: list[str] = []
    added: list[str] = []
    default_inserted = False
    for line in robot_xml.splitlines(keepends=True):
        if not default_inserted and "<worldbody>" in line:
            out.append(_backlash_default(half_range, damping, armature, frictionloss, total_deg))
            default_inserted = True
        out.append(line)
        match = JOINT_RE.match(line)
        if match is None:
            continue
        attrs = dict(ATTR_RE.findall(line))
        if attrs.get("class") != "chosen_actuator" or "name" not in attrs:
            continue
        pos = f' pos="{attrs["pos"]}"' if "pos" in attrs else ""
        out.append(
            f'{match.group(1)}<joint axis="{attrs.get("axis", "0 0 1")}"{pos} '
            f'name="passive_{attrs["name"]}_backlash" type="hinge" class="backlash"/>\n'
        )
        added.append(attrs["name"])
    if not default_inserted or not added:
        raise ValueError("no <worldbody> or no chosen_actuator joints found")
    return "".join(out), added


def interleave_keyframe(task_xml: str, num_joints: int) -> str:
    match = QPOS_RE.search(task_xml)
    if match is None:
        raise ValueError('no <key name="home" qpos="..."> found')
    values = match.group(2).split()
    if len(values) != 7 + num_joints:
        raise ValueError(f"home qpos has {len(values)} values, expected {7 + num_joints}")
    joints = values[7:]
    rows = [" ".join(values[:7])] + [f"{value} 0" for value in joints]
    qpos = "\n        " + "\n        ".join(rows)
    return task_xml[: match.start(2)] + qpos + task_xml[match.end(2):]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backlash-deg", type=float, default=2.0, help="total peak-to-peak play, degrees")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--armature", type=float, default=0.001)
    parser.add_argument("--frictionloss", type=float, default=0.0)
    args = parser.parse_args()

    robot, added = add_backlash(
        (ROBOT_DIR / "microduck_bam.xml").read_text(),
        total_deg=args.backlash_deg,
        damping=args.damping,
        armature=args.armature,
        frictionloss=args.frictionloss,
    )
    (ROBOT_DIR / "microduck_bam_backlash.xml").write_text(robot)

    scene = (ROBOT_DIR / "scene_flat_bam.xml").read_text()
    if scene.count('<include file="microduck_bam.xml"/>') != 1:
        raise ValueError("scene_flat_bam.xml must include microduck_bam.xml once")
    (ROBOT_DIR / "scene_flat_bam_backlash.xml").write_text(
        scene.replace('<include file="microduck_bam.xml"/>', '<include file="microduck_bam_backlash.xml"/>')
    )

    task = (ROBOT_DIR / "locomotion_task.xml").read_text()
    (ROBOT_DIR / "locomotion_task_backlash.xml").write_text(interleave_keyframe(task, len(added)))
    print(f"added {len(added)} backlash joints (+/-{args.backlash_deg / 2:g} deg): {', '.join(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
