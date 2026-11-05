"""Run the hardware bring-up checks against a robot adapter.

The adapter is any object with ``read() -> SensorReading``, ``heading()``,
``write_goals(goals)``, and ``set_torque(enabled)`` (see
``microduck_rl_unilab.deploy.bringup.RobotIO``), built by a factory given as
``module:function``. The checks run in the order of ``docs/HARDWARE_BRINGUP.md``
and stop at the first failure; the operator answers prompts on the terminal.

Usage:
    uv run --no-sync scripts/hardware_bringup.py examples/sprint_stand/deploy \
        --io my_robot.io:make_io
    uv run --no-sync scripts/hardware_bringup.py examples/sprint_stand/deploy \
        --io my_robot.io:make_io --checks imu_static,imu_axes
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from microduck_rl_unilab.deploy import Controller
from microduck_rl_unilab.deploy.bringup import (
    CheckResult,
    check_default_pose,
    check_imu_axes,
    check_imu_static,
    check_joint_directions,
    check_loop_timing,
    check_policy_stand,
)

CHECKS = (
    "imu_static",
    "imu_axes",
    "default_pose",
    "joint_directions",
    "loop_timing",
    "policy_stand",
)
SETUP = {
    "imu_static": "Torque off. Stand the robot upright on a level surface and keep it still.",
    "imu_axes": "Torque off. Hold the robot in your hands; follow each prompt.",
    "default_pose": "Torque off. Hold or pose the robot in the default (home) pose.",
    "joint_directions": "Hang the robot so the feet are off the ground. Torque turns on.",
    "loop_timing": "Torque may stay off; the robot is not moved.",
    "policy_stand": "Feet on the ground, harness on, hand on the torque switch. Torque turns on.",
}


def _prompt(message: str) -> None:
    input(f"  {message}, then press Enter... ")


def _confirm(message: str) -> bool:
    return input(f"  {message} [y/n] ").strip().lower().startswith("y")


def _load_io(spec: str) -> object:
    module_name, _, factory = spec.partition(":")
    if not factory:
        raise SystemExit("--io must be module:function")
    return getattr(importlib.import_module(module_name), factory)()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("deploy_dir", type=Path)
    parser.add_argument("--io", required=True, help="adapter factory, module:function")
    parser.add_argument("--checks", default=",".join(CHECKS))
    parser.add_argument("--stand-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    names = [name for name in args.checks.split(",") if name]
    unknown = sorted(set(names) - set(CHECKS))
    if unknown:
        raise SystemExit(f"unknown checks: {unknown}")
    io = _load_io(args.io)
    controller = Controller(args.deploy_dir)
    manifest = controller.manifest
    runs = {
        "imu_static": lambda: check_imu_static(io),
        "imu_axes": lambda: check_imu_axes(io, _prompt),
        "default_pose": lambda: check_default_pose(io, manifest),
        "joint_directions": lambda: check_joint_directions(io, manifest, confirm=_confirm),
        "loop_timing": lambda: check_loop_timing(io, controller),
        "policy_stand": lambda: check_policy_stand(
            io, controller, duration_s=args.stand_seconds
        ),
    }

    results: list[CheckResult] = []
    try:
        for name in names:
            print(f"\n[{name}] {SETUP[name]}")
            input("  Ready? Press Enter to start... ")
            result = runs[name]()
            results.append(result)
            print(f"  {'PASS' if result.passed else 'FAIL'}: {result.detail}")
            if not result.passed:
                print("  Stopping: fix this before the next check.")
                break
    finally:
        io.set_torque(False)

    passed = len(results) == len(names) and all(r.passed for r in results)
    output = args.output or Path(f"bringup_{time.strftime('%Y%m%d_%H%M%S')}.json")
    output.write_text(
        json.dumps({"passes": passed, "results": [asdict(r) for r in results]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nbring-up {'PASS' if passed else 'FAIL'}: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
