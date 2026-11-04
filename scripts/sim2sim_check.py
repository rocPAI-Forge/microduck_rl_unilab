"""MotrixSim sim2sim gate for MuJoCo-trained sprint checkpoints.

Training uses MuJoCo only. MotrixSim is held out: a checkpoint is checked on it
after training and is never trained on it, so the check stays independent.
A checkpoint passes when, on the Motrix owner with the target clamp on,

* 300 s straight runs at each ``--gate-speeds`` command survive at least
  ``--min-survival``, and
* 60 s turns at ``--turn-speed`` and each ``--turn-yaw-rates`` command survive
  at least ``--min-survival`` with yaw-rate error at most ``--max-yaw-error``.

``--report-speeds`` are scored and written out but not gated. The default
gate band is the 0-0.6 m/s band recommended for first hardware runs; 1.0 m/s
is reported only, since the cross-check found it does not transfer.

Usage:
    uv run --no-sync scripts/sim2sim_check.py examples/sprint_stand/model_59000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = "microduck_sprint_straightspeed_flat/motrix"


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def _run(
    checkpoint: Path,
    out_dir: Path,
    *,
    speed: float,
    yaw_rate: float,
    duration: float,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    yaw_tag = f"_y{yaw_rate:g}" if yaw_rate else ""
    output = out_dir / f"motrix_{checkpoint.stem}_{int(duration)}s_seed{seed}_v{speed:g}{yaw_tag}.json"
    command = [
        "uv", "run", "--no-sync", "scripts/eval_sprint_speed.py", str(checkpoint),
        "--owner", OWNER, "--clip-targets",
        f"--yaw-rate={yaw_rate}", "--speed", str(speed),
        "--duration", str(duration), "--num-envs", str(num_envs), "--seed", str(seed),
        "--output", str(output),
    ]
    print("+", " ".join(command), flush=True)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"eval failed for speed {speed}, yaw {yaw_rate}")
    battery = json.loads(output.read_text(encoding="utf-8"))
    measured_yaw = float(battery["yaw_rate_rad_s"])
    return {
        "command_m_s": speed,
        "command_yaw_rate_rad_s": yaw_rate,
        "duration_s": duration,
        "survival_fraction": float(battery["survival_fraction"]),
        "straight_pose_survival_fraction": battery.get("straight_pose_survival_fraction"),
        "yaw_rate_rad_s": measured_yaw,
        "yaw_rate_error_rad_s": abs(measured_yaw - yaw_rate),
        "file": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gate-speeds", type=_floats, default=[0.0, 0.3, 0.6])
    parser.add_argument("--report-speeds", type=_floats, default=[1.0])
    parser.add_argument("--straight-duration", type=float, default=300.0)
    parser.add_argument("--turn-speed", type=float, default=0.5)
    parser.add_argument("--turn-yaw-rates", type=_floats, default=[-0.4, 0.4])
    parser.add_argument("--turn-duration", type=float, default=60.0)
    parser.add_argument("--min-survival", type=float, default=0.95)
    parser.add_argument("--max-yaw-error", type=float, default=0.10)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    out_dir = checkpoint.parent / "sim2sim"
    out_dir.mkdir(exist_ok=True)
    common = {"num_envs": args.num_envs, "seed": args.seed}

    gated: list[dict[str, Any]] = []
    for speed in args.gate_speeds:
        row = _run(checkpoint, out_dir, speed=speed, yaw_rate=0.0,
                   duration=args.straight_duration, **common)
        row["passes"] = row["survival_fraction"] >= args.min_survival
        gated.append(row)
    for yaw_rate in args.turn_yaw_rates:
        row = _run(checkpoint, out_dir, speed=args.turn_speed, yaw_rate=yaw_rate,
                   duration=args.turn_duration, **common)
        row["passes"] = (
            row["survival_fraction"] >= args.min_survival
            and row["yaw_rate_error_rad_s"] <= args.max_yaw_error
        )
        gated.append(row)
    reported = [
        _run(checkpoint, out_dir, speed=speed, yaw_rate=0.0,
             duration=args.straight_duration, **common)
        for speed in args.report_speeds
    ]

    result = {
        "checkpoint": str(checkpoint),
        "simulator": "motrix",
        "owner": OWNER,
        "min_survival": args.min_survival,
        "max_yaw_error_rad_s": args.max_yaw_error,
        "passes": all(row["passes"] for row in gated),
        "gated": gated,
        "reported": reported,
    }
    output = args.output or out_dir / f"sim2sim_{checkpoint.stem}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in gated + reported:
        mark = {True: "pass", False: "FAIL", None: "report"}[row.get("passes")]
        print(
            f"v={row['command_m_s']:.2f} yaw={row['command_yaw_rate_rad_s']:+.2f} "
            f"{row['duration_s']:.0f}s survival={row['survival_fraction']:.3f} "
            f"yaw_err={row['yaw_rate_error_rad_s']:.3f} [{mark}]"
        )
    print(f"sim2sim {'PASS' if result['passes'] else 'FAIL'}: {output}")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
