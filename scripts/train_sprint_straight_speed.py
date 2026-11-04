"""Raise along-ray speed of the published straight sprint without leaving the ray.

Resumes ``examples/sprint_straight_long/model_19297.pt`` on
``microduck_sprint_straightspeed_flat``. Body-frame forward progress stays at
the straightfix value. The new reward is capped spawn-ray speed, ramped
together with the heading penalty.

Checkpoints are scored after training, on the same closed-loop owner the
policy was trained with. A checkpoint counts only when a 60 s battery keeps
the head upright, the heading inside the published band, and the sticky
straight-pose gate up. The fastest such checkpoint then gets the 300 s battery.

``--eval-speeds`` scores every checkpoint at several commands and gates on
the worst one; with ``--max-speed-error`` the speed gate becomes
``|measured - command|`` instead of a minimum speed. ``--stress-speeds`` adds
the push/CoM/tilt stress battery at those commands.

Usage:
    HIP_VISIBLE_DEVICES=0 uv run --no-sync scripts/train_sprint_straight_speed.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOAD_RUN = REPO_ROOT / "examples" / "sprint_straight_long"
DEFAULT_CHECKPOINT = "19297"
DEFAULT_RESULTS = REPO_ROOT / "results" / "sprint_straight_speed.jsonl"
# Scoring always uses the push-free closed-loop owner so every round is judged
# on the same battery, whichever owner trained it.
OWNER = "microduck_sprint_straightspeed_flat/mujoco"
STRESS_OWNER = "microduck_sprint_straighthold_flat/mujoco"
# Extra eval_sprint_speed.py flags shared by every battery of one run.
EVAL_FLAGS: list[str] = []
DEFAULT_TASK = "microduck_sprint_straightspeed_flat"


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )


def _newest_run(log_root: Path, *, after: float) -> Path | None:
    if not log_root.is_dir():
        return None
    candidates = [
        path
        for path in log_root.glob("*_mujoco")
        if path.is_dir() and path.stat().st_mtime >= after - 5.0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _train_env() -> dict[str, str]:
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _run_json(command: list[str], *, output: Path) -> dict[str, Any] | None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_train_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not output.is_file():
        print(completed.stdout[-2000:], flush=True)
        print(completed.stderr[-2000:], flush=True)
        return None
    return json.loads(output.read_text(encoding="utf-8"))


def _battery(
    checkpoint: Path,
    *,
    speed: float,
    num_envs: int,
    duration: float,
    seed: int,
    stress: bool = False,
    yaw_rate: float = 0.0,
) -> dict[str, Any] | None:
    tag = "stress_" if stress else ""
    yaw_tag = f"_y{yaw_rate:g}" if yaw_rate else ""
    battery_out = checkpoint.with_name(
        f"battery_{tag}{checkpoint.stem}_{int(duration)}s_seed{seed}_v{speed:g}{yaw_tag}.json"
    )
    battery = _run_json(
        [
            "uv", "run", "--no-sync", "scripts/eval_sprint_speed.py", str(checkpoint),
            "--owner", STRESS_OWNER if stress else OWNER,
            *(["--stress"] if stress else []),
            *EVAL_FLAGS,
            "--yaw-rate", str(yaw_rate),
            "--speed", str(speed),
            "--num-envs", str(num_envs),
            "--warmup", "1",
            "--duration", str(duration),
            "--seed", str(seed),
            "--device", "cuda:0",
            "--output", str(battery_out),
        ],
        output=battery_out,
    )
    if battery is None:
        return None
    speed_block = battery.get("body_forward_speed_m_s") or {}
    horizon = battery.get("straight_pose_horizon_s") or {}
    measured = speed_block.get("mean")
    measured_yaw = battery.get("yaw_rate_rad_s")
    return {
        "command_yaw_rate_rad_s": yaw_rate,
        "yaw_rate_rad_s": measured_yaw,
        "yaw_rate_error_rad_s": (
            None if measured_yaw is None else abs(float(measured_yaw) - yaw_rate)
        ),
        "command_m_s": speed,
        "speed_m_s": measured,
        "speed_error_m_s": None if measured is None else abs(float(measured) - speed),
        "spawn_forward_m": (battery.get("final_spawn_forward_displacement_m") or {}).get("mean"),
        "survival_fraction": battery.get("survival_fraction"),
        "heading_error_deg": battery.get("mean_absolute_heading_error_deg"),
        "straight_pose_survival_fraction": battery.get("straight_pose_survival_fraction"),
        "straight_pose_horizon_s": horizon.get("mean"),
        "head_inverted_time_fraction": battery.get("head_inverted_time_fraction"),
        "head_up_tilt_deg": battery.get("head_up_tilt_deg"),
        "neck_pitch_deg": battery.get("neck_pitch_deg"),
        "head_forward_in_trunk_m": battery.get("head_forward_in_trunk_m"),
        "action_target_overshoot_rad": battery.get("action_target_overshoot_rad"),
    }


def _score_checkpoint(
    checkpoint: Path,
    *,
    speeds: list[float],
    num_envs: int,
    duration: float,
    seed: int,
    stress_speeds: list[float] | None = None,
    stress_duration: float = 60.0,
    turn_cases: list[tuple[float, float]] | None = None,
    turn_duration: float = 30.0,
) -> dict[str, Any] | None:
    """Run one battery per command speed and keep the worst case of each gate."""
    per_speed = []
    for speed in speeds:
        battery = _battery(
            checkpoint, speed=speed, num_envs=num_envs, duration=duration, seed=seed
        )
        if battery is None or battery["speed_m_s"] is None:
            return None
        per_speed.append(battery)
    stress_rows = []
    for speed in stress_speeds or []:
        battery = _battery(
            checkpoint,
            speed=speed,
            num_envs=num_envs,
            duration=stress_duration,
            seed=seed,
            stress=True,
        )
        if battery is None or battery["survival_fraction"] is None:
            return None
        stress_rows.append(battery)
    turn_rows = []
    for speed, yaw_rate in turn_cases or []:
        battery = _battery(
            checkpoint,
            speed=speed,
            num_envs=num_envs,
            duration=turn_duration,
            seed=seed,
            yaw_rate=yaw_rate,
        )
        if battery is None or battery["yaw_rate_error_rad_s"] is None:
            return None
        turn_rows.append(battery)

    def worst(key: str, pick: Any) -> Any:
        values = [row[key] for row in per_speed]
        return None if any(value is None for value in values) else pick(values)

    record = {
        "checkpoint": str(checkpoint),
        "iteration": int(checkpoint.stem.split("_")[1]),
        "duration_s": duration,
        "seed": seed,
        "speed_m_s": sum(row["speed_m_s"] for row in per_speed) / len(per_speed),
        "speed_error_m_s": worst("speed_error_m_s", max),
        "survival_fraction": worst("survival_fraction", min),
        "heading_error_deg": worst("heading_error_deg", max),
        "straight_pose_survival_fraction": worst("straight_pose_survival_fraction", min),
        "head_inverted_time_fraction": worst("head_inverted_time_fraction", max),
        "head_forward_in_trunk_m": worst("head_forward_in_trunk_m", max),
        "neck_pitch_deg": worst("neck_pitch_deg", min),
        "action_target_overshoot_rad": worst("action_target_overshoot_rad", max),
    }
    if len(per_speed) == 1:
        record.update({key: value for key, value in per_speed[0].items() if key not in record})
    else:
        record["per_speed"] = per_speed
    if stress_rows:
        record["stress_per_speed"] = stress_rows
        record["stress_survival_fraction"] = min(
            float(row["survival_fraction"]) for row in stress_rows
        )
    if turn_rows:
        record["turn_per_case"] = turn_rows
        record["yaw_rate_error_rad_s"] = max(
            float(row["yaw_rate_error_rad_s"]) for row in turn_rows
        )
        record["turn_survival_fraction"] = min(
            float(row["survival_fraction"]) for row in turn_rows
        )
        turn_head = [row["head_forward_in_trunk_m"] for row in turn_rows]
        if record.get("head_forward_in_trunk_m") is not None and None not in turn_head:
            record["head_forward_in_trunk_m"] = max(
                record["head_forward_in_trunk_m"], *turn_head
            )
    return record


def _passes(record: dict[str, Any], args: argparse.Namespace) -> bool:
    keys = (
        "speed_m_s",
        "speed_error_m_s",
        "survival_fraction",
        "heading_error_deg",
        "straight_pose_survival_fraction",
        "head_inverted_time_fraction",
    )
    if any(record.get(key) is None for key in keys):
        return False
    speed, error, survival, heading, straight, inverted = (float(record[key]) for key in keys)
    if args.max_speed_error is None:
        speed_ok = speed >= args.min_speed
    else:
        speed_ok = error <= args.max_speed_error
    stress = record.get("stress_survival_fraction")
    stress_ok = stress is None or float(stress) >= args.min_stress_survival
    head_forward = record.get("head_forward_in_trunk_m")
    head_ok = args.max_head_forward is None or (
        head_forward is not None and float(head_forward) <= args.max_head_forward
    )
    overshoot = record.get("action_target_overshoot_rad")
    range_ok = args.max_target_overshoot is None or (
        overshoot is not None and float(overshoot) <= args.max_target_overshoot
    )
    yaw_error = record.get("yaw_rate_error_rad_s")
    turn_survival = record.get("turn_survival_fraction")
    turn_ok = yaw_error is None or (
        float(yaw_error) <= args.max_yaw_error
        and float(turn_survival) >= args.min_turn_survival
    )
    return (
        speed_ok
        and stress_ok
        and head_ok
        and range_ok
        and turn_ok
        and survival >= args.min_survival
        and heading <= args.max_heading
        and straight >= args.min_straight
        and inverted <= args.max_inverted
    )


def _value(record: dict[str, Any], key: str, missing: float) -> float:
    value = record.get(key)
    return missing if value is None else float(value)


def _score(record: dict[str, Any], args: argparse.Namespace) -> float:
    if args.max_speed_error is None:
        speed_term = _value(record, "speed_m_s", 0.0)
    else:
        error = _value(record, "speed_error_m_s", 1.0)
        speed_term = max(0.0, 1.0 - error / (3.0 * args.max_speed_error))
    survival = _value(record, "survival_fraction", 0.0)
    straight = _value(record, "straight_pose_survival_fraction", 0.0)
    heading = _value(record, "heading_error_deg", 90.0)
    inverted = _value(record, "head_inverted_time_fraction", 1.0)
    heading_quality = max(0.0, 1.0 - heading / 25.0)
    stress = _value(record, "stress_survival_fraction", 1.0)
    turn = 1.0
    if record.get("yaw_rate_error_rad_s") is not None:
        yaw_error = _value(record, "yaw_rate_error_rad_s", 1.0)
        turn = max(0.0, 1.0 - yaw_error / (3.0 * args.max_yaw_error)) * _value(
            record, "turn_survival_fraction", 0.0
        )
    return (
        speed_term * survival * straight * heading_quality * (1.0 - inverted) * stress * turn
    )


def _append(results_path: Path, record: dict[str, Any]) -> None:
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--load-run", default=str(DEFAULT_LOAD_RUN))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--algo-log-name", default="rsl_rl_ppo_straightspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-duration", type=float, default=60.0)
    parser.add_argument("--final-envs", type=int, default=128)
    parser.add_argument("--final-duration", type=float, default=300.0)
    parser.add_argument(
        "--eval-speeds",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[2.2],
        help="Comma-separated command speeds; every gate takes the worst speed.",
    )
    parser.add_argument(
        "--max-speed-error",
        type=float,
        default=None,
        help="Gate on |measured - command| instead of --min-speed.",
    )
    parser.add_argument(
        "--stress-speeds",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[],
        help="Also run the push/CoM/tilt stress battery at these command speeds.",
    )
    parser.add_argument("--stress-duration", type=float, default=60.0)
    parser.add_argument("--min-stress-survival", type=float, default=0.95)
    parser.add_argument(
        "--turn-speeds",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[],
        help="Forward commands for the turning battery (spawn-heading loop off).",
    )
    parser.add_argument(
        "--turn-yaw-rates",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[],
        help="Yaw-rate commands, rad/s; every speed is run with every yaw rate.",
    )
    parser.add_argument("--turn-duration", type=float, default=30.0)
    parser.add_argument("--max-yaw-error", type=float, default=0.1)
    parser.add_argument("--min-turn-survival", type=float, default=0.95)
    parser.add_argument(
        "--eval-clip",
        action="store_true",
        help="Score with BAM targets clamped to the joint limits; use with clipped owners.",
    )
    parser.add_argument(
        "--max-target-overshoot",
        type=float,
        default=None,
        help="Gate on the worst summed position-target overshoot past the joint limits, rad.",
    )
    parser.add_argument(
        "--max-head-forward",
        type=float,
        default=None,
        help="Gate on the worst head offset ahead of trunk_base, in meters.",
    )
    parser.add_argument("--min-speed", type=float, default=0.95)
    parser.add_argument("--min-survival", type=float, default=0.95)
    parser.add_argument("--max-heading", type=float, default=6.0)
    parser.add_argument("--min-straight", type=float, default=0.90)
    parser.add_argument("--max-inverted", type=float, default=0.02)
    parser.add_argument("--results-jsonl", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override passed to training, e.g. reward.tracking_lin_vel.weight=3.0",
    )
    args = parser.parse_args()
    if args.eval_clip:
        EVAL_FLAGS.append("--clip-targets")
    turn_cases = [(speed, yaw) for speed in args.turn_speeds for yaw in args.turn_yaw_rates]

    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    log_root = REPO_ROOT / "logs" / args.algo_log_name / "MicroduckSprintFlat"
    deadline = time.time() + max(args.budget_hours, 0.0) * 3600.0

    command = [
        "uv", "run", "--no-sync", "microduck-train",
        "--algo", "ppo",
        "--task", args.task,
        "--sim", "mujoco",
        f"algo.num_envs={args.num_envs}",
        f"algo.load_run={args.load_run}",
        f"algo.checkpoint={args.checkpoint}",
        f"algo.max_iterations={args.iterations}",
        f"algo.save_interval={args.save_interval}",
        f"algo.algo_log_name={args.algo_log_name}",
        f"algo.seed={args.seed}",
        "training.task_name=MicroduckSprintFlat",
        "training.no_play=true",
        "training.device=cuda:0",
        *args.override,
    ]
    print("+", " ".join(command), flush=True)
    started = time.time()
    training_log = REPO_ROOT / "logs" / f"straight_speed_{int(started)}.log"
    training_log.parent.mkdir(parents=True, exist_ok=True)
    with training_log.open("w", encoding="utf-8") as handle:
        trainer = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_train_env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(f"training log: {training_log}", flush=True)

    try:
        while trainer.poll() is None:
            if time.time() > deadline:
                print("budget exhausted, stopping training", flush=True)
                trainer.terminate()
                trainer.wait(timeout=300)
                break
            time.sleep(30)
    finally:
        if trainer.poll() is None:
            trainer.terminate()
            trainer.wait(timeout=300)

    if trainer.returncode not in (0, None, -15):
        print(f"training exited {trainer.returncode}", flush=True)

    run_dir = _newest_run(log_root, after=started)
    if run_dir is None:
        raise SystemExit(f"no run directory under {log_root}")
    checkpoints = _checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {run_dir}")

    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        record = _score_checkpoint(
            checkpoint,
            speeds=args.eval_speeds,
            num_envs=args.eval_envs,
            duration=args.eval_duration,
            seed=args.seed,
            stress_speeds=args.stress_speeds,
            stress_duration=args.stress_duration,
            turn_cases=turn_cases,
            turn_duration=args.turn_duration,
        )
        if record is None:
            continue
        record["passes_gate"] = _passes(record, args)
        record["score"] = _score(record, args)
        records.append(record)
        _append(results_path, record)

    passing = [record for record in records if record["passes_gate"]]
    if not passing:
        raise SystemExit("no checkpoint kept the straight-pose gate")
    best = max(passing, key=lambda record: _score(record, args))
    print(
        f"best gated checkpoint: {best['checkpoint']}\n"
        f"{json.dumps(best, ensure_ascii=False, indent=2)}",
        flush=True,
    )
    final = _score_checkpoint(
        Path(best["checkpoint"]),
        speeds=args.eval_speeds,
        num_envs=args.final_envs,
        duration=args.final_duration,
        seed=123,
        stress_speeds=args.stress_speeds,
        stress_duration=args.stress_duration,
        turn_cases=turn_cases,
        turn_duration=args.turn_duration,
    )
    if final is None:
        raise SystemExit("300 s battery failed")
    final["passes_gate"] = _passes(final, args)
    final["score"] = _score(final, args)
    final["label"] = "final_300s"
    _append(results_path, final)


if __name__ == "__main__":
    main()
