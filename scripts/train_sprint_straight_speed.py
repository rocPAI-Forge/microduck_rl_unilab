"""Raise along-ray speed of the published straight sprint without leaving the ray.

Resumes ``examples/sprint_straight_long/model_19297.pt`` on
``microduck_sprint_straightspeed_flat``. Body-frame forward progress stays at
the straightfix value. The new reward is capped spawn-ray speed, ramped
together with the heading penalty.

Checkpoints are scored after training, on the same closed-loop owner the
policy was trained with. A checkpoint counts only when a 60 s battery keeps
the head upright, the heading inside the published band, and the sticky
straight-pose gate up. The fastest such checkpoint then gets the 300 s battery.

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


def _score_checkpoint(
    checkpoint: Path,
    *,
    num_envs: int,
    duration: float,
    seed: int,
) -> dict[str, Any] | None:
    battery_out = checkpoint.with_name(
        f"battery_{checkpoint.stem}_{int(duration)}s_seed{seed}.json"
    )
    battery = _run_json(
        [
            "uv", "run", "--no-sync", "scripts/eval_sprint_speed.py", str(checkpoint),
            "--owner", OWNER,
            "--speed", "2.2",
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
    return {
        "checkpoint": str(checkpoint),
        "iteration": int(checkpoint.stem.split("_")[1]),
        "duration_s": duration,
        "seed": seed,
        "speed_m_s": speed_block.get("mean"),
        "spawn_forward_m": (battery.get("final_spawn_forward_displacement_m") or {}).get("mean"),
        "survival_fraction": battery.get("survival_fraction"),
        "heading_error_deg": battery.get("mean_absolute_heading_error_deg"),
        "straight_pose_survival_fraction": battery.get("straight_pose_survival_fraction"),
        "straight_pose_horizon_s": horizon.get("mean"),
        "head_inverted_time_fraction": battery.get("head_inverted_time_fraction"),
        "head_up_tilt_deg": battery.get("head_up_tilt_deg"),
    }


def _passes(record: dict[str, Any], args: argparse.Namespace) -> bool:
    values = (
        record.get("speed_m_s"),
        record.get("survival_fraction"),
        record.get("heading_error_deg"),
        record.get("straight_pose_survival_fraction"),
        record.get("head_inverted_time_fraction"),
    )
    if any(value is None for value in values):
        return False
    speed, survival, heading, straight, inverted = (float(value) for value in values)
    return (
        speed >= args.min_speed
        and survival >= args.min_survival
        and heading <= args.max_heading
        and straight >= args.min_straight
        and inverted <= args.max_inverted
    )


def _value(record: dict[str, Any], key: str, missing: float) -> float:
    value = record.get(key)
    return missing if value is None else float(value)


def _score(record: dict[str, Any]) -> float:
    speed = _value(record, "speed_m_s", 0.0)
    survival = _value(record, "survival_fraction", 0.0)
    straight = _value(record, "straight_pose_survival_fraction", 0.0)
    heading = _value(record, "heading_error_deg", 90.0)
    inverted = _value(record, "head_inverted_time_fraction", 1.0)
    heading_quality = max(0.0, 1.0 - heading / 25.0)
    return speed * survival * straight * heading_quality * (1.0 - inverted)


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
            num_envs=args.eval_envs,
            duration=args.eval_duration,
            seed=args.seed,
        )
        if record is None:
            continue
        record["passes_gate"] = _passes(record, args)
        record["score"] = _score(record)
        records.append(record)
        _append(results_path, record)

    passing = [record for record in records if record["passes_gate"]]
    if not passing:
        raise SystemExit("no checkpoint kept the straight-pose gate")
    best = max(passing, key=_score)
    print(
        f"best gated checkpoint: {best['checkpoint']}\n"
        f"{json.dumps(best, ensure_ascii=False, indent=2)}",
        flush=True,
    )
    final = _score_checkpoint(
        Path(best["checkpoint"]),
        num_envs=args.final_envs,
        duration=args.final_duration,
        seed=123,
    )
    if final is None:
        raise SystemExit("300 s battery failed")
    final["passes_gate"] = _passes(final, args)
    final["score"] = _score(final)
    final["label"] = "final_300s"
    _append(results_path, final)


if __name__ == "__main__":
    main()
