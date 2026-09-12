"""Gated speed-ramp for the human-like sprint gait contract.

Starts from scratch with alternate-step / stride / bow / head-upright rewards
already in the discovery owner. Each stage freezes the velocity command band
and only advances if eval clears survival, inversion, and speed gates. The
overnight curriculum that collapsed into standing is intentionally not used.

Usage:
    uv run --no-sync scripts/train_sprint_gait_ramp.py --budget-hours 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl_ppo_gait" / "MicroduckSprintFlat"
RESULTS_PATH = REPO_ROOT / "results" / "sprint_gait_ramp.jsonl"

# Frozen command bands. Eval uses the top of the band.
STAGES: tuple[dict[str, Any], ...] = (
    {"name": "S1", "vx": (0.20, 0.45), "iters": 1200, "eval_speed": 0.45, "min_speed": 0.18},
    {"name": "S2", "vx": (0.40, 0.65), "iters": 600, "eval_speed": 0.65, "min_speed": 0.28},
    {"name": "S3", "vx": (0.60, 0.90), "iters": 600, "eval_speed": 0.90, "min_speed": 0.40},
    {"name": "S4", "vx": (0.85, 1.20), "iters": 600, "eval_speed": 1.20, "min_speed": 0.55},
    {"name": "S5", "vx": (1.10, 1.50), "iters": 500, "eval_speed": 1.50, "min_speed": 0.70},
    {"name": "S6", "vx": (1.35, 1.80), "iters": 500, "eval_speed": 1.80, "min_speed": 0.85},
    {"name": "S7", "vx": (1.65, 2.20), "iters": 500, "eval_speed": 2.20, "min_speed": 0.95},
)


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _run_dirs() -> list[Path]:
    if not LOG_ROOT.is_dir():
        return []
    return [path for path in LOG_ROOT.glob("*_mujoco") if path.is_dir()]


def _latest_checkpoint(run_dir: Path) -> Path:
    models = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not models:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return models[-1]


def _append_result(record: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _payload_speed(payload: dict[str, Any]) -> float:
    block = payload.get("body_forward_speed_m_s") or {}
    value = block.get("mean")
    return float(value) if value is not None else 0.0


def _score(payload: dict[str, Any]) -> float:
    survival = float(payload.get("survival_fraction") or 0.0)
    speed = max(_payload_speed(payload), 0.0)
    invert = float(payload.get("head_inverted_time_fraction") or 0.0)
    yaw = float(payload.get("mean_absolute_heading_error_deg") or 180.0)
    heading = max(0.15, 1.0 - min(yaw, 180.0) / 180.0)
    return survival * speed * max(0.0, 1.0 - invert) * heading


def _passed(payload: dict[str, Any], stage: dict[str, Any]) -> bool:
    survival = float(payload.get("survival_fraction") or 0.0)
    speed = _payload_speed(payload)
    invert = float(payload.get("head_inverted_time_fraction") or 1.0)
    yaw = float(payload.get("mean_absolute_heading_error_deg") or 180.0)
    return (
        survival >= 0.75
        and speed >= float(stage["min_speed"])
        and invert <= 0.08
        and yaw <= 75.0
    )


def _train_chunk(
    *,
    label: str,
    iterations: int,
    vx: tuple[float, float],
    load_run: str,
    checkpoint: str,
    known_runs: set[Path],
    num_envs: int,
    learning_rate: float,
) -> tuple[Path, Path]:
    print(f"=== gait stage {label} vx={vx} iters={iterations} ===", flush=True)
    before = {path.resolve() for path in known_runs}
    started = time.time()
    command = [
        "uv",
        "run",
        "--no-sync",
        "microduck-train",
        "--algo",
        "ppo",
        "--task",
        "microduck_sprint_flat",
        "--sim",
        "mujoco",
        f"algo.num_envs={num_envs}",
        f"algo.max_iterations={iterations}",
        "algo.save_interval=200",
        "algo.algo_log_name=rsl_rl_ppo_gait",
        "training.task_name=MicroduckSprintFlat",
        "training.no_play=true",
        "env.curriculum.running_speed_range=null",
        "env.curriculum.action_rate_weight=null",
        f"env.commands.twist.ranges.lin_vel_x=[{vx[0]},{vx[1]}]",
        "env.commands.twist.ranges.lin_vel_y=[-0.01,0.01]",
        "env.commands.twist.ranges.ang_vel_z=[-0.02,0.02]",
        f"algo.algorithm.learning_rate={learning_rate}",
    ]
    if load_run and checkpoint:
        command.extend(
            [
                f"algo.load_run={load_run}",
                f"algo.checkpoint={checkpoint}",
                "algo.algorithm.schedule=fixed",
                "algo.algorithm.entropy_coef=0.01",
            ]
        )
    _run(command)
    created = [
        path
        for path in _run_dirs()
        if path.resolve() not in before and path.stat().st_mtime >= started - 5.0
    ]
    run_dir = max(created or _run_dirs(), key=lambda path: path.stat().st_mtime)
    return run_dir, _latest_checkpoint(run_dir)


def _eval_checkpoint(
    *,
    ckpt: Path,
    run_dir: Path,
    label: str,
    eval_envs: int,
    speed: float,
) -> dict[str, Any]:
    eval_out = run_dir / f"eval_{speed:.2f}_{eval_envs}_{label}.json"
    _run(
        [
            "uv",
            "run",
            "--no-sync",
            "scripts/eval_sprint_speed.py",
            str(ckpt),
            "--num-envs",
            str(eval_envs),
            "--speed",
            str(speed),
            "--warmup",
            "1",
            "--duration",
            "8",
            "--output",
            str(eval_out),
        ]
    )
    payload = json.loads(eval_out.read_text(encoding="utf-8"))
    record = {
        "label": label,
        "run_dir": run_dir.name,
        "checkpoint": ckpt.name,
        "eval_speed": speed,
        "survival_fraction": payload.get("survival_fraction"),
        "body_forward_speed_m_s": payload.get("body_forward_speed_m_s"),
        "mean_absolute_heading_error_deg": payload.get("mean_absolute_heading_error_deg"),
        "head_up_tilt_deg": payload.get("head_up_tilt_deg"),
        "head_inverted_time_fraction": payload.get("head_inverted_time_fraction"),
        "head_roll_abs_deg": payload.get("head_roll_abs_deg"),
        "head_yaw_abs_deg": payload.get("head_yaw_abs_deg"),
        "score": _score(payload),
        "eval_path": str(eval_out),
    }
    _append_result(record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--load-run", default="")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(args.budget_hours, 0.0) * 3600.0
    known_runs = set(_run_dirs())
    load_run = args.load_run
    checkpoint = args.checkpoint
    best: dict[str, Any] | None = None
    best_payload: dict[str, Any] | None = None
    best_ckpt: tuple[str, str] | None = None

    for index, stage in enumerate(STAGES):
        if time.time() + 20.0 * 60.0 >= deadline:
            print("budget exhausted before next stage", flush=True)
            break
        attempts = 0
        advanced = False
        while attempts < 2 and time.time() + 20.0 * 60.0 < deadline:
            attempts += 1
            label = str(stage["name"]) if attempts == 1 else f"{stage['name']}retry"
            learning_rate = 1.0e-3 if not load_run else 2.0e-4
            run_dir, ckpt = _train_chunk(
                label=label,
                iterations=int(stage["iters"]),
                vx=stage["vx"],
                load_run=load_run,
                checkpoint=checkpoint,
                known_runs=known_runs,
                num_envs=args.num_envs,
                learning_rate=learning_rate,
            )
            known_runs.add(run_dir)
            payload = _eval_checkpoint(
                ckpt=ckpt,
                run_dir=run_dir,
                label=label,
                eval_envs=args.eval_envs,
                speed=float(stage["eval_speed"]),
            )
            if best is None or _score(payload) > float(best["score"]):
                best = {
                    "label": label,
                    "score": _score(payload),
                    "run_dir": run_dir.name,
                    "checkpoint": ckpt.name,
                }
                best_payload = payload
                best_ckpt = (run_dir.name, ckpt.name)
            if _passed(payload, stage):
                load_run = run_dir.name
                checkpoint = ckpt.name
                advanced = True
                break
            print(f"{label} missed gate; keeping previous resume source", flush=True)
            if best_ckpt is not None:
                load_run, checkpoint = best_ckpt
        if not advanced:
            print(f"holding at {stage['name']}; not raising the command band", flush=True)
            if index == 0 and (best_payload is None or _payload_speed(best_payload) < 0.08):
                print("no locomotion emerged; stopping rather than ramping a stand", flush=True)
                break

    if best_ckpt is not None and time.time() + 8.0 * 60.0 < deadline:
        run_dir = LOG_ROOT / best_ckpt[0]
        ckpt = run_dir / best_ckpt[1]
        print("=== official 2.2 m/s battery on best gait checkpoint ===", flush=True)
        _eval_checkpoint(
            ckpt=ckpt,
            run_dir=run_dir,
            label="official_2p2",
            eval_envs=512,
            speed=2.2,
        )
    print("=== gait ramp done ===", flush=True)
    if best is not None:
        print("best", json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
