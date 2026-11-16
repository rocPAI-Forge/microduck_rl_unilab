"""Run the stilt curriculum: support blend at 2 cm, then height.

After the playground's executed lineage (experiments/stilts/TRAINING.md in
Vottivott/microduck-playground): support is narrowed at a fixed 2 cm, then the
blend-0.50 tip is raised; height and support never get harder in the same
step. Each stage writes its morphology with ``make_stilt_robot.py``, resumes
the previous stage's last checkpoint, and is scored by ``eval_walk.py`` at
0.15 m/s. Iterations are about twice the playground's per stage, since this
machine runs 2048 environments instead of 4096.

Usage:
    uv run --no-sync scripts/train_stilts.py --load-run examples/velocity_flat_mujoco --checkpoint 499
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stilt_robot import stilt_tag, write_stage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = "microduck_stilt_flat/mujoco"
# (height_cm, blend, iterations)
STAGES = (
    (2.0, 0.0, 800),
    (2.0, 0.25, 200),
    (2.0, 0.5, 200),
    (3.0, 0.5, 200),
    (4.0, 0.5, 200),
    (5.0, 0.5, 1500),
    (7.5, 0.5, 200),
    (10.0, 0.5, 300),
)


def _stage_overrides(height_cm: float, blend: float) -> list[str]:
    paths = write_stage(height_cm, blend)
    rel = {k: str(v.relative_to(REPO_ROOT)) for k, v in paths.items()}
    return [
        f"env.scene.model_file={rel['scene']}",
        f"env.scene.fragment_files=[{rel['task']}]",
        f"reward.foot_slip.weight={-0.1 - 0.1 * blend:g}",
    ]


def _latest(run_root: Path) -> Path:
    checkpoints = sorted(run_root.rglob("model_*.pt"), key=lambda p: (p.stat().st_mtime, p.name))
    if not checkpoints:
        raise RuntimeError(f"no checkpoint under {run_root}")
    return checkpoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-run", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--first-stage", type=int, default=0)
    parser.add_argument("--last-stage", type=int, default=len(STAGES) - 1)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--prefix", default="rsl_rl_ppo_stilt")
    args = parser.parse_args()

    env = dict(os.environ, HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0")
    env.setdefault("MUJOCO_GL", "egl")
    load_run, checkpoint = args.load_run.resolve(), str(args.checkpoint)
    results = []
    for index in range(args.first_stage, args.last_stage + 1):
        height_cm, blend, iterations = STAGES[index]
        tag = stilt_tag(height_cm, blend)
        overrides = _stage_overrides(height_cm, blend)
        log_name = f"{args.prefix}_{index}_{tag}"
        print(f"stage {index}: {tag}, {iterations} iterations from {load_run} @ {checkpoint}", flush=True)
        command = [
            "uv", "run", "--no-sync", "microduck-train", "--algo", "ppo", "--task", "microduck_stilt_flat",
            "--sim", "mujoco",
            f"algo.num_envs={args.num_envs}",
            f"algo.max_iterations={iterations}",
            "algo.save_interval=100",
            f"algo.load_run={load_run}",
            f"algo.checkpoint={checkpoint}",
            f"algo.algo_log_name={log_name}",
            "training.no_play=true",
            "training.device=cuda:0",
            *overrides,
        ]
        log_path = REPO_ROOT / "logs" / f"{log_name}.log"
        with log_path.open("w") as log:
            if subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode:
                print(f"stage {index} failed; see {log_path}", flush=True)
                return 1
        latest = _latest(REPO_ROOT / "logs" / log_name)
        load_run, checkpoint = latest.parent, latest.stem.split("_")[1]
        out = latest.with_name(f"walk_{latest.stem}_v0.15.json")
        subprocess.run(
            ["uv", "run", "--no-sync", "scripts/eval_walk.py", str(latest), "--owner", OWNER,
             "--speed", "0.15", "--num-envs", "64", "--output", str(out)]
            + [arg for o in overrides for arg in ("--override", o)],
            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        score = json.loads(out.read_text()) if out.is_file() else {}
        row = {"stage": index, "tag": tag, "checkpoint": str(latest),
               "survival": score.get("survival_fraction"), "speed": score.get("forward_speed_m_s")}
        results.append(row)
        print(json.dumps(row), flush=True)
    (REPO_ROOT / "logs" / f"{args.prefix}_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
