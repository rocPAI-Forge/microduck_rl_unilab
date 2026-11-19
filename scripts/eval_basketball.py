#!/usr/bin/env python3
"""Score a basketball checkpoint on a free ball at a fixed twist command.

After the playground's matched battery: hold 0 (free ball), curricula off,
episodes longer than the run, first fall counts as failure. Reports survival
at 10/20/60 s (as far as ``--duration`` reaches) and, over surviving samples,
the mean body-forward speed and the yaw-rate tracking error. Pushes stay on
(0.3x the walking recipe) unless ``--no-push``.

Usage:
    uv run --no-sync scripts/eval_basketball.py logs/.../model_5999.pt --speed 0.1 --duration 60
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root
from unilab.training import algo_config_dict

from microduck_rl_unilab.conf_searchpath import register_conf_search_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={args.owner}", *args.override])
    with open_dict(cfg):
        cfg.algo.seed = args.seed
        cfg.env.seed = args.seed
        cfg.env.max_episode_seconds = args.duration + 5.0
        cfg.env.actions.ball_hold.levels = [args.hold]
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [args.speed, args.speed]
        twist.ranges.lin_vel_y = [args.lateral, args.lateral]
        twist.ranges.ang_vel_z = [args.yaw_rate, args.yaw_rate]
        twist.resampling_time_range = [args.duration + 5.0, args.duration + 5.0]
        for key in (
            "rel_standing_envs",
            "rel_heading_envs",
            "rel_world_envs",
            "rel_forward_envs",
            "turn_in_place_fraction",
        ):
            if key in twist:
                twist[key] = 0.0
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        if args.no_push:
            cfg.env.events.push_robot = None
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    raw_env = cast(Any, registry.make(str(cfg.training.task_name), sim_backend="mujoco",
                                      num_envs=args.num_envs, env_cfg_override=override))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = RslRlVecEnvWrapper(raw_env, device=device)
    try:
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        if isinstance(train_cfg.get("algorithm"), dict):
            train_cfg["algorithm"]["enable_compile"] = False
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
        runner.load(str(args.checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        obs = env.get_observations()
        robot = raw_env.scene["robot"]
        dt = float(raw_env.cfg.ctrl_dt)
        alive = np.ones(args.num_envs, dtype=bool)
        survival: dict[str, float] = {}
        speed_sum = yaw_err_sum = yaw_sum = 0.0
        samples = 0
        for step in range(1, round(args.duration / dt) + 1):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.cpu().numpy().astype(bool)
            t = step * dt
            if alive.any() and t >= 1.0:
                speed_sum += float(np.sum(np.asarray(robot.data.root_link_lin_vel_b)[alive, 0]))
                yaw = np.asarray(robot.data.root_link_ang_vel_w)[alive, 2]
                yaw_err_sum += float(np.sum(np.abs(yaw - args.yaw_rate)))
                yaw_sum += float(np.sum(yaw))
                samples += int(alive.sum())
            for mark in (10, 20, 60):
                if abs(t - mark) < dt / 2:
                    survival[f"survival_{mark}s"] = float(alive.mean())
        return {
            "checkpoint": str(args.checkpoint),
            "command": [args.speed, args.lateral, args.yaw_rate],
            "hold": args.hold,
            "push": not args.no_push,
            "duration_s": args.duration,
            "num_envs": args.num_envs,
            "seed": args.seed,
            **survival,
            "survival_end": float(alive.mean()),
            "forward_speed_m_s": speed_sum / samples if samples else None,
            "yaw_rate_abs_error": yaw_err_sum / samples if samples else None,
            "yaw_rate_mean": yaw_sum / samples if samples else None,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", default="microduck_basketball_flat/mujoco")
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
