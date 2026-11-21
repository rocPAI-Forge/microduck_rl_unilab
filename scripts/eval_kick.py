#!/usr/bin/env python3
"""Score a ball-kicking checkpoint: ball travel along the goal, survival, touches.

Runs ``microduck_kick_flat`` (or another kick owner) with pushes off. Every
episode resets the duck, the ball in front of it and a goal direction
(``reset_kick``); ``BallPursuitCommand`` drives the twist command. Reports:
  - survival: envs that did not fall within ``--duration``;
  - touched: the ball moved more than 2 cm;
  - ball displacement along the goal direction (median over all envs and over
    survivors) and the share past 0.5 m; mean displacement across the goal.

    uv run --no-sync python scripts/eval_kick.py logs/.../model_60499.pt --seed 101
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
        cfg.env.max_episode_seconds = args.duration + 2.0
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
        robot, ball = raw_env.scene["robot"], raw_env.scene["ball"]
        goal_angle = np.asarray(raw_env._kick_state.goal).copy()
        goal = np.stack((np.cos(goal_angle), np.sin(goal_angle)), axis=1)
        ball0 = np.asarray(ball.data.root_link_pos_w)[:, :2].copy()
        dt = float(raw_env.cfg.ctrl_dt)
        n = args.num_envs
        alive = np.ones(n, dtype=bool)
        touched = np.zeros(n, dtype=bool)
        for _ in range(round(args.duration / dt)):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.cpu().numpy().astype(bool)
            touched |= np.linalg.norm(np.asarray(ball.data.root_link_pos_w)[:, :2] - ball0, axis=1) > 0.02
        disp = np.asarray(ball.data.root_link_pos_w)[:, :2] - ball0
        along = np.sum(disp * goal, axis=1)
        across = disp[:, 0] * goal[:, 1] - disp[:, 1] * goal[:, 0]
        return {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "num_envs": n,
            "seed": args.seed,
            "duration_s": args.duration,
            "survival": float(alive.mean()),
            "touched": float(touched.mean()),
            "ball_along_goal_m_median_all": float(np.median(along)),
            "ball_along_goal_m_median_survivors": float(np.median(along[alive])) if alive.any() else None,
            "ball_past_0p5m_all": float((along > 0.5).mean()),
            "ball_across_goal_m_mean_abs": float(np.abs(across).mean()),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", default="microduck_kick_flat/mujoco")
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
