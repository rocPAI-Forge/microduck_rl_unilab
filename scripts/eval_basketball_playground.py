#!/usr/bin/env python3
"""Score a basketball checkpoint under a playground-style protocol.

Closer to the released playground battery than eval_basketball.py:
  - random commands from the owner's training ranges, resampled as in training
    (the owner's ranges are already the playground's 2x scale: +-0.30 m/s,
    +-0.20 m/s lateral, +-1.0 rad/s yaw);
  - pushes every 0.5 to 1.0 s, +-0.09 m/s (not the 3 to 6 s of training);
  - free ball (hold 0), curricula off, first fall counts as failure;
  - 1024 envs, 60 s.

Metrics, over surviving env-steps after the first second:
  - survival at 10/20/60 s;
  - forward MAE: mean |body vx - commanded vx|;
  - yaw MAE: mean |body yaw rate - commanded wz|;
  - baselines: the same MAE for a predictor that outputs 0, and one that
    outputs the mean body vx of the run (a constant). A policy that does not
    follow commands cannot beat the constant baseline.

    uv run --no-sync python scripts/eval_basketball_playground.py \
        logs/.../model_12491.pt --owner microduck_basketball_ft9_flat/mujoco
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
from unilab.training import algo_config_dict

from microduck_rl_unilab.conf_searchpath import register_conf_search_path

import eval_basketball as eb


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(eb.CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={args.owner}"])
    with open_dict(cfg):
        cfg.algo.seed = args.seed
        cfg.env.seed = args.seed
        cfg.env.max_episode_seconds = args.duration + 5.0
        cfg.env.actions.ball_hold.levels = [args.hold]
        twist = cfg.env.commands.twist
        for key in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs",
                    "turn_in_place_fraction"):
            if key in twist:
                twist[key] = 0.0
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        push = cfg.env.events.push_robot
        if push is not None:
            push.interval_range_s = [args.push_min, args.push_max]
            push.params.velocity_range = {"x": [-args.push, args.push], "y": [-args.push, args.push]}
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=eb.REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
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
        cmd_x_all, vx_all, cmd_w_all, wz_all = [], [], [], []
        for step in range(1, round(args.duration / dt) + 1):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.cpu().numpy().astype(bool)
            t = step * dt
            if alive.any() and t >= 1.0:
                cmd = np.asarray(raw_env.command_manager.get_command("twist"), dtype=np.float64)
                vx = np.asarray(robot.data.root_link_lin_vel_b, dtype=np.float64)[:, 0]
                wz = np.asarray(robot.data.root_link_ang_vel_b, dtype=np.float64)[:, 2]
                cmd_x_all.append(cmd[alive, 0]); vx_all.append(vx[alive])
                cmd_w_all.append(cmd[alive, 2]); wz_all.append(wz[alive])
            for mark in (10, 20, 60):
                if abs(t - mark) < dt / 2:
                    survival[f"survival_{mark}s"] = float(alive.mean())
        cx = np.concatenate(cmd_x_all); vx = np.concatenate(vx_all)
        cw = np.concatenate(cmd_w_all); wz = np.concatenate(wz_all)
        return {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "num_envs": args.num_envs,
            "duration_s": args.duration,
            "push": {"interval_s": [args.push_min, args.push_max], "velocity": args.push},
            "seed": args.seed,
            **survival,
            "samples": int(cx.size),
            "forward_mae": float(np.mean(np.abs(vx - cx))),
            "forward_mae_zero_predictor": float(np.mean(np.abs(cx))),
            "forward_mae_constant_predictor": float(np.mean(np.abs(vx - vx.mean()))),
            "forward_mean_vx": float(vx.mean()),
            "yaw_mae": float(np.mean(np.abs(wz - cw))),
            "yaw_mae_zero_predictor": float(np.mean(np.abs(cw))),
            "yaw_mae_constant_predictor": float(np.mean(np.abs(wz - wz.mean()))),
            "yaw_mean_wz": float(wz.mean()),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", default="microduck_basketball_ft9_flat/mujoco")
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--push", type=float, default=0.09)
    parser.add_argument("--push-min", type=float, default=0.5)
    parser.add_argument("--push-max", type=float, default=1.0)
    parser.add_argument("--device", default=None)
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
