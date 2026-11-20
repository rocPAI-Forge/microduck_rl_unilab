"""Score a walking checkpoint under any owner at a fixed twist command.

Builds the owner's own task, pins the twist command, turns curricula and
rewards off, and reports survival (tilt under 60 deg), mean body-forward
speed, and mean yaw rate over ``--duration`` after a 1 s warm-up.

Usage:
    uv run --no-sync scripts/eval_walk.py logs/.../model_899.pt \
        --owner microduck_stilt_flat/mujoco --speed 0.15 \
        --override env.scene.model_file=assets/robots/microduck/generated/scene_flat_bam_stilt_h2_b050.xml
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
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
    total = args.duration + 1.0
    with open_dict(cfg):
        cfg.algo.seed = args.seed
        cfg.env.max_episode_seconds = total + 1.0
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [args.speed, args.speed]
        twist.ranges.lin_vel_y = [0.0, 0.0]
        twist.ranges.ang_vel_z = [args.yaw_rate, args.yaw_rate]
        twist.resampling_time_range = [total + 1.0, total + 1.0]
        for key in ("rel_standing_envs", "rel_heading_envs", "turn_in_place_fraction"):
            if key in twist:
                twist[key] = 0.0
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        for name in cfg.reward:
            cfg.reward[name] = None
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    raw_env = cast(
        Any,
        registry.make(
            str(cfg.training.task_name),
            sim_backend=str(cfg.training.sim_backend),
            num_envs=args.num_envs,
            env_cfg_override=override,
        ),
    )
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
        speed_sum = yaw_sum = 0.0
        samples = 0
        for step in range(round(total / dt)):
            with torch.inference_mode():
                obs, _, dones, infos = env.step(policy(obs))
            terminated = dones.cpu().numpy().astype(bool)
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                terminated &= ~time_outs.cpu().numpy().astype(bool)
            gravity = np.asarray(robot.data.projected_gravity_b)
            alive &= ~terminated & (np.degrees(np.arccos(np.clip(-gravity[:, 2], -1, 1))) <= 60.0)
            if step * dt >= 1.0 and alive.any():
                speed_sum += float(np.sum(np.asarray(robot.data.root_link_lin_vel_b)[alive, 0]))
                yaw_sum += float(np.sum(np.asarray(robot.data.root_link_ang_vel_w)[alive, 2]))
                samples += int(alive.sum())
        return {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "overrides": list(args.override),
            "command_m_s": args.speed,
            "command_yaw_rate_rad_s": args.yaw_rate,
            "duration_s": args.duration,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "survival_fraction": float(alive.mean()),
            "forward_speed_m_s": speed_sum / samples if samples else None,
            "yaw_rate_rad_s": yaw_sum / samples if samples else None,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
