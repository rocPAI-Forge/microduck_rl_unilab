"""Score a sit/stand checkpoint on a fixed command schedule.

Every environment resets standing and is commanded: stand until
``--sit-at``, sit until ``--stand-at``, stand until ``--duration``. Reports:

* sit success: trunk within ``--band`` of the sit height and tilt under 25 deg
  at the end of the sit window;
* seated hold: fraction of the last 3 s of the sit window spent seated upright;
* rise success: trunk within ``--band`` of the stand height and upright at the
  end;
* peak descent and rise speeds (trunk v_z) and falls (tilt over 60 deg).

Usage:
    uv run --no-sync scripts/eval_sitstand.py logs/.../model_2499.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
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
SIT_Z = 0.060
STAND_Z = 0.115


def _compose(owner: str, seconds: float, seed: int, extra: list[str]) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}", *extra])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.seed = seed
        cfg.env.max_episode_seconds = seconds + 1.0
        cfg.env.commands.twist.resampling_time_range = [seconds + 1.0, seconds + 1.0]
        cfg.env.commands.twist.sit_prob = 0.0
        ground = cfg.env.events.set_ground_state.params
        ground.standing_prob, ground.sitting_prob = 1.0, 0.0
        ground.face_down_prob, ground.face_up_prob = 0.0, 0.0
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        for name in cfg.reward:
            cfg.reward[name] = None
    return cfg


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _compose(args.owner, args.duration, args.seed, args.override)
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
        twist = raw_env.command_manager.get_term("twist")
        dt = float(raw_env.cfg.ctrl_dt)
        steps = round(args.duration / dt)
        sit_step, stand_step = round(args.sit_at / dt), round(args.stand_at / dt)
        hold_from = stand_step - round(3.0 / dt)
        n = args.num_envs
        fallen = np.zeros(n, dtype=bool)
        seated_steps = np.zeros(n)
        sat = np.zeros(n, dtype=bool)
        down_peak = np.zeros(n)
        up_peak = np.zeros(n)
        z_at_stand = np.full(n, np.nan)
        for step in range(steps):
            flag = 1.0 if sit_step <= step < stand_step else 0.0
            twist.vel_command_b[:, 0] = flag
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            z = np.asarray(robot.data.root_link_pos_w[:, 2])
            gravity = np.asarray(robot.data.projected_gravity_b)
            tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
            vz = np.asarray(robot.data.root_link_lin_vel_w[:, 2])
            fallen |= tilt > 60.0
            down_peak = np.maximum(down_peak, -vz)
            up_peak = np.maximum(up_peak, vz)
            seated = (np.abs(z - SIT_Z) < args.band) & (tilt < 25.0)
            if hold_from <= step < stand_step:
                seated_steps += seated
            if step == stand_step - 1:
                sat = seated.copy()
                z_at_stand = z.copy()
        stood = (np.abs(z - STAND_Z) < args.band) & (tilt < 25.0)
        return {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "num_envs": n,
            "seed": args.seed,
            "schedule_s": {"sit_at": args.sit_at, "stand_at": args.stand_at, "end": args.duration},
            "sit_success": float(np.mean(sat & ~fallen)),
            "seated_hold_fraction": float(np.mean(seated_steps / max(stand_step - hold_from, 1))),
            "rise_success": float(np.mean(stood & ~fallen)),
            "fall_fraction": float(np.mean(fallen)),
            "descent_speed_p95_m_s": float(np.quantile(down_peak, 0.95)),
            "rise_speed_p95_m_s": float(np.quantile(up_peak, 0.95)),
            "final_z_mean_m": float(np.mean(z)),
            "final_z_p10_p50_p90_m": [float(v) for v in np.quantile(z[~fallen], [0.1, 0.5, 0.9])] if (~fallen).any() else None,
            "final_tilt_p50_p90_deg": [float(v) for v in np.quantile(tilt[~fallen], [0.5, 0.9])] if (~fallen).any() else None,
            "seated_z_at_stand_cmd_p50_m": float(np.median(z_at_stand[~fallen])) if (~fallen).any() else None,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", default="microduck_sitstand_flat/mujoco")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--sit-at", type=float, default=1.0)
    parser.add_argument("--stand-at", type=float, default=7.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--band", type=float, default=0.012)
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
