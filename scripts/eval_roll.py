#!/usr/bin/env python3
"""Score a forward-roll checkpoint (roulade or tumble) from standing starts.

Works with ``microduck_roulade_flat`` (reads ``env._roulade_state``: rotation
maximum and the head-top latch) and ``microduck_tumble_flat`` /
``microduck_roll_flat`` (reads ``env._roll_state``). Every episode starts
standing; curricula and pushes are off. Reports, per env:
  - roll complete: rotation maximum past ``--complete-deg`` (and, for the
    roulade, the head-top latch set);
  - time to complete, peak forward pitch rate;
  - stood up after the roll (root above 0.10 m and upright cos > 0.9) at any
    time, and standing at the end;
  - more than one turn (rotation maximum past 1.5 turns).

    uv run --no-sync python scripts/eval_roll.py examples/roulade/model_999.pt \
        --owner microduck_roulade_flat/mujoco --num-envs 512 --seed 101
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
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
STAND_Z = 0.10
UPRIGHT_COS = 0.9


def _rotation(raw_env: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """(rotation maximum, head latch or None) from whichever roll state exists."""
    roulade = getattr(raw_env, "_roulade_state", None)
    if roulade is not None:
        return np.asarray(roulade.max), np.asarray(roulade.latch)
    return np.asarray(raw_env._roll_state.peak), None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={args.owner}", *args.override])
    with open_dict(cfg):
        cfg.algo.seed = args.seed
        cfg.env.seed = args.seed
        cfg.env.max_episode_seconds = args.duration + 1.0
        events = cfg.env.events
        if "reset_roulade" in events and events.reset_roulade is not None:
            events.reset_roulade.params.standing_prob = 1.0
            events.reset_roulade.params.midroll_prob = 0.0
        if "set_ground_state" in events and events.set_ground_state is not None:
            g = events.set_ground_state.params
            g.standing_prob, g.sitting_prob, g.face_down_prob, g.face_up_prob = 1.0, 0.0, 0.0, 0.0
        if "push_robot" in events:
            events.push_robot = None
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
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
        n = args.num_envs
        threshold = math.radians(args.complete_deg)
        t_done = np.full(n, np.nan)
        t_stood = np.full(n, np.nan)
        peak_rate = np.zeros(n)
        for step in range(round(args.duration / dt)):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            t = (step + 1) * dt
            rot_max, latch = _rotation(raw_env)
            done = rot_max >= threshold
            if latch is not None:
                done &= latch
            t_done[done & np.isnan(t_done)] = t
            z = np.asarray(robot.data.root_link_pos_w)[:, 2]
            up = -np.asarray(robot.data.projected_gravity_b)[:, 2]
            stood = done & (z > STAND_Z) & (up > UPRIGHT_COS)
            t_stood[stood & np.isnan(t_stood)] = t
            peak_rate = np.maximum(peak_rate, np.abs(np.asarray(robot.data.root_link_ang_vel_b)[:, 1]))
        rot_max, latch = _rotation(raw_env)
        z = np.asarray(robot.data.root_link_pos_w)[:, 2]
        up = -np.asarray(robot.data.projected_gravity_b)[:, 2]
        complete = ~np.isnan(t_done)
        result: dict[str, Any] = {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "num_envs": n,
            "seed": args.seed,
            "duration_s": args.duration,
            "roll_complete": float(complete.mean()),
            "time_to_complete_s_median": float(np.nanmedian(t_done)) if complete.any() else None,
            "rotation_max_deg_median": float(np.degrees(np.median(rot_max))),
            "rotation_max_deg_p90": float(np.degrees(np.percentile(rot_max, 90))),
            "more_than_one_turn": float((rot_max >= 3.0 * math.pi).mean()),
            "peak_pitch_rate_median": float(np.median(peak_rate)),
            "stood_up_after_roll": float((~np.isnan(t_stood)).mean()),
            "time_to_stand_s_median": float(np.nanmedian(t_stood)) if (~np.isnan(t_stood)).any() else None,
            "standing_at_end": float(((z > STAND_Z) & (up > UPRIGHT_COS)).mean()),
        }
        if latch is not None:
            result["head_top_latch"] = float(np.asarray(latch).mean())
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", default="microduck_roulade_flat/mujoco")
    parser.add_argument("--duration", type=float, default=4.9)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--complete-deg", type=float, default=260.0)
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
