#!/usr/bin/env python3
"""Run a ball-kick policy (rsl_rl checkpoint or exported ONNX) and score the kick.

Standing start on ``microduck_ballkick_flat`` with the ball in front of the
kicking foot; the policy runs for ``--duration`` with zero commands (it kicks
and then stands back up on its own). ``--policy-s`` stops it earlier and holds
the default joint targets instead; that is not a stand (with no balance
control the duck falls), only a way to look at the kick on its own.
Reports, over all envs:
  - ball speed along the robot's reset heading: peak, and mean over envs;
  - ball displacement along the heading and across it at the end;
  - kicked: ball moved more than 5 cm along the heading;
  - robot standing at the end (root above 0.10 m, upright cos > 0.9).

    uv run --no-sync python scripts/eval_ballkick.py ball_kick_right.onnx --foot right
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


def _zero_commands(cfg: Any) -> None:
    twist = cfg.env.commands.twist
    twist.ranges.lin_vel_x = [0.0, 0.0]
    twist.ranges.lin_vel_y = [0.0, 0.0]
    twist.ranges.ang_vel_z = [0.0, 0.0]
    for key in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs", "turn_in_place_fraction"):
        if key in twist:
            twist[key] = 0.0
    for name in ("head_pose", "body_pose"):
        command = cfg.env.commands[name]
        command.ranges = [[0.0, 0.0] for _ in command.ranges]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={args.owner}", *args.override])
    with open_dict(cfg):
        cfg.algo.seed = args.seed
        cfg.env.seed = args.seed
        cfg.env.max_episode_seconds = args.duration + 1.0
        cfg.env.events.reset_ballkick.params.foot = args.foot
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        _zero_commands(cfg)
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    raw_env = cast(Any, registry.make(str(cfg.training.task_name), sim_backend="mujoco",
                                      num_envs=args.num_envs, env_cfg_override=override))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = RslRlVecEnvWrapper(raw_env, device=device)
    try:
        if args.policy.suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(args.policy))
            name = session.get_inputs()[0].name

            def policy(obs: Any) -> torch.Tensor:
                o = obs["policy"].detach().cpu().numpy().astype(np.float32)
                a = np.concatenate([session.run(None, {name: o[i:i + 1]})[0] for i in range(o.shape[0])], axis=0)
                return torch.as_tensor(a, device=device)
        else:
            train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
            train_cfg.setdefault("runner", {})["logger"] = "none"
            train_cfg["logger"] = "none"
            if isinstance(train_cfg.get("algorithm"), dict):
                train_cfg["algorithm"]["enable_compile"] = False
            runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
            runner.load(str(args.policy), map_location=device)
            policy = runner.get_inference_policy(device=device)

        obs = env.get_observations()
        robot, ball = raw_env.scene["robot"], raw_env.scene["ball"]
        goal_angle = np.asarray(raw_env._kick_state.goal).copy()
        goal = np.stack((np.cos(goal_angle), np.sin(goal_angle)), axis=1)
        ball0 = np.asarray(ball.data.root_link_pos_w)[:, :2].copy()
        dt = float(raw_env.cfg.ctrl_dt)
        n = args.num_envs
        policy_s = args.duration if args.policy_s is None else args.policy_s
        hold = torch.zeros((n, 14), device=device)
        peak_speed = np.zeros(n)
        for step in range(round(args.duration / dt)):
            t = step * dt
            with torch.inference_mode():
                action = policy(obs) if t < policy_s else hold
                obs, _, _, _ = env.step(action)
            v = np.asarray(ball.data.root_link_lin_vel_w)[:, :2]
            peak_speed = np.maximum(peak_speed, np.sum(v * goal, axis=1))
        disp = np.asarray(ball.data.root_link_pos_w)[:, :2] - ball0
        along = np.sum(disp * goal, axis=1)
        across = disp[:, 0] * goal[:, 1] - disp[:, 1] * goal[:, 0]
        z = np.asarray(robot.data.root_link_pos_w)[:, 2]
        up = -np.asarray(robot.data.projected_gravity_b)[:, 2]
        return {
            "policy": str(args.policy),
            "owner": args.owner,
            "foot": args.foot,
            "num_envs": n,
            "seed": args.seed,
            "policy_s": policy_s,
            "duration_s": args.duration,
            "kicked": float((along > 0.05).mean()),
            "peak_ball_speed_m_s_median": float(np.median(peak_speed)),
            "ball_along_m_median": float(np.median(along)),
            "ball_across_m_mean_abs": float(np.abs(across).mean()),
            "standing_at_end": float(((z > 0.10) & (up > 0.9)).mean()),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", type=Path, help="rsl_rl checkpoint (.pt) or exported ONNX")
    parser.add_argument("--owner", default="microduck_ballkick_flat/mujoco")
    parser.add_argument("--foot", choices=("right", "left"), default="right")
    parser.add_argument("--policy-s", type=float, default=None, help="default: the whole --duration")
    parser.add_argument("--duration", type=float, default=2.0)
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
