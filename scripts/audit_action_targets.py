"""Audit how far a checkpoint's BAM position targets go past the joint limits.

The BAM action does not clamp ``q_des``, so a policy can hold a joint on its
stop by commanding a target far past it, which keeps the servo at full
voltage. A real servo clamps its goal position. This runs a checkpoint under
any task owner, reports the requested-target overshoot per joint, and with
``--clip-targets`` reruns the same rollout with the clamp on so the effect on
the task can be compared.

Usage:
    uv run --no-sync scripts/audit_action_targets.py \\
      examples/standup/model_14999.pt --owner microduck_standup_flat/mujoco \\
      --override env.events.set_ground_state.params.face_down_prob=1 ...
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
UPRIGHT_TILT_DEG = 20.0
OVERSHOOT_FLAG_RAD = 0.1


def _compose(owner: str, overrides: list[str], seed: int, clip_targets: bool) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}", *overrides])
    with open_dict(cfg):
        cfg.algo.seed = seed
        if clip_targets:
            cfg.env.actions.joint_pos.clip_target_to_joint_limits = True
    return cfg


def audit(
    checkpoint: Path,
    *,
    owner: str,
    overrides: list[str],
    num_envs: int,
    duration_s: float,
    seed: int,
    clip_targets: bool,
    device: str,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    cfg = _compose(owner, overrides, seed, clip_targets)
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    raw_env = cast(
        Any,
        registry.make(
            str(cfg.training.task_name),
            sim_backend=str(cfg.training.sim_backend),
            num_envs=num_envs,
            env_cfg_override=override,
        ),
    )
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        wrapped = RslRlVecEnvWrapper(raw_env, device=device)
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        if isinstance(train_cfg.get("algorithm"), dict):
            train_cfg["algorithm"]["enable_compile"] = False
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)

        robot = raw_env.scene["robot"]
        term = raw_env.action_manager.get_term("joint_pos")
        joint_ids = np.asarray(term.joint_ids)
        names = [robot.joint_names[int(i)] for i in joint_ids]
        limits = np.asarray(robot.data.soft_joint_pos_limits)[joint_ids]
        steps = round(duration_s / float(raw_env.cfg.ctrl_dt))

        overshoot_sum = np.zeros(len(names), dtype=np.float64)
        flagged_steps = np.zeros(len(names), dtype=np.int64)
        terminations = 0
        obs = wrapped.get_observations()
        for _ in range(steps):
            with torch.inference_mode():
                obs, _, dones, infos = wrapped.step(policy(obs))
            request = np.asarray(term.requested_target)
            overshoot = np.maximum(limits[None, :, 0] - request, 0.0) + np.maximum(
                request - limits[None, :, 1], 0.0
            )
            overshoot_sum += overshoot.sum(axis=0)
            flagged_steps += (overshoot > OVERSHOOT_FLAG_RAD).sum(axis=0)
            done = dones.detach().cpu().numpy().astype(bool)
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                done &= ~time_outs.detach().cpu().numpy().astype(bool)
            terminations += int(done.sum())

        gravity = np.asarray(robot.data.projected_gravity_b)
        tilt = np.degrees(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
        samples = steps * num_envs
        per_joint = {
            name: {
                "mean_overshoot_rad": round(float(overshoot_sum[i] / samples), 4),
                "time_past_0p1_rad": round(float(flagged_steps[i] / samples), 4),
            }
            for i, name in enumerate(names)
        }
        return {
            "checkpoint": str(checkpoint),
            "owner": owner,
            "overrides": overrides,
            "clip_targets": clip_targets,
            "num_envs": num_envs,
            "duration_s": duration_s,
            "seed": seed,
            "summed_mean_overshoot_rad": round(float(overshoot_sum.sum() / samples), 4),
            "per_joint": per_joint,
            "terminations": terminations,
            "final_upright_fraction": round(float(np.mean(tilt < UPRIGHT_TILT_DEG)), 4),
            "final_mean_tilt_deg": round(float(np.mean(tilt)), 2),
        }
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            raw_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", required=True, help="Hydra task owner, e.g. microduck_standup_flat/mujoco")
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-targets", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.num_envs <= 0:
        raise SystemExit("duration and num-envs must be positive")
    result = audit(
        args.checkpoint,
        owner=args.owner,
        overrides=args.override,
        num_envs=args.num_envs,
        duration_s=args.duration,
        seed=args.seed,
        clip_targets=args.clip_targets,
        device=args.device,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
