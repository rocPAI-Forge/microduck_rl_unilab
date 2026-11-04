"""Export a MicroDuck checkpoint for deployment and check the export.

Writes ``policy.onnx`` (weights inline, one file), ``policy.pt`` (TorchScript),
and ``deploy_manifest.json``. The manifest records everything an onboard
runtime has to reproduce: observation segment order, action joint order,
default joint positions, action scale, the goal-position clamp, the control
period, and the observation noise and delay the policy was trained with.

The check rolls the policy out in simulation, feeds the recorded
observations to both the PyTorch inference policy and the ONNX file through
onnxruntime, and reports the largest action difference.

Usage:
    uv run --no-sync scripts/export_policy.py examples/sprint_stand/model_59000.pt \\
      --owner microduck_sprint_standturn_flat/mujoco --clip-targets \\
      --output-dir results/export_sprint_stand
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root
from unilab.training import algo_config_dict

from microduck_rl_unilab.conf_searchpath import register_conf_search_path
from microduck_rl_unilab.tasks.microduck.deploy_contract import (
    MICRODUCK_ACTOR_OBS_DIM,
    MICRODUCK_NUM_ACTION,
    MICRODUCK_OBS_SEGMENTS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"


def _compose(owner: str, seed: int, clip_targets: bool) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        if clip_targets:
            cfg.env.actions.joint_pos.clip_target_to_joint_limits = True
    return cfg


def _observation_noise(cfg: Any) -> dict[str, Any]:
    terms = OmegaConf.to_container(cfg.env.observations.policy.terms, resolve=True)
    report = {}
    for name, term in cast(dict, terms).items():
        if term is None:
            continue
        noise = term.get("noise") or {}
        report[name] = {
            "noise_uniform": [noise.get("n_min"), noise.get("n_max")] if noise else None,
            "delay_steps": [term.get("delay_min_lag", 0), term.get("delay_max_lag", 0)],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-targets", action="store_true")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    cfg = _compose(args.owner, args.seed, args.clip_targets)
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
    wrapped = RslRlVecEnvWrapper(raw_env, device=args.device)
    try:
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        if isinstance(train_cfg.get("algorithm"), dict):
            train_cfg["algorithm"]["enable_compile"] = False
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=args.device)
        runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        runner.export_policy_to_onnx(str(out), filename="policy_external.onnx")
        model = onnx.load(str(out / "policy_external.onnx"), load_external_data=True)
        onnx.save_model(model, str(out / "policy.onnx"), save_as_external_data=False)
        (out / "policy_external.onnx").unlink()
        data_file = out / "policy_external.onnx.data"
        if data_file.exists():
            data_file.unlink()
        onnx.checker.check_model(str(out / "policy.onnx"))
        runner.export_policy_to_jit(str(out), filename="policy.pt")

        session = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        obs = wrapped.get_observations()
        max_diff = 0.0
        samples = 0
        for _ in range(args.steps):
            flat = obs["policy"] if not isinstance(obs, torch.Tensor) else obs
            with torch.inference_mode():
                action = policy(obs)
            rows = flat.detach().cpu().numpy().astype(np.float32)
            torch_rows = action.detach().cpu().numpy()
            # The exported graph takes one observation at a time, as onboard.
            for row, torch_row in zip(rows, torch_rows):
                onnx_row = session.run(None, {input_name: row[None, :]})[0][0]
                max_diff = max(max_diff, float(np.max(np.abs(onnx_row - torch_row))))
            samples += rows.shape[0]
            with torch.inference_mode():
                obs, _, _, _ = wrapped.step(action)

        robot = raw_env.scene["robot"]
        term = raw_env.action_manager.get_term("joint_pos")
        joint_ids = np.asarray(term.joint_ids)
        names = [robot.joint_names[int(i)] for i in joint_ids]
        defaults = np.asarray(robot.data.default_joint_pos)[0, joint_ids]
        limits = np.asarray(robot.data.soft_joint_pos_limits)[joint_ids]
        manifest = {
            "checkpoint": str(args.checkpoint),
            "owner": args.owner,
            "onnx": "policy.onnx",
            "onnx_input": {"name": input_name, "shape": [str(d) for d in input_shape]},
            "observation_dim": MICRODUCK_ACTOR_OBS_DIM,
            "observation_segments": [{"name": n, "dim": d} for n, d in MICRODUCK_OBS_SEGMENTS],
            "observation_training_noise_and_delay": _observation_noise(cfg),
            "action_dim": MICRODUCK_NUM_ACTION,
            "action_joint_order": names,
            "action_to_target": "q_des = action * scale + default_joint_pos",
            "action_scale": float(cfg.env.actions.joint_pos.scale),
            "default_joint_pos_rad": {n: round(float(v), 6) for n, v in zip(names, defaults)},
            "goal_position_clamp": bool(args.clip_targets),
            "soft_joint_limits_rad": {
                n: [round(float(lo), 6), round(float(hi), 6)] for n, (lo, hi) in zip(names, limits)
            },
            "control_period_s": float(raw_env.cfg.ctrl_dt),
            "export_check": {
                "samples": samples,
                "max_abs_action_difference": max_diff,
            },
        }
        (out / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest["export_check"]), flush=True)
        print(f"wrote {out}", flush=True)
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
