"""Run the onboard runtime closed-loop in simulation.

The environment is only the plant: every action comes from
``microduck_rl_unilab.deploy.Controller`` running the exported ONNX from raw
readings (gyro, gravity direction, absolute joint angles and velocities,
heading, position), the way it would on the robot. Each step checks

* the runtime's 61-D observation against the environment's policy
  observation with noise, delay, IMU misalignment, and encoder bias off,
  per segment, and
* the runtime's clamped goal positions against the environment's own clamped
  BAM targets,

and each scenario reports survival, forward speed, and yaw rate. Scenarios:
stand, straight 0.6 m/s with the heading hold, and the 0.5 m radius
turn-around in both directions.

Usage:
    uv run --no-sync scripts/deploy_sim_loop.py examples/sprint_stand/deploy
    uv run --no-sync scripts/deploy_sim_loop.py examples/sprint_stand/deploy --sim motrix
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
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from unilab.base import registry
from omegaconf import open_dict
from unilab.base.config_adapter import BackendAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sprint_speed import REPO_ROOT, _compose_owner  # noqa: E402

from microduck_rl_unilab.deploy import Command, Controller, SensorReading  # noqa: E402

SEGMENTS = (
    ("gyro", 0, 3),
    ("gravity", 3, 6),
    ("joint_pos", 6, 20),
    ("joint_vel", 20, 34),
    ("last_action", 34, 48),
    ("twist_cmd", 48, 51),
    ("pose_cmds", 51, 61),
)
SCENARIOS: dict[str, tuple[Command, float, float]] = {
    # name: (runtime command, env forward command, env yaw command)
    "stand": (Command.stand(), 0.0, 0.0),
    "straight_0.6": (Command.straight(0.6), 0.6, 0.0),
    "turn_around_left": (Command.turn_around(1), 0.2, 0.4),
    "turn_around_right": (Command.turn_around(-1), 0.2, -0.4),
}


def _run_scenario(
    deploy_dir: Path,
    name: str,
    *,
    sim: str,
    num_envs: int,
    duration_s: float,
    seed: int,
) -> dict[str, Any]:
    command, env_speed, env_yaw = SCENARIOS[name]
    cfg = _compose_owner(
        speed=env_speed,
        duration_s=duration_s,
        seed=seed,
        owner=f"microduck_sprint_straightspeed_flat/{sim}",
        clip_targets=True,
        yaw_rate=env_yaw,
    )
    with open_dict(cfg):
        # The reference is the policy observation itself with training
        # corruption off: no noise, delay, IMU misalignment, or encoder bias,
        # and the body-pose command at the zero the runtime sends.
        policy_obs = cfg.env.observations.policy
        policy_obs.enable_corruption = False
        for term in policy_obs.terms.values():
            if term is None:
                continue
            term.delay_min_lag = 0
            term.delay_max_lag = 0
            if "max_angle_deg" in (term.get("params") or {}):
                term.params.max_angle_deg = 0.0
            if "biased" in (term.get("params") or {}):
                term.params.biased = False
        cfg.env.commands.body_pose.ranges = [[0.0, 0.0]] * 6
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    raw_env = cast(
        Any,
        registry.make(
            "MicroduckSprintFlat",
            sim_backend=str(cfg.training.sim_backend),
            num_envs=num_envs,
            env_cfg_override=override,
        ),
    )
    env = RslRlVecEnvWrapper(raw_env, device="cpu")
    try:
        robot = raw_env.scene["robot"]
        action_term = raw_env.action_manager.get_term("joint_pos")
        controllers = [Controller(deploy_dir) for _ in range(num_envs)]
        names = controllers[0].manifest.joint_names
        joint_ids = np.asarray([list(robot.joint_names).index(n) for n in names])
        action_names = [robot.joint_names[i] for i in np.asarray(action_term.joint_ids)]
        if action_names != list(names):
            raise RuntimeError(f"action order {action_names} differs from the manifest {names}")

        obs = env.get_observations()
        data = robot.data
        for i, controller in enumerate(controllers):
            controller.set_command(
                command, float(data.heading_w[i]), np.asarray(data.root_link_pos_w[i, :2])
            )

        steps = round(duration_s / float(raw_env.cfg.ctrl_dt))
        alive = np.ones(num_envs, dtype=np.bool_)
        obs_error = {segment: 0.0 for segment, _, _ in SEGMENTS}
        target_error = 0.0
        speed_sum = yaw_sum = 0.0
        samples = 0
        for step in range(steps):
            reference = obs["policy"].cpu().numpy()
            targets = np.empty((num_envs, len(names)), dtype=np.float32)
            actions = np.empty_like(targets)
            for i, controller in enumerate(controllers):
                reading = SensorReading(
                    gyro_b=np.asarray(data.root_link_ang_vel_b[i]),
                    gravity_b=np.asarray(data.projected_gravity_b[i]),
                    joint_pos=np.asarray(data.joint_pos[i, joint_ids]),
                    joint_vel=np.asarray(data.joint_vel[i, joint_ids]),
                )
                targets[i] = controller.step(
                    reading, float(data.heading_w[i]), np.asarray(data.root_link_pos_w[i, :2])
                )
                actions[i] = controller.observations.last_action
                if alive[i] and step > 0:
                    built = controller.last_obs
                    for segment, lo, hi in SEGMENTS:
                        if segment == "twist_cmd" and env_yaw == 0.0 and not command.hold_heading:
                            # Standing sends a zero yaw; the env still runs its heading loop.
                            continue
                        obs_error[segment] = max(
                            obs_error[segment], float(np.max(np.abs(built[lo:hi] - reference[i, lo:hi])))
                        )
            obs, _, dones, infos = env.step(torch.from_numpy(actions))
            applied = np.asarray(action_term.processed_action)
            target_error = max(
                target_error, float(np.max(np.abs(applied[alive] - targets[alive]), initial=0.0))
            )
            terminated = dones.cpu().numpy().astype(bool)
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                terminated &= ~time_outs.cpu().numpy().astype(bool)
            gravity = np.asarray(data.projected_gravity_b)
            alive &= ~terminated & (np.arccos(np.clip(-gravity[:, 2], -1, 1)) <= math.radians(70))
            for i in np.flatnonzero(terminated):
                controllers[i].reset()
                controllers[i].set_command(
                    command, float(data.heading_w[i]), np.asarray(data.root_link_pos_w[i, :2])
                )
            if step * float(raw_env.cfg.ctrl_dt) >= 1.0 and np.any(alive):
                speed_sum += float(np.sum(np.asarray(data.root_link_lin_vel_b)[alive, 0]))
                yaw_sum += float(np.sum(np.asarray(data.root_link_ang_vel_w)[alive, 2]))
                samples += int(np.sum(alive))
        return {
            "scenario": name,
            "command_m_s": command.forward_m_s,
            "command_yaw_rate_rad_s": command.yaw_rate_rad_s,
            "hold_heading": command.hold_heading,
            "survival_fraction": float(np.mean(alive)),
            "forward_speed_m_s": speed_sum / samples if samples else None,
            "yaw_rate_rad_s": yaw_sum / samples if samples else None,
            "max_obs_error": obs_error,
            "max_target_error_rad": target_error,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("deploy_dir", type=Path)
    parser.add_argument("--sim", choices=("mujoco", "motrix"), default="mujoco")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-obs-error", type=float, default=1e-4)
    parser.add_argument("--min-survival", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = [
        _run_scenario(
            args.deploy_dir,
            name,
            sim=args.sim,
            num_envs=args.num_envs,
            duration_s=args.duration,
            seed=args.seed,
        )
        for name in args.scenarios.split(",")
    ]
    passes = True
    for row in results:
        worst_obs = max(row["max_obs_error"].values())
        ok = (
            worst_obs <= args.max_obs_error
            and row["max_target_error_rad"] <= args.max_obs_error
            and row["survival_fraction"] >= args.min_survival
        )
        passes &= ok
        row["passes"] = ok
        print(
            f"{row['scenario']:18s} survival={row['survival_fraction']:.3f} "
            f"forward={row['forward_speed_m_s']:.3f} yaw={row['yaw_rate_rad_s']:+.3f} "
            f"obs_err={worst_obs:.2e} target_err={row['max_target_error_rad']:.2e} "
            f"[{'pass' if ok else 'FAIL'}]"
        )
    summary = {"deploy_dir": str(args.deploy_dir), "sim": args.sim, "passes": passes, "results": results}
    output = args.output or args.deploy_dir / f"sim_loop_{args.sim}.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"deploy sim loop {'PASS' if passes else 'FAIL'}: {output}")
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
