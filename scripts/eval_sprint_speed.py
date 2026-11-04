"""Evaluate a Sprint checkpoint with the published Running battery.

The primary comparison uses a 2.20 m/s command, one second of warm-up, and a
ten-second measurement window. Speed samples stop accumulating for an
environment after its trunk exceeds the configured fall angle.

Usage:
    uv run --no-sync scripts/eval_sprint_speed.py model_7499.pt
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
from unilab.utils.rotation import np_quat_apply, np_quat_apply_inverse, np_wrap_to_pi

from microduck_rl_unilab.conf_searchpath import register_conf_search_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"
# Head body plus its local "top of the head" axis, measured at the home keyframe.
HEAD_BODY = "jaw_soft"
TRUNK_BODY = "trunk_base"
HEAD_UP_AXIS_B = np.array([[1.0, 0.0, 0.0]])


def _compose_owner(
    *,
    speed: float,
    duration_s: float,
    seed: int,
    stress: bool = False,
    owner: str | None = None,
    heading_stiffness: float | None = None,
    heading_yaw_limit: float | None = None,
    cross_track_stiffness: float | None = None,
    clip_targets: bool = False,
    yaw_rate: float = 0.0,
) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    selected_owner = owner or (
        "microduck_sprint_robust_flat/mujoco"
        if stress
        else "microduck_sprint_flat/mujoco"
    )
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={selected_owner}"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.max_episode_seconds = duration_s + 1.0
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [speed, speed]
        twist.ranges.lin_vel_y = [0.0, 0.0]
        twist.ranges.ang_vel_z = [yaw_rate, yaw_rate]
        if yaw_rate != 0.0:
            # The spawn-heading loop writes the yaw command itself; a fixed
            # turn command needs it off.
            twist.hold_spawn_heading = False
        if heading_stiffness is not None:
            twist.spawn_heading_stiffness = heading_stiffness
        if heading_yaw_limit is not None:
            twist.spawn_heading_yaw_rate_limit = heading_yaw_limit
        if cross_track_stiffness is not None:
            twist.spawn_cross_track_stiffness = cross_track_stiffness
        twist.rel_standing_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.turn_in_place_fraction = 0.0
        twist.resampling_time_range = [duration_s + 1.0, duration_s + 1.0]
        if clip_targets:
            cfg.env.actions.joint_pos.clip_target_to_joint_limits = True
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        # The battery reads kinematics directly. Keeping tilt/nan terminations
        # recycles unsafe rows, while disabling rewards prevents dead rows from
        # tripping finite checks before their reset is applied.
        for name in cfg.reward:
            cfg.reward[name] = None
        if stress:
            # Held-out stress from the playground release record: wider trunk
            # CoM than the last training stage, plus the stage-C push/tilt.
            tilt = math.radians(2.0)
            cfg.env.events.push_robot.params.velocity_range.x = [-0.10, 0.10]
            cfg.env.events.push_robot.params.velocity_range.y = [-0.10, 0.10]
            cfg.env.events.base_com.params.com_range.x = [-0.010, 0.010]
            cfg.env.events.base_com.params.com_range.y = [-0.010, 0.010]
            cfg.env.events.base_com.params.com_range.z = [-0.010, 0.010]
            cfg.env.events.head_com.params.com_range.x = [-0.006, 0.006]
            cfg.env.events.head_com.params.com_range.y = [-0.006, 0.006]
            cfg.env.events.head_com.params.com_range.z = [-0.006, 0.006]
            cfg.env.events.reset_base.params.pose_range.roll = [-tilt, tilt]
            cfg.env.events.reset_base.params.pose_range.pitch = [-tilt, tilt]
    return cfg


def _quantile(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values.size else None


def evaluate(
    checkpoint: Path,
    *,
    speed: float = 2.2,
    num_envs: int = 512,
    duration_s: float = 10.0,
    warmup_s: float = 1.0,
    fall_tilt_deg: float = 70.0,
    seed: int = 123,
    device: str | None = None,
    stress: bool = False,
    owner: str | None = None,
    heading_stiffness: float | None = None,
    heading_yaw_limit: float | None = None,
    cross_track_stiffness: float | None = None,
    clip_targets: bool = False,
    yaw_rate: float = 0.0,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if speed < 0.0:
        raise ValueError("speed must be nonnegative")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= warmup_s < duration_s:
        raise ValueError("warmup_s must be in [0, duration_s)")
    if not 0.0 < fall_tilt_deg <= 180.0:
        raise ValueError("fall_tilt_deg must be in (0, 180]")

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _compose_owner(
        speed=speed,
        duration_s=warmup_s + duration_s,
        seed=seed,
        stress=stress,
        owner=owner,
        heading_stiffness=heading_stiffness,
        heading_yaw_limit=heading_yaw_limit,
        cross_track_stiffness=cross_track_stiffness,
        clip_targets=clip_targets,
        yaw_rate=yaw_rate,
    )
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(
        cfg,
        root_dir=REPO_ROOT,
        algo_name="ppo",
    ).build_task_env_cfg_override()
    raw_env = cast(
        Any,
        registry.make(
            "MicroduckSprintFlat",
            sim_backend=str(cfg.training.sim_backend),
            num_envs=num_envs,
            env_cfg_override=override,
        ),
    )
    wrapped_env: RslRlVecEnvWrapper | None = None
    try:
        wrapped_env = RslRlVecEnvWrapper(raw_env, device=device)
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        algorithm_cfg = train_cfg.get("algorithm")
        if isinstance(algorithm_cfg, dict):
            algorithm_cfg["enable_compile"] = False
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        obs = wrapped_env.get_observations()

        robot = raw_env.scene["robot"]
        ctrl_dt = float(raw_env.cfg.ctrl_dt)
        total_steps = round((warmup_s + duration_s) / ctrl_dt)
        warmup_steps = round(warmup_s / ctrl_dt)
        alive = np.ones(num_envs, dtype=np.bool_)
        speed_sum = np.zeros(num_envs, dtype=np.float64)
        lateral_sum = np.zeros(num_envs, dtype=np.float64)
        yaw_rate_sum = np.zeros(num_envs, dtype=np.float64)
        heading_error_sum = np.zeros(num_envs, dtype=np.float64)
        head_error_sum = np.zeros(num_envs, dtype=np.float64)
        head_yaw_sum = np.zeros(num_envs, dtype=np.float64)
        head_roll_sum = np.zeros(num_envs, dtype=np.float64)
        head_pitch_delta_sum = np.zeros(num_envs, dtype=np.float64)
        head_velocity_sum = np.zeros(num_envs, dtype=np.float64)
        head_tilt_sum = np.zeros(num_envs, dtype=np.float64)
        head_inverted_sum = np.zeros(num_envs, dtype=np.float64)
        sample_count = np.zeros(num_envs, dtype=np.int64)
        heading_ref = np.asarray(robot.data.heading_w).copy()
        origin_xy = np.asarray(robot.data.root_link_pos_w)[:, :2].copy()
        forward_axis = np.stack(
            [np.cos(heading_ref), np.sin(heading_ref)], axis=1
        )
        lateral_axis = np.stack(
            [-np.sin(heading_ref), np.cos(heading_ref)], axis=1
        )
        straight_pose_alive = np.ones(num_envs, dtype=np.bool_)
        straight_pose_steps = np.zeros(num_envs, dtype=np.int64)
        alive_steps = np.zeros(num_envs, dtype=np.int64)
        break_fall = np.zeros(num_envs, dtype=np.bool_)
        break_heading = np.zeros(num_envs, dtype=np.bool_)
        break_head = np.zeros(num_envs, dtype=np.bool_)
        final_heading_error = np.zeros(num_envs, dtype=np.float64)
        final_cross_track = np.zeros(num_envs, dtype=np.float64)
        final_forward_displacement = np.zeros(num_envs, dtype=np.float64)
        head_ids, _ = robot.find_joints(
            ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
            preserve_order=True,
        )
        head_body_ids, _ = robot.find_bodies([HEAD_BODY, TRUNK_BODY], preserve_order=True)
        head_body_id, trunk_body_id = int(head_body_ids[0]), int(head_body_ids[1])
        neck_pitch_sum = np.zeros(num_envs, dtype=np.float64)
        head_forward_sum = np.zeros(num_envs, dtype=np.float64)
        action_term = raw_env.action_manager.get_term("joint_pos")
        action_joint_ids = np.asarray(action_term.joint_ids)
        action_limits = np.asarray(robot.data.soft_joint_pos_limits)[action_joint_ids]
        overshoot_sum = np.zeros(num_envs, dtype=np.float64)
        overshoot_joint_sum = np.zeros(action_joint_ids.size, dtype=np.float64)
        overshoot_samples = 0

        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = wrapped_env.step(actions)
            terminated = dones
            time_outs = infos.get("time_outs")
            if isinstance(time_outs, torch.Tensor):
                terminated = dones & ~time_outs
            alive &= ~terminated.detach().cpu().numpy()
            gravity = np.asarray(robot.data.projected_gravity_b)
            tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
            alive &= tilt <= math.radians(fall_tilt_deg)
            alive_steps[alive] += 1
            if step < warmup_steps:
                continue
            velocity = np.nan_to_num(np.asarray(robot.data.root_link_lin_vel_b))
            heading_error = np.abs(np_wrap_to_pi(np.asarray(robot.data.heading_w) - heading_ref))
            head_delta = (
                np.asarray(robot.data.joint_pos)[:, head_ids]
                - np.asarray(robot.data.default_joint_pos)[:, head_ids]
            )
            head_error = np.mean(np.abs(head_delta), axis=1)
            head_velocity = np.sqrt(
                np.mean(np.square(np.asarray(robot.data.joint_vel)[:, head_ids]), axis=1)
            )
            head_quat = np.asarray(robot.data.body_link_quat_w)[:, head_body_id, :]
            head_up_w = np_quat_apply(head_quat.astype(np.float64), HEAD_UP_AXIS_B)
            head_up_cos = np.nan_to_num(head_up_w[:, 2], nan=-1.0)
            head_tilt = np.arccos(np.clip(head_up_cos, -1.0, 1.0))
            body_pos = np.asarray(robot.data.body_link_pos_w)
            trunk_quat = np.asarray(robot.data.body_link_quat_w)[:, trunk_body_id, :]
            head_in_trunk = np_quat_apply_inverse(
                trunk_quat.astype(np.float64),
                (body_pos[:, head_body_id] - body_pos[:, trunk_body_id]).astype(np.float64),
            )
            displacement = (
                np.asarray(robot.data.root_link_pos_w)[:, :2] - origin_xy
            )
            cross_track = np.abs(np.sum(displacement * lateral_axis, axis=1))
            forward_displacement = np.sum(displacement * forward_axis, axis=1)
            straight_pose_now = (
                alive
                & (heading_error <= math.radians(25.0))
                & (head_up_cos >= 0.0)
            )
            breaking = straight_pose_alive & ~straight_pose_now
            break_fall[breaking & ~alive] = True
            break_heading[
                breaking & alive & (heading_error > math.radians(25.0))
            ] = True
            break_head[breaking & alive & (head_up_cos < 0.0)] = True
            straight_pose_alive &= straight_pose_now
            straight_pose_steps[straight_pose_alive] += 1
            final_heading_error[alive] = heading_error[alive]
            final_cross_track[alive] = cross_track[alive]
            final_forward_displacement[alive] = forward_displacement[alive]
            speed_sum[alive] += velocity[alive, 0]
            lateral_sum[alive] += np.abs(velocity[alive, 1])
            yaw_rate_sum[alive] += np.asarray(robot.data.root_link_ang_vel_w)[alive, 2]
            heading_error_sum[alive] += heading_error[alive]
            head_error_sum[alive] += head_error[alive]
            head_yaw_sum[alive] += np.abs(head_delta[alive, 2])
            head_roll_sum[alive] += np.abs(head_delta[alive, 3])
            head_pitch_delta_sum[alive] += np.mean(head_delta[alive, :2], axis=1)
            head_velocity_sum[alive] += head_velocity[alive]
            head_tilt_sum[alive] += head_tilt[alive]
            head_inverted_sum[alive] += (head_up_cos[alive] < 0.0).astype(np.float64)
            neck_pitch_sum[alive] += np.asarray(robot.data.joint_pos)[alive, head_ids[0]]
            head_forward_sum[alive] += head_in_trunk[alive, 0]
            target = np.asarray(
                getattr(action_term, "requested_target", action_term.processed_action)
            )
            overshoot = np.maximum(action_limits[None, :, 0] - target, 0.0) + np.maximum(
                target - action_limits[None, :, 1], 0.0
            )
            overshoot_sum[alive] += np.sum(overshoot[alive], axis=1)
            if np.any(alive):
                overshoot_joint_sum += np.sum(overshoot[alive], axis=0)
                overshoot_samples += int(np.sum(alive))
            sample_count[alive] += 1

        sampled = sample_count > 0
        mean_speed = speed_sum[sampled] / sample_count[sampled]
        clean = alive & sampled
        return {
            "schema_version": 1,
            "task": "MicroduckSprintFlat",
            "checkpoint": str(checkpoint.resolve()),
            "backend": str(cfg.training.sim_backend),
            "device": device,
            "seed": seed,
            "num_envs": num_envs,
            "command_forward_m_s": speed,
            "warmup_s": warmup_s,
            "duration_s": duration_s,
            "fall_tilt_deg": fall_tilt_deg,
            "stress": stress,
            "owner": owner or (
                "microduck_sprint_robust_flat/mujoco"
                if stress
                else "microduck_sprint_flat/mujoco"
            ),
            "survival_fraction": float(np.mean(alive)),
            # Rollout time (warm-up included) at which each fallen row went down.
            "fall_times_s": sorted(
                float(steps * ctrl_dt) for steps in alive_steps[~alive]
            ),
            "body_forward_speed_m_s": {
                "mean": float(np.mean(mean_speed)) if mean_speed.size else None,
                "p10": _quantile(mean_speed, 0.10),
                "median": _quantile(mean_speed, 0.50),
                "p90": _quantile(mean_speed, 0.90),
                "clean_survivor_mean": (
                    float(np.mean(speed_sum[clean] / sample_count[clean]))
                    if np.any(clean)
                    else None
                ),
            },
            "command_yaw_rate_rad_s": yaw_rate,
            "yaw_rate_rad_s": (
                float(np.mean(yaw_rate_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            "mean_absolute_lateral_speed_m_s": (
                float(np.mean(lateral_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            "mean_absolute_heading_error_deg": (
                math.degrees(
                    float(np.mean(heading_error_sum[sampled] / sample_count[sampled]))
                )
                if np.any(sampled)
                else None
            ),
            "final_absolute_heading_error_deg": {
                "mean": (
                    math.degrees(float(np.mean(final_heading_error[sampled])))
                    if np.any(sampled)
                    else None
                ),
                "p90": (
                    math.degrees(float(np.quantile(final_heading_error[sampled], 0.90)))
                    if np.any(sampled)
                    else None
                ),
            },
            "final_cross_track_error_m": {
                "mean": (
                    float(np.mean(final_cross_track[sampled]))
                    if np.any(sampled)
                    else None
                ),
                "p90": (
                    float(np.quantile(final_cross_track[sampled], 0.90))
                    if np.any(sampled)
                    else None
                ),
            },
            "final_spawn_forward_displacement_m": {
                "mean": (
                    float(np.mean(final_forward_displacement[sampled]))
                    if np.any(sampled)
                    else None
                ),
                "p10": (
                    float(np.quantile(final_forward_displacement[sampled], 0.10))
                    if np.any(sampled)
                    else None
                ),
            },
            # Sticky gate: once an environment falls, inverts its head, or
            # exceeds 25 degrees from its spawn ray, its straight-pose horizon
            # ends even if a later correction returns it inside the band.
            "straight_pose_survival_fraction": float(
                np.mean(straight_pose_alive)
            ),
            # Why each row first left the straight pose. Heading and head can
            # break on the same step, so the counts may overlap.
            "straight_pose_break_counts": {
                "fall": int(np.sum(break_fall)),
                "heading": int(np.sum(break_heading)),
                "head_inverted": int(np.sum(break_head)),
            },
            "straight_pose_horizon_s": {
                "mean": float(np.mean(straight_pose_steps) * ctrl_dt),
                "p10": float(np.quantile(straight_pose_steps, 0.10) * ctrl_dt),
                "median": float(np.median(straight_pose_steps) * ctrl_dt),
            },
            "head_home_error_deg": (
                math.degrees(float(np.mean(head_error_sum[sampled] / sample_count[sampled])))
                if np.any(sampled)
                else None
            ),
            "head_yaw_abs_deg": (
                math.degrees(float(np.mean(head_yaw_sum[sampled] / sample_count[sampled])))
                if np.any(sampled)
                else None
            ),
            "head_roll_abs_deg": (
                math.degrees(float(np.mean(head_roll_sum[sampled] / sample_count[sampled])))
                if np.any(sampled)
                else None
            ),
            "head_forward_pitch_delta_deg": (
                math.degrees(
                    float(np.mean(head_pitch_delta_sum[sampled] / sample_count[sampled]))
                )
                if np.any(sampled)
                else None
            ),
            "head_joint_velocity_rms_rad_s": (
                float(np.mean(head_velocity_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            # Angle between the top of the head and world up, plus the share of
            # sampled time the head is past horizontal (i.e. upside down).
            "head_up_tilt_deg": (
                math.degrees(float(np.mean(head_tilt_sum[sampled] / sample_count[sampled])))
                if np.any(sampled)
                else None
            ),
            "head_inverted_time_fraction": (
                float(np.mean(head_inverted_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            # Absolute neck_pitch (home +20 deg, lower stop -90 deg) and how far
            # the head body sits ahead of trunk_base along the trunk x axis
            # (about -0.01 m at home).
            "neck_pitch_deg": (
                math.degrees(float(np.mean(neck_pitch_sum[sampled] / sample_count[sampled])))
                if np.any(sampled)
                else None
            ),
            "head_forward_in_trunk_m": (
                float(np.mean(head_forward_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            # Position targets past the soft joint limits, summed over joints
            # (radians per step), and the per-joint mean behind that total.
            "action_target_overshoot_rad": (
                float(np.mean(overshoot_sum[sampled] / sample_count[sampled]))
                if np.any(sampled)
                else None
            ),
            "action_target_overshoot_by_joint_rad": (
                dict(
                    zip(
                        [robot.joint_names[int(i)] for i in action_joint_ids],
                        (overshoot_joint_sum / overshoot_samples).round(3).tolist(),
                    )
                )
                if overshoot_samples
                else None
            ),
        }
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        else:
            raw_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=2.2)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=1.0)
    parser.add_argument("--fall-tilt-deg", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument(
        "--owner",
        default=None,
        help="Hydra task owner used to build the evaluation environment.",
    )
    parser.add_argument("--heading-stiffness", type=float, default=None)
    parser.add_argument("--heading-yaw-limit", type=float, default=None)
    parser.add_argument("--cross-track-stiffness", type=float, default=None)
    parser.add_argument(
        "--clip-targets",
        action="store_true",
        help="Clamp BAM position targets to the joint limits (real-servo behavior).",
    )
    parser.add_argument(
        "--yaw-rate",
        type=float,
        default=0.0,
        help="Fixed yaw-rate command, rad/s. Nonzero turns the spawn-heading loop off.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(
        args.checkpoint,
        speed=args.speed,
        num_envs=args.num_envs,
        duration_s=args.duration,
        warmup_s=args.warmup,
        fall_tilt_deg=args.fall_tilt_deg,
        seed=args.seed,
        device=args.device,
        stress=args.stress,
        owner=args.owner,
        heading_stiffness=args.heading_stiffness,
        heading_yaw_limit=args.heading_yaw_limit,
        cross_track_stiffness=args.cross_track_stiffness,
        clip_targets=args.clip_targets,
        yaw_rate=args.yaw_rate,
    )
    result["clip_targets"] = args.clip_targets
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
