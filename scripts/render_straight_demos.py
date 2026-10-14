"""Render straight-sprint demo clips from the long-horizon checkpoint.

Two clips, both using ``microduck_sprint_straightfix_flat`` and a 2.2 m/s
forward command:

* a close rear-quarter tracking shot of one robot (world azimuth 45°, which
  sits 45° off the -x rear axis when the robot faces +x)
* a fixed far shot of eight robots leaving the origin on headings
  0°, 45°, …, 315°

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \\
      uv run --no-sync scripts/render_straight_demos.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

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
from unilab.utils.rotation import np_yaw_to_quat
from unilab.visualization import render_many
from unisim.backend.mujoco.playback import resolve_render_play_model_files
from unisim.backend.playback_common import write_playback_video

from microduck_rl_unilab.conf_searchpath import register_conf_search_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = package_root() / "conf" / "ppo"
DEFAULT_CHECKPOINT = REPO_ROOT / "examples/sprint_straight_long/model_19297.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "examples/sprint_straight_long"
OWNER = "microduck_sprint_straightfix_flat/mujoco"
SPEED_M_S = 2.2
# MuJoCo azimuth 45° places the camera on the rear-left diagonal when the
# robot's forward axis is +x. Elevation and distance are a close chase.
FOLLOW_AZIMUTH = 45.0
FOLLOW_ELEVATION = -14.0
FOLLOW_DISTANCE = 1.15


def _compose_owner(*, duration_s: float, seed: int) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={OWNER}"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.max_episode_seconds = duration_s + 5.0
        twist = cfg.env.commands.twist
        twist.ranges.lin_vel_x = [SPEED_M_S, SPEED_M_S]
        twist.ranges.lin_vel_y = [0.0, 0.0]
        twist.ranges.ang_vel_z = [0.0, 0.0]
        twist.hold_spawn_heading = True
        twist.rel_standing_envs = 0.0
        twist.rel_forward_envs = 0.0
        twist.turn_in_place_fraction = 0.0
        twist.resampling_time_range = [duration_s + 5.0, duration_s + 5.0]
        pose = cfg.env.events.reset_base.params.pose_range
        pose.x = [0.0, 0.0]
        pose.y = [0.0, 0.0]
        pose.z = [0.0, 0.0]
        pose.yaw = [0.0, 0.0]
        cfg.env.events.push_robot = None
    return cfg


def _make_env(cfg: Any, num_envs: int) -> Any:
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    return registry.make(
        "MicroduckSprintFlat",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )


def _place_robots(raw_env: Any, *, yaws: np.ndarray, xy: np.ndarray, speed: float) -> None:
    """Teleport roots onto the requested rays and retarget spawn-heading hold."""
    backend = raw_env._backend
    snap = np.asarray(raw_env.get_physics_state_snapshot(), dtype=np.float64)
    nq, nv = int(backend.nq), int(backend.nv)
    qpos = snap[:, 1 : 1 + nq].copy()
    qvel = snap[:, 1 + nq : 1 + nq + nv].copy()
    if qpos.shape[0] != yaws.shape[0]:
        raise RuntimeError(f"snapshot envs {qpos.shape[0]} != yaw count {yaws.shape[0]}")
    qpos[:, 0:2] = xy
    qpos[:, 3:7] = np_yaw_to_quat(yaws.astype(np.float64))
    qvel[:, :6] = 0.0
    env_ids = np.arange(qpos.shape[0], dtype=np.int32)
    backend.set_state(env_ids, qpos, qvel)

    twist = raw_env.command_manager.get_term("twist")
    twist._spawn_heading[:] = yaws
    twist._spawn_position_xy[:] = xy
    twist._capture_spawn_heading[:] = False
    with raw_env.scene._scoped_state_reads():
        raw_env.command_manager.compute(dt=0.0)
        raw_env.command_manager.post_compute()
        twist.vel_command_b[:, 0] = speed
        twist.vel_command_b[:, 1] = 0.0
        twist.vel_command_w[:, 0] = speed
        twist.vel_command_w[:, 1] = 0.0
        twist.is_standing_env[:] = False
        manager_obs = raw_env.observation_manager.compute(update_history=True)
    mapped = raw_env._map_observations(manager_obs, num_rows=raw_env.num_envs)
    for name, values in mapped.items():
        raw_env._state.obs[name][:] = values


def _rollout(
    raw_env: Any,
    wrapped: Any,
    policy: Any,
    *,
    yaws: np.ndarray,
    xy: np.ndarray,
    steps: int,
) -> list[np.ndarray]:
    obs = wrapped.get_observations()
    _place_robots(raw_env, yaws=yaws, xy=xy, speed=SPEED_M_S)
    obs = wrapped.get_observations()
    states = [np.asarray(raw_env.get_physics_state_snapshot(), dtype=np.float32).copy()]
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions)
        states.append(np.asarray(raw_env.get_physics_state_snapshot(), dtype=np.float32).copy())
    return states


def _subsample(states: list[np.ndarray], stride: int) -> list[np.ndarray]:
    kept = states[::stride]
    if kept[-1] is not states[-1]:
        kept.append(states[-1])
    return kept


def _travel(states: list[np.ndarray]) -> np.ndarray:
    xy0 = states[0][:, 1:3]
    xy1 = states[-1][:, 1:3]
    return np.linalg.norm(xy1 - xy0, axis=1)


def _render_follow(states: list[np.ndarray], model_file: str, output: Path, fps: int) -> None:
    frames = render_many.render_states_get_frames_tracking(
        states,
        model_file,
        width=1280,
        height=720,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=FOLLOW_DISTANCE,
        cam_elevation=FOLLOW_ELEVATION,
        cam_azimuth=FOLLOW_AZIMUTH,
        render_spacing=0.0,
    )
    if not frames:
        raise RuntimeError(f"follow render produced no frames for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_playback_video(str(output), frames, fps=fps)


def _far_camera(radius_m: float, *, low: bool = False) -> tuple[float, float, float, list[float]]:
    """Steep fixed camera framed on an 8.8 m ring (all eight headings visible).

    ``low`` keeps the same -75° overhead angle and moves the camera in from
    16 m to 13 m (about 15.5 m high to about 12 m), so each robot is larger.
    At that height the outer robots leave the 1280×720 frame after about 5.5 s.
    """
    base = 13.0 if low else 16.0
    distance = base * max(radius_m, 1.0) / 8.8
    return float(distance), -75.0, 25.0, [0.0, 0.0, 0.0]


def _render_radial(
    states: list[np.ndarray], model_file: str, output: Path, fps: int, *, low: bool = False
) -> None:
    radius = float(np.max(np.linalg.norm(states[-1][:, 1:3], axis=1)))
    distance, elevation, azimuth, lookat = _far_camera(radius, low=low)
    print(
        f"radial radius={radius:.2f} m camera distance={distance:.1f} "
        f"elevation={elevation:.0f} azimuth={azimuth:.0f}",
        flush=True,
    )
    frames = render_many.render_states_get_frames(
        states,
        model_file,
        width=1280,
        height=720,
        num_processes=1,
        cam_distance=distance,
        cam_elevation=elevation,
        cam_azimuth=azimuth,
        cam_lookat=lookat,
        render_spacing=0.0,
    )
    if not frames:
        raise RuntimeError(f"radial render produced no frames for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_playback_video(str(output), frames, fps=fps)


def _run_clip(
    *,
    checkpoint: Path,
    num_envs: int,
    yaws: np.ndarray,
    duration_s: float,
    seed: int,
    device: str,
    output: Path,
    stride: int,
    kind: str,
    low_camera: bool = False,
) -> None:
    cfg = _compose_owner(duration_s=duration_s, seed=seed)
    # algo cfg is the hydra cfg, not stored on the env. Load the policy with it.
    raw_env = _make_env(cfg, num_envs)
    wrapped: RslRlVecEnvWrapper | None = None
    try:
        wrapped = RslRlVecEnvWrapper(raw_env, device=device)
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        algorithm_cfg = train_cfg.get("algorithm")
        if isinstance(algorithm_cfg, dict):
            algorithm_cfg["enable_compile"] = False
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device=device)
        steps = int(round(duration_s / float(raw_env.cfg.ctrl_dt)))
        xy = np.zeros((num_envs, 2), dtype=np.float64)
        states = _rollout(raw_env, wrapped, policy, yaws=yaws, xy=xy, steps=steps)
        travel = _travel(states)
        print(
            f"{kind}: {len(states)} states, travel m = {np.round(travel, 2).tolist()}",
            flush=True,
        )
        shown = _subsample(states, stride)
        video_fps = max(1, int(round((1.0 / float(raw_env.cfg.ctrl_dt)) / stride)))
        with tempfile.TemporaryDirectory(prefix="straight-demo-") as tmp_dir:
            model_file = resolve_render_play_model_files(
                raw_env, num_envs=num_envs, tmp_dir=tmp_dir
            )
            if kind == "follow":
                _render_follow(shown, model_file, output, video_fps)
            else:
                _render_radial(shown, model_file, output, video_fps, low=low_camera)
        print(f"wrote {output}", flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--follow-seconds", type=float, default=8.0)
    parser.add_argument("--radial-seconds", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=2, help="Keep every Nth control step.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", choices=("follow", "radial", "both"), default="both")
    parser.add_argument(
        "--low-camera",
        action="store_true",
        help="Render the radial clip from the lower 13 m overhead camera.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if args.only in ("follow", "both"):
        _run_clip(
            checkpoint=args.checkpoint,
            num_envs=1,
            yaws=np.zeros(1, dtype=np.float64),
            duration_s=args.follow_seconds,
            seed=123,
            device=device,
            output=args.out_dir / "play_video_follow.mp4",
            stride=args.stride,
            kind="follow",
        )
    if args.only in ("radial", "both"):
        yaws = np.arange(8, dtype=np.float64) * (np.pi / 4.0)
        radial_name = "play_video_radial_low.mp4" if args.low_camera else "play_video_radial.mp4"
        _run_clip(
            checkpoint=args.checkpoint,
            num_envs=8,
            yaws=yaws,
            duration_s=args.radial_seconds,
            seed=123,
            device=device,
            output=args.out_dir / radial_name,
            stride=args.stride,
            kind="radial",
            low_camera=args.low_camera,
        )


if __name__ == "__main__":
    main()
