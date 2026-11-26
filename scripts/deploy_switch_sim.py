#!/usr/bin/env python3
"""Rehearse switching exported policies on one simulated robot, as on hardware.

The environment is only the plant: the ground-contact duck with a ball in
front of its right foot (``microduck_ballkick_flat``). Every action comes from
``microduck_rl_unilab.deploy.Controller`` running exported ONNX files from raw
readings. At each segment the script switches policy with
``Controller.take_over``, which keeps the last action in the observation, the
way upstream microduck_rl's runtime hot-swaps policies behind one buffer.

Default sequence (``sprint_stand`` holds the robot between tricks):

    stand 2 s → kick (right) 2 s → stand 2 s → roulade 2.5 s → stand 2 s
    → sit 4 s → stand up 4 s → stand 2 s → tumble 2 s → stand 3 s

Per segment it reports the share of envs standing at the end (root above
0.10 m, upright cos > 0.9), and for the tricks: peak ball speed along the
heading (kick), forward rotation (roulade, tumble), and seated at the end
(sit, root below 0.075 m). Rotation is about the world lateral axis at the
segment start, as in ``eval_roll.py``. The sprint gait was trained on the walking model,
whose only floor contacts are the feet; here it stands on the ground-contact
model, which adds collision geoms that touch the floor only when the duck is
down.

Usage:
    uv run --no-sync scripts/deploy_switch_sim.py --num-envs 32 --output results/deploy_switch.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper
from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.cli import package_root

from microduck_rl_unilab.conf_searchpath import register_conf_search_path
from microduck_rl_unilab.deploy import Command, Controller, SensorReading

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
STAND_Z, UPRIGHT_COS, SEATED_Z = 0.10, 0.9, 0.075

# segment name: (deploy dir, command, seconds)
SEGMENTS: dict[str, tuple[Path, Command, float]] = {
    "stand": (EXAMPLES / "velstand_getup" / "deploy", Command.stand(), 2.0),
    "sprint_stand": (EXAMPLES / "sprint_stand" / "deploy", Command.stand(), 2.0),
    "kick_right": (EXAMPLES / "ballkick" / "deploy_right", Command.stand(), 2.0),
    "roulade": (EXAMPLES / "roulade" / "deploy", Command.stand(), 2.5),
    "sit": (EXAMPLES / "sitstand" / "deploy", Command.walk(1.0), 4.0),
    "stand_up": (EXAMPLES / "sitstand" / "deploy", Command.stand(), 4.0),
    "tumble": (EXAMPLES / "tumble" / "deploy", Command.stand(), 2.0),
}
DEFAULT_SEQUENCE = "stand,kick_right,stand,roulade,stand,sit,stand_up,stand,tumble,stand"


def _make_env(num_envs: int, seed: int, seconds: float, ball_offset_x: float) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=microduck_ballkick_flat/mujoco"])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.seed = seed
        cfg.env.max_episode_seconds = seconds + 5.0
        cfg.env.events.reset_ballkick.params.foot = "right"
        cfg.env.events.reset_ballkick.params.offset_x = ball_offset_x
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
    importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    return cast(Any, registry.make(str(cfg.training.task_name), sim_backend="mujoco",
                                   num_envs=num_envs, env_cfg_override=override))


def run(args: argparse.Namespace) -> dict[str, Any]:
    sequence = [(name, *SEGMENTS[name]) for name in args.sequence.split(",")]
    total = sum(seconds for *_, seconds in sequence)
    raw_env = _make_env(args.num_envs, args.seed, total, args.ball_offset_x)
    env = RslRlVecEnvWrapper(raw_env, device="cpu")
    try:
        n = args.num_envs
        robot, ball = raw_env.scene["robot"], raw_env.scene["ball"]
        data = robot.data
        dt = float(raw_env.cfg.ctrl_dt)
        first = Controller(sequence[0][1])
        names = first.manifest.joint_names
        default = first.manifest.default_joint_pos
        joint_ids = np.asarray([list(robot.joint_names).index(name) for name in names])
        env.get_observations()
        goal_angle = np.asarray(raw_env._kick_state.goal).copy()
        goal = np.stack((np.cos(goal_angle), np.sin(goal_angle)), axis=1)

        controllers: list[Controller | None] = [None] * n
        current_dir: Path | None = None
        states = [np.asarray(raw_env.get_physics_state_snapshot(), dtype=np.float32).copy()] if args.video else None
        segments = []
        for name, deploy_dir, command, seconds in sequence:
            if deploy_dir != current_dir:
                for i in range(n):
                    nxt = Controller(deploy_dir)
                    if controllers[i] is not None and args.handover == "keep":
                        nxt.take_over(controllers[i], command)
                    else:
                        nxt.set_command(command)
                    controllers[i] = nxt
                current_dir = deploy_dir
            else:
                for controller in controllers:
                    controller.set_command(command)
            # Forward rotation about the lateral axis the segment started with,
            # in the world frame: the tumble twists, so the body-frame pitch
            # rate under-counts it.
            heading = np.asarray(data.heading_w, dtype=np.float64)
            lateral = np.stack((-np.sin(heading), np.cos(heading), np.zeros(n)), axis=1)
            rotation = np.zeros(n)
            ball_peak = np.zeros(n)
            for _ in range(round(seconds / dt)):
                actions = np.empty((n, len(names)), dtype=np.float32)
                for i, controller in enumerate(controllers):
                    reading = SensorReading(
                        gyro_b=np.asarray(data.root_link_ang_vel_b[i]),
                        gravity_b=np.asarray(data.projected_gravity_b[i]),
                        joint_pos=np.asarray(data.joint_pos[i, joint_ids]),
                        joint_vel=np.asarray(data.joint_vel[i, joint_ids]),
                    )
                    actions[i] = controller.step(reading) - default
                env.step(torch.from_numpy(actions))
                rotation += np.sum(np.asarray(data.root_link_ang_vel_w) * lateral, axis=1) * dt
                ball_peak = np.maximum(ball_peak, np.sum(np.asarray(ball.data.root_link_lin_vel_w)[:, :2] * goal, axis=1))
                if states is not None:
                    states.append(np.asarray(raw_env.get_physics_state_snapshot(), dtype=np.float32).copy())
            z = np.asarray(data.root_link_pos_w)[:, 2]
            up = -np.asarray(data.projected_gravity_b)[:, 2]
            row: dict[str, Any] = {
                "segment": name,
                "policy": str(deploy_dir.relative_to(REPO_ROOT)),
                "seconds": seconds,
                "standing_at_end": float(((z > STAND_Z) & (up > UPRIGHT_COS)).mean()),
            }
            if name.startswith("kick"):
                row["peak_ball_speed_m_s_median"] = float(np.median(ball_peak))
            if name in ("roulade", "tumble"):
                row["forward_rotation_deg_median"] = float(np.degrees(np.median(rotation)))
                row["full_turn"] = float((rotation >= math.radians(330)).mean())
            if name == "sit":
                row["seated_at_end"] = float(((z < SEATED_Z) & (up > UPRIGHT_COS)).mean())
            segments.append(row)
            print(json.dumps(row), flush=True)
        if states is not None:
            _write_video(raw_env, states[::2], n, args.video)
        return {"num_envs": n, "seed": args.seed, "segments": segments,
                "standing_at_end_of_sequence": segments[-1]["standing_at_end"]}
    finally:
        env.close()


def _write_video(raw_env: Any, states: list[np.ndarray], n: int, out: Path) -> None:
    from unilab.visualization import render_many
    from unisim.backend.mujoco.playback import resolve_render_play_model_files
    from unisim.backend.playback_common import write_playback_video

    with tempfile.TemporaryDirectory() as tmp:
        model_file = resolve_render_play_model_files(raw_env, num_envs=n, tmp_dir=tmp)
        frames = render_many.render_states_get_frames_tracking(
            states, model_file, width=960, height=720, tracking_env_idx=0, max_extra_envs=0,
            cam_distance=0.9, cam_elevation=-12.0, cam_azimuth=120.0, render_spacing=0.0,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_playback_video(str(out), frames, fps=25)
    print(f"wrote {out}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help=f"comma list of {sorted(SEGMENTS)}")
    parser.add_argument("--handover", choices=("keep", "zero"), default="keep",
                        help="keep: the new policy sees the previous last action; zero: a fresh start")
    parser.add_argument("--ball-offset-x", type=float, default=0.09,
                        help="ball ahead of the right foot, m; large to keep it out of the way")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--video", type=Path, default=None, help="render env 0 to this mp4")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
