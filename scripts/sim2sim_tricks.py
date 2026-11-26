#!/usr/bin/env python3
"""MotrixSim cross-check for the motion examples, against MuJoCo on the same protocol.

Training is MuJoCo only; MotrixSim is held out (as in ``sim2sim_check.py`` for
the sprint gait). For each example the same checkpoint runs on both backends
with the same protocol, and the script measures one headline metric itself, so
both backends are scored the same way. A check passes when the Motrix value is
no more than ``--max-drop`` (absolute) below the MuJoCo value.

To run an owner on Motrix the script registers its task there and drops what
Motrix cannot do or does not accept: the interval push, joint-armature,
geom-friction and body-inertia randomization (as the walking Motrix owners do),
the roulade contact sensors (Motrix needs ``subtree2`` with ``subtree1``; the
plain task fragment is used), every reward term (unused when only
evaluating), and the basketball hold action (a body wrench Motrix lacks; zero
on the free ball). Commands are zero unless the protocol sets them.

| Example | Protocol | Headline metric |
|---|---|---|
| roulade, tumble | standing start, 4.9 s | full turn (≥ 330° about the start lateral axis) and standing at the end |
| sitstand | stand 1 s, sit until 7 s, stand until 12 s | seated at 6.5 s and standing at the end |
| ballkick | standing start, right foot, 4 s | ball past 0.5 m forward and standing at the end |
| kick | dribble, 12 s | ball past 0.5 m along the goal |
| scoot | backward 0.10 m/s, 10 s | survival with mean forward speed ≤ -0.05 m/s |
| stilts | 0.15 m/s, 60 s | survival |
| basketball | free ball, still, 60 s | survival |

    uv run --no-sync scripts/sim2sim_tricks.py --examples roulade,tumble --num-envs 128
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, cast

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
GEN = "assets/robots/microduck/generated"
STILT_OVERRIDES = [
    f"env.scene.model_file={GEN}/scene_flat_bam_stilt_h10_b050.xml",
    f"env.scene.fragment_files=[{GEN}/locomotion_task_stilt_h10_b050.xml]",
]
MOTRIX_DROPPED_EVENTS = ("push_robot", "randomize_armature", "foot_friction", "randomize_mass_inertia")
STAND_Z, UPRIGHT_COS, SEATED_Z = 0.10, 0.9, 0.075
# The rest keep their owner's reset (scoot starts seated).
STANDING_STARTS = ("roulade", "tumble", "sitstand", "ballkick")

# example: (checkpoint, owner, seconds, extra overrides)
EXAMPLES: dict[str, tuple[str, str, float, list[str]]] = {
    "roulade": ("examples/roulade/model_999.pt", "microduck_roulade_flat/mujoco", 4.9, []),
    "tumble": ("examples/tumble/model_17997.pt", "microduck_tumble_flat/mujoco", 4.9, []),
    "sitstand": ("examples/sitstand/model_6500.pt", "microduck_sitstand_fix_flat/mujoco", 12.0, []),
    "ballkick": ("examples/ballkick/model_right.pt", "microduck_ballkick_flat/mujoco", 4.0, []),
    "kick": ("examples/kick/model_60499.pt", "microduck_kick_flat/mujoco", 12.0, []),
    "scoot": ("examples/scoot/model_2999.pt", "microduck_scoot_sym_flat/mujoco", 10.0, []),
    "stilts": ("examples/stilts/model_4091.pt", "microduck_stilt_flat/mujoco", 60.0, STILT_OVERRIDES),
    "basketball": ("examples/basketball/model_12491.pt", "microduck_basketball_ft9_flat/mujoco", 60.0, []),
}


def _compose(owner: str, extra: list[str], backend: str, seed: int, seconds: float, example: str) -> Any:
    register_conf_search_path()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(package_root() / "conf" / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task={owner}", *extra])
    with open_dict(cfg):
        cfg.algo.seed = seed
        cfg.env.seed = seed
        cfg.env.max_episode_seconds = seconds + 2.0
        cfg.training.sim_backend = backend
        for name in cfg.env.curriculum:
            cfg.env.curriculum[name] = None
        events = cfg.env.events
        if "push_robot" in events:
            events.push_robot = None
        if "reset_roulade" in events and events.reset_roulade is not None:
            events.reset_roulade.params.standing_prob = 1.0
            events.reset_roulade.params.midroll_prob = 0.0
        if example in STANDING_STARTS and "set_ground_state" in events and events.set_ground_state is not None:
            g = events.set_ground_state.params
            g.standing_prob, g.sitting_prob, g.face_down_prob, g.face_up_prob = 1.0, 0.0, 0.0, 0.0
        if "ball_hold" in cfg.env.actions:
            cfg.env.actions.ball_hold.levels = [0.0]
        twist = cfg.env.commands.twist
        if example != "kick":
            twist.ranges.lin_vel_x = [0.0, 0.0]
            twist.ranges.lin_vel_y = [0.0, 0.0]
            twist.ranges.ang_vel_z = [0.0, 0.0]
            twist.resampling_time_range = [seconds + 2.0, seconds + 2.0]
            for key in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs",
                        "turn_in_place_fraction", "straight_fraction"):
                if key in twist:
                    twist[key] = 0.0
            if "sit_prob" in twist:
                twist.sit_prob = 0.0
        for name in ("head_pose", "body_pose"):
            if name in cfg.env.commands:
                command = cfg.env.commands[name]
                command.ranges = [[0.0, 0.0] for _ in command.ranges]
        if backend == "motrix":
            for name in MOTRIX_DROPPED_EVENTS:
                if name in events:
                    events[name] = None
            cfg.env.scene.fragment_files = [
                f.replace("assets/robots/microduck/roulade_task.xml", "assets/robots/microduck/locomotion_task.xml")
                for f in cfg.env.scene.fragment_files
            ]
            for name in list(cfg.reward.keys()):
                cfg.reward[name] = None
            if "ball_hold" in cfg.env.actions:
                # Applies a body wrench every step, which Motrix lacks; at
                # hold level 0 (the free ball) the wrench is zero anyway.
                cfg.env.actions.ball_hold = None
    return cfg


def _make(cfg: Any, backend: str, num_envs: int) -> Any:
    tasks = importlib.import_module("microduck_rl_unilab.tasks.microduck")
    registry.ensure_registries()
    task = str(cfg.training.task_name)
    if backend == "motrix":
        try:
            registry.register_env(task, tasks.make_microduck_velocity_env, sim_backend="motrix")
        except (KeyError, ValueError):
            pass  # already registered (walking tasks)
    override = BackendAdapter(cfg, root_dir=REPO_ROOT, algo_name="ppo").build_task_env_cfg_override()
    return cast(Any, registry.make(task, sim_backend=backend, num_envs=num_envs, env_cfg_override=override))


def _standing(data: Any) -> np.ndarray:
    z = np.asarray(data.root_link_pos_w)[:, 2]
    up = -np.asarray(data.projected_gravity_b)[:, 2]
    return (z > STAND_Z) & (up > UPRIGHT_COS)


def _upright(data: Any) -> np.ndarray:
    return -np.asarray(data.projected_gravity_b)[:, 2] > math.cos(math.radians(60))


def evaluate(example: str, backend: str, num_envs: int, seed: int) -> dict[str, Any]:
    checkpoint, owner, seconds, extra = EXAMPLES[example]
    cfg = _compose(owner, extra, backend, seed, seconds, example)
    raw = _make(cfg, backend, num_envs)
    env = RslRlVecEnvWrapper(raw, device="cpu")
    try:
        train_cfg = normalize_ppo_train_cfg(algo_config_dict(cfg))
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        if isinstance(train_cfg.get("algorithm"), dict):
            train_cfg["algorithm"]["enable_compile"] = False
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
        runner.load(str(REPO_ROOT / checkpoint), map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        obs = env.get_observations()
        data = raw.scene["robot"].data
        dt = float(raw.cfg.ctrl_dt)
        n = num_envs
        heading = np.asarray(data.heading_w, dtype=np.float64).copy()
        lateral = np.stack((-np.sin(heading), np.cos(heading), np.zeros(n)), axis=1)
        rotation = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        vx_sum, vx_count = np.zeros(n), 0
        seated = np.zeros(n, dtype=bool)
        ball = raw.scene["ball"] if example in ("ballkick", "kick") else None
        ball0 = np.asarray(ball.data.root_link_pos_w)[:, :2].copy() if ball is not None else None
        goal = None
        if ball is not None:
            angle = np.asarray(raw._kick_state.goal).copy()
            goal = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        twist = raw.command_manager.get_term("twist")
        for step in range(round(seconds / dt)):
            t = step * dt
            if example == "sitstand":
                twist.vel_command_b[:, 0] = 1.0 if 1.0 <= t < 7.0 else 0.0
            elif example == "scoot":
                twist.vel_command_b[:, 0] = -0.10
            elif example == "stilts":
                twist.vel_command_b[:, 0] = 0.15
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            rotation += np.sum(np.asarray(data.root_link_ang_vel_w) * lateral, axis=1) * dt
            if example in ("kick", "scoot", "stilts", "basketball"):
                alive &= ~dones.cpu().numpy().astype(bool) & _upright(data)
            if example in ("scoot", "stilts") and t >= 1.0:
                vx_sum += np.asarray(data.root_link_lin_vel_b)[:, 0]
                vx_count += 1
            if example == "sitstand" and abs(t - 6.5) < dt / 2:
                z = np.asarray(data.root_link_pos_w)[:, 2]
                seated = (z < SEATED_Z) & _upright(data)
        standing = _standing(data)
        if example in ("roulade", "tumble"):
            value = (rotation >= math.radians(330)) & standing
            label = "full turn and standing at the end"
        elif example == "sitstand":
            value = seated & standing
            label = "seated at 6.5 s and standing at the end"
        elif example == "ballkick":
            along = np.sum((np.asarray(ball.data.root_link_pos_w)[:, :2] - ball0) * goal, axis=1)
            value = (along > 0.5) & standing
            label = "ball past 0.5 m and standing at the end"
        elif example == "kick":
            along = np.sum((np.asarray(ball.data.root_link_pos_w)[:, :2] - ball0) * goal, axis=1)
            value = (along > 0.5) & alive
            label = "ball past 0.5 m along the goal, not fallen"
        elif example == "scoot":
            value = alive & (vx_sum / max(vx_count, 1) <= -0.05)
            label = "survived and backed up at least 0.05 m/s"
        else:
            value = alive
            label = "survival"
        result: dict[str, Any] = {
            "example": example,
            "backend": backend,
            "checkpoint": checkpoint,
            "num_envs": n,
            "seed": seed,
            "seconds": seconds,
            "metric": label,
            "value": float(np.mean(value)),
            "standing_at_end": float(np.mean(standing)),
        }
        if example in ("scoot", "stilts"):
            result["forward_speed_m_s_mean"] = float(np.mean(vx_sum / max(vx_count, 1)))
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=",".join(EXAMPLES), help=f"comma list of {list(EXAMPLES)}")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--max-drop", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for example in args.examples.split(","):
        pair = {backend: evaluate(example, backend, args.num_envs, args.seed) for backend in ("mujoco", "motrix")}
        drop = pair["mujoco"]["value"] - pair["motrix"]["value"]
        row = {"example": example, "metric": pair["mujoco"]["metric"], "mujoco": pair["mujoco"]["value"],
               "motrix": pair["motrix"]["value"], "drop": drop, "pass": bool(drop <= args.max_drop), "runs": pair}
        rows.append(row)
        print(f"{example:10s} {row['metric']:45s} mujoco {row['mujoco']:.3f}  motrix {row['motrix']:.3f}  "
              f"{'pass' if row['pass'] else 'FAIL'}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"max_drop": args.max_drop, "rows": rows}, indent=2) + "\n")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
