"""Thin train/eval entrypoints that delegate to the UniLab package scripts.

The heavy lifting (runner, learner, collector, playback) stays inside the
published ``unilab`` wheel. This CLI only contributes the two seams UniLab
exposes for downstream repositories:

1. ``UNILAB_EXTRA_REGISTRY_PACKAGES`` — imports this package's task registry
   bootstrap inside every training/collector subprocess.
2. Hydra ``--config-dir`` — appends ``microduck_rl_unilab/conf`` to the UniLab
   config search path so external ``task=<task>/<sim>`` owner YAMLs resolve.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from unilab.cli import package_root as _unilab_package_root

SCRIPT_NAMES = {
    "ppo": "train_rsl_rl.py",
    "appo": "train_appo.py",
    "sac": "train_sac.py",
    "td3": "train_td3.py",
    "flashsac": "train_flashsac.py",
}
SUPPORTED_SIMS = ("mujoco", "mjwarp", "motrix")

# Per-algo config trees, mirroring UniLab's own conf/<algo>/ layout; the CLI
# appends conf/<algo> to the UniLab config search path via Hydra --config-dir.
CONF_ROOT = Path(__file__).resolve().parent / "conf"
_REGISTRY_PACKAGE = "microduck_rl_unilab.tasks"
_EXTRA_PACKAGES_ENV = "UNILAB_EXTRA_REGISTRY_PACKAGES"


def _ensure_registry_env() -> None:
    existing = os.environ.get(_EXTRA_PACKAGES_ENV, "")
    packages = [p.strip() for p in existing.split(",") if p.strip()]
    if _REGISTRY_PACKAGE not in packages:
        packages.append(_REGISTRY_PACKAGE)
    os.environ[_EXTRA_PACKAGES_ENV] = ",".join(packages)


def _build_command(mode: str, argv: Sequence[str] | None) -> list[str]:
    parser = argparse.ArgumentParser(prog=f"microduck-{mode}")
    parser.add_argument("--algo", required=True, choices=sorted(SCRIPT_NAMES))
    parser.add_argument("--task", required=True)
    parser.add_argument("--sim", required=True, choices=SUPPORTED_SIMS)
    if mode == "eval":
        parser.add_argument("--load-run", default=None)
    args, overrides = parser.parse_known_args(argv)

    script = _unilab_package_root() / "scripts" / SCRIPT_NAMES[args.algo]
    if not script.is_file():
        raise SystemExit(f"UniLab entrypoint script not found: {script}")
    conf_dir = CONF_ROOT / args.algo
    owner_yaml = conf_dir / "task" / args.task / f"{args.sim}.yaml"
    if not owner_yaml.is_file():
        raise SystemExit(
            f"No owner config exists for algo={args.algo}, task={args.task}, "
            f"sim={args.sim}: {owner_yaml}"
        )

    generated = [f"task={args.task}/{args.sim}"]
    if mode == "eval":
        generated.append("training.play_only=true")
        if args.load_run is not None:
            generated.append(f"algo.load_run={args.load_run}")
    return [
        sys.executable,
        str(script),
        "--config-dir",
        str(conf_dir),
        *generated,
        *overrides,
    ]


def _run(mode: str, argv: Sequence[str] | None) -> int:
    _ensure_registry_env()
    return subprocess.run(_build_command(mode, argv), check=False).returncode


def train_main(argv: Sequence[str] | None = None) -> int:
    return _run("train", argv)


def eval_main(argv: Sequence[str] | None = None) -> int:
    return _run("eval", argv)
