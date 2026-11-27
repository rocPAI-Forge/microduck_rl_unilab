"""Publish an exported policy to the Hugging Face Hub and, optionally, enter it in Microduck Arena.

Does what microduck-challenges' ``uv run publish --onnx ... --enter`` does for an
ONNX already in hand, without its mjlab/CUDA dependencies:

1. create (or reuse) a public model repo tagged ``microduck-policy``;
2. upload ``policy.onnx``, a ``manifest.json`` the Arena checks
   (61 observations, 14 actions, robot ``microduck`` on legs) and a README,
   plus ``timeline.json`` for a stage event;
3. with ``--enter``, POST the repo at that commit to the Arena's
   ``/api/events/<event>/submit`` with the user's Hub token, and print the result.

The token is the one ``hf auth login`` stored (or ``HF_TOKEN``); it is sent to
the Arena only as the Bearer header ``publish --enter`` sends.

    uv run --no-sync scripts/arena_publish.py examples/sprint_stand/deploy/policy.onnx \
        --repo <user>/microduck-sprint-stand --event sprint-2m --enter
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token

ARENA = "https://pollen-robotics-microduck-arena.hf.space"
REPO_URL = "https://github.com/rocPAI-Forge/microduck_rl_unilab"


def manifest(name: str, description: str, event: str, kind: str) -> dict:
    return {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": 61,
        "action_len": 14,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50, "accessories": []},
        "name": name,
        "kind": kind,
        "entry_pose": "standing",
        "description": description,
        "arena": {"event": event},
        "training": {"method": "PPO (rsl_rl) on MuJoCo with UniLab, trained on an AMD ROCm GPU", "source": REPO_URL},
    }


def readme(name: str, description: str, event: str) -> str:
    return (
        "---\nlicense: apache-2.0\ntags:\n- microduck-policy\n- robotics\n- onnx\n---\n\n"
        f"# {name}\n\n{description}\n\n"
        f"Microduck walking-robot controller entered in the "
        f"[Microduck Arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena) `{event}` event. "
        "61 observations in, 14 joint position targets out, at 50 Hz.\n\n"
        f"Trained with PPO on MuJoCo ([microduck_rl_unilab]({REPO_URL})) on an AMD ROCm GPU.\n"
    )


def publish(args: argparse.Namespace, token: str) -> str:
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "policy.onnx").write_bytes(args.onnx.read_bytes())
        (folder / "manifest.json").write_text(json.dumps(manifest(args.name, args.description, args.event, args.kind), indent=2) + "\n")
        (folder / "README.md").write_text(readme(args.name, args.description, args.event))
        if args.timeline:
            (folder / "timeline.json").write_text(args.timeline.read_text())
        commit = api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(folder),
                                   commit_message=f"Publish {args.name} for {args.event}")
    return commit.oid


def enter(args: argparse.Namespace, token: str, revision: str) -> dict:
    form = {"repo_id": args.repo, "revision": revision, "name": args.name}
    if args.livery:
        form["livery"] = args.livery
    if args.command_vx is not None:
        form["command_vx"] = str(args.command_vx)
    response = requests.post(f"{ARENA}/api/events/{args.event}/submit", data=form,
                             headers={"Authorization": f"Bearer {token}"}, timeout=900)
    if response.status_code != 200:
        raise SystemExit(f"Arena refused the entry ({response.status_code}): {response.text}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--repo", required=True, help="<user>/<name> on the Hub")
    parser.add_argument("--event", required=True)
    parser.add_argument("--name", default=None, help="policy name on the board (default: the repo name)")
    parser.add_argument("--description", default="Microduck policy trained with microduck_rl_unilab.")
    parser.add_argument("--kind", default="perpetual")
    parser.add_argument("--timeline", type=Path, default=None, help="a stage event's timeline.json")
    parser.add_argument("--livery", default=None)
    parser.add_argument("--command-vx", type=float, default=None, help="race at one speed instead of the sweep")
    parser.add_argument("--enter", action="store_true")
    args = parser.parse_args()
    args.name = args.name or args.repo.split("/", 1)[1]
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token: run `hf auth login` (a write token) or set HF_TOKEN.")
    revision = publish(args, token)
    print(json.dumps({"repo": args.repo, "revision": revision}))
    if args.enter:
        print(json.dumps(enter(args, token, revision), indent=2)[:4000])


if __name__ == "__main__":
    main()
