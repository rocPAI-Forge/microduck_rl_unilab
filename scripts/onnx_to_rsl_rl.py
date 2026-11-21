#!/usr/bin/env python3
"""Turn an exported microduck ONNX policy into an rsl_rl checkpoint to fine-tune.

The ONNX must be the upstream export layout: ``Sub(obs, mean)``, ``Div(., std + eps)``,
then Gemm/Elu layers named ``mlp.<i>.weight`` / ``mlp.<i>.bias`` (rsl_rl
``MLPModel`` with ``obs_normalization``). The actor comes from the ONNX; the
critic (and its normalizer) from a donor checkpoint of the same task, so PPO
does not start from a random value function. The optimizer state is reset.

    uv run --no-sync python scripts/onnx_to_rsl_rl.py roulade.onnx \
        --donor logs/.../model_6000.pt --out logs/roulade_upstream/model_0.pt --std 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper

NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization default, baked into the ONNX divisor


def convert(onnx_path: Path, donor_path: Path, std: float, count: int) -> dict:
    graph = onnx.load(str(onnx_path)).graph
    init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}
    div = next(n for n in graph.node if n.op_type == "Div")
    divisor = init[div.input[1]]
    for node in graph.node:
        if node.op_type == "Gemm":
            attrs = {a.name: a.i for a in node.attribute}
            if attrs.get("transB", 0) != 1:
                raise ValueError(f"Gemm {node.name} is not transB=1; layout not supported")

    donor = torch.load(str(donor_path), map_location="cpu", weights_only=False)
    actor = {k: v.clone() for k, v in donor["actor_state_dict"].items()}
    for name, value in init.items():
        if name.startswith("mlp."):
            if actor[name].shape != value.shape:
                raise ValueError(f"{name}: donor {tuple(actor[name].shape)} vs onnx {tuple(value.shape)}")
            actor[name] = value.to(actor[name].dtype)
    obs_std = (divisor - NORMALIZER_EPS).clamp_min(0.0)
    actor["obs_normalizer._mean"] = init["obs_normalizer._mean"].to(actor["obs_normalizer._mean"].dtype)
    actor["obs_normalizer._std"] = obs_std.to(actor["obs_normalizer._std"].dtype)
    actor["obs_normalizer._var"] = (obs_std * obs_std).to(actor["obs_normalizer._var"].dtype)
    actor["obs_normalizer.count"] = torch.tensor(count, dtype=actor["obs_normalizer.count"].dtype)
    actor["distribution.std_param"] = torch.full_like(actor["distribution.std_param"], std)

    optimizer = donor["optimizer_state_dict"]
    return {
        "actor_state_dict": actor,
        "critic_state_dict": donor["critic_state_dict"],
        "optimizer_state_dict": {"state": {}, "param_groups": optimizer["param_groups"]},
        "iter": 0,
        "infos": None,
        "unilab_logger_state": donor.get("unilab_logger_state"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--donor", type=Path, required=True, help="checkpoint of the same task (critic, layout)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--std", type=float, default=0.2, help="initial action std for fine-tuning")
    parser.add_argument("--count", type=int, default=100_000_000, help="normalizer sample count (high = frozen)")
    args = parser.parse_args()
    ckpt = convert(args.onnx, args.donor, args.std, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
