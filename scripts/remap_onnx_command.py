"""Remap the forward-speed command an exported policy reads, inside its ONNX graph.

Microduck Arena races sweep forward commands 0.20-1.20 m/s, while the fast
sprint checkpoints were trained on 1.65-2.20 m/s. This prepends
``obs * scale + bias`` to the graph, identity on every observation except
``twist_cmd`` x (index 48 of the 61-D contract), which maps the interval
``--from`` linearly onto ``--to``. The result is still one 61-in, 14-out ONNX.

    uv run --no-sync scripts/remap_onnx_command.py examples/x/deploy/policy.onnx \
        --from 0.2 1.2 --to 1.65 2.2 --output /tmp/x_remapped.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

TWIST_X = 48
OBS_SIZE = 61


def remap(model: onnx.ModelProto, lo: float, hi: float, new_lo: float, new_hi: float) -> onnx.ModelProto:
    graph = model.graph
    (inp,) = graph.input
    scale = np.ones((1, OBS_SIZE), dtype=np.float32)
    bias = np.zeros((1, OBS_SIZE), dtype=np.float32)
    scale[0, TWIST_X] = (new_hi - new_lo) / (hi - lo)
    bias[0, TWIST_X] = new_lo - lo * scale[0, TWIST_X]
    original, scaled, remapped = inp.name, f"{inp.name}_scaled", f"{inp.name}_remapped"
    for node in graph.node:
        node.input[:] = [remapped if name == original else name for name in node.input]
    graph.initializer.extend([numpy_helper.from_array(scale, "command_remap_scale"),
                              numpy_helper.from_array(bias, "command_remap_bias")])
    graph.node.insert(0, helper.make_node("Add", [scaled, "command_remap_bias"], [remapped]))
    graph.node.insert(0, helper.make_node("Mul", [original, "command_remap_scale"], [scaled]))
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--from", dest="src", type=float, nargs=2, default=[0.2, 1.2])
    parser.add_argument("--to", dest="dst", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = remap(onnx.load(str(args.onnx)), *args.src, *args.dst)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))

    before = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    after = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(1, OBS_SIZE)).astype(np.float32)
    obs[0, TWIST_X] = args.src[1]
    direct = obs.copy()
    direct[0, TWIST_X] = args.dst[1]
    name_b, name_a = before.get_inputs()[0].name, after.get_inputs()[0].name
    diff = np.abs(after.run(None, {name_a: obs})[0] - before.run(None, {name_b: direct})[0]).max()
    print(f"wrote {args.output}; command {args.src[1]} -> {args.dst[1]} matches the original within {diff:.2e}")


if __name__ == "__main__":
    main()
