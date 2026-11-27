"""Play a Microduck Arena stage performance with local ONNX policies.

The Arena only borrows policies from the Hub, so this plays a troupe whose
ducks are all ``"self"`` in the timeline and swaps in one local ONNX per duck
(``--policies``, in duck order). It runs the Arena's own stage loop, prints the
Arena's per-duck summary and saves the trace for
``scripts/render_arena_trace.py`` (the Arena's stage model carries collision
geometry only, so it is rendered on this repo's visual model). Run it inside a
Microduck Arena checkout:

    uv run --project /path/to/microduck-arena python scripts/arena_stage_preview.py \
        arena_entries/talent/timeline.json --policies a.onnx b.onnx --trace-out talent.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from arena.constants import CTRL_DT
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import Performer, run_stage
from arena.stage import summaries
from arena.timeline import load_troupe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--policies", type=Path, nargs="+", required=True, help="one ONNX per duck, in order")
    parser.add_argument("--event", default="got-a-talent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trace-out", type=Path, default=None)
    args = parser.parse_args()

    manifest = load_event(args.event)
    troupe = load_troupe(args.timeline, manifest)
    if len(args.policies) != len(troupe.ducks):
        raise SystemExit(f"{len(troupe.ducks)} ducks, {len(args.policies)} policies")
    loaded = {path: load_policy(path) for path in args.policies}
    cast = [
        Performer(loaded[path], manifest.rig, duck.start, duck.timeline)
        for path, duck in zip(args.policies, troupe.ducks, strict=True)
    ]
    outcome = run_stage(manifest, cast, args.seed)
    print(json.dumps({"status": outcome.status, "played_s": round(outcome.trace.shape[1] * CTRL_DT, 2)}))
    rows = summaries(outcome.trace, manifest.termination)
    for k, (path, row) in enumerate(zip(args.policies, rows, strict=True)):
        print(json.dumps({"duck": k, "policy": str(path), **row}))
    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace_out, trace=outcome.trace, settle=outcome.settle, ctrl_dt=CTRL_DT)
        print(f"wrote {args.trace_out}")


if __name__ == "__main__":
    main()
