"""Write the "Domino ducks" Got a Talent timeline (timeline.json beside this file).

Ducks, in `ducks` order (the first is the entry itself):

- 0-2 front row, tumble policy: `twist = [-1, 0, 0]` holds it standing, and
  releasing to zeros sets off one sideways tumble about 0.2 s later.
- 3-4 flanks, roulade policy: rolls forward once, at the start.
- 5-6 back row, sitstand policy: `twist[0]` is the sit flag.
- 7 rear, sprint_stand policy: walks across the back (from standing it
  needs 0.3 m/s; 0.2 m/s does not start a step).

Borrowed ducks name their Hub repo at a commit; ``POLICIES`` holds those
references; scripts/arena_stage_preview.py plays the file with one local
ONNX per duck instead.
"""

from __future__ import annotations

import json
from pathlib import Path

DURATION = 20.0
HOLD = [-1.0, 0.0, 0.0]
SIT = [1.0, 0.0, 0.0]
POLICIES = {
    "tumble": "self",
    "roulade": "alexhegit/microduck-roulade@b64aeaf525ecd0f8cb5db3804590922bef4c4ba5",
    "sitstand": "alexhegit/microduck-sitstand@10b4b9ba0e5ca743066bd6f0ff43b655610e2612",
    "walker": "alexhegit/microduck-sprint-stand@14f9b31211c6aace1557d4598a14ca77bba823d8",
}


def _keyframes(changes: list[tuple[float, dict]]) -> list[dict]:
    return [{"t": t, **groups} for t, groups in changes]


def tumbler(releases: list[float], holds: list[float]) -> list[dict]:
    """Hold at 0 and at each `holds` time; release (and wheee, then quack) at each `releases` time."""
    changes: list[tuple[float, dict]] = [(0.0, {"twist": HOLD})]
    events = sorted([(t, "release") for t in releases] + [(t, "hold") for t in holds])
    for t, kind in events:
        if kind == "release":
            changes += [(t, {"triggers": [0, 1]}), (t + 0.9, {"triggers": [1, 0]}), (t + 1.1, {})]
        else:
            changes.append((t, {"twist": HOLD}))
    return _keyframes(sorted(changes, key=lambda c: c[0]))


def roller() -> list[dict]:
    return _keyframes([(0.0, {"triggers": [0, 1]}), (1.0, {}), (1.6, {"triggers": [1, 0]}), (1.8, {})])


def sitter(sits: list[tuple[float, float]], bow: float) -> list[dict]:
    changes: list[tuple[float, dict]] = [(0.0, {})]
    for down, up in sits:
        changes += [(down, {"twist": SIT}), (up, {})]
    changes += [(bow, {"twist": SIT}), (bow + 0.5, {"twist": SIT, "triggers": [1, 0]}),
                (bow + 0.7, {"twist": SIT})]
    return _keyframes(changes)


def walker() -> list[dict]:
    return _keyframes([(0.0, {}), (2.0, {"twist": [0.3, 0.0, 0.0]}), (12.0, {})])


def timeline() -> dict:
    fanfare = [("C5", 18.0, 0.3), ("E5", 18.35, 0.3), ("G5", 18.7, 0.3), ("C6", 19.05, 0.8)]
    ducks = [
        {"policy": "self", "start": {"x": -0.05, "y": 0.0, "yaw": 0},
         "voice": {"pitch": 0.55},
         "song": [{"t": t, "note": n, "for": d, "vowel": "ah"} for n, t, d in fanfare],
         "keyframes": tumbler([3.6, 10.6, 17.0], [6.6, 13.6])},
        {"policy": POLICIES["tumble"], "start": {"x": -0.05, "y": -0.7, "yaw": 0},
         "voice": {"pitch": 0.4},
         "keyframes": tumbler([3.0, 11.2, 17.0], [6.0, 14.2])},
        {"policy": POLICIES["tumble"], "start": {"x": -0.05, "y": 0.7, "yaw": 0},
         "voice": {"pitch": 0.7},
         "keyframes": tumbler([4.2, 10.0, 17.0], [7.2, 13.0])},
        {"policy": POLICIES["roulade"], "start": {"x": 0.0, "y": -1.4, "yaw": 0},
         "voice": {"pitch": 0.3}, "keyframes": roller()},
        {"policy": POLICIES["roulade"], "start": {"x": 0.0, "y": 1.4, "yaw": 0},
         "voice": {"pitch": 0.8}, "keyframes": roller()},
        {"policy": POLICIES["sitstand"], "start": {"x": -0.5, "y": -0.5, "yaw": 0},
         "voice": {"pitch": 0.35}, "keyframes": sitter([(5.0, 8.5), (12.0, 15.0)], 18.0)},
        {"policy": POLICIES["sitstand"], "start": {"x": -0.5, "y": 0.5, "yaw": 0},
         "voice": {"pitch": 0.65}, "keyframes": sitter([(5.6, 9.1), (12.6, 15.6)], 18.3)},
        {"policy": POLICIES["walker"], "start": {"x": -0.95, "y": -1.5, "yaw": 90},
         "voice": {"pitch": 0.5}, "keyframes": walker()},
    ]
    return {"timeline_version": 2, "duration_s": DURATION, "ducks": ducks}


if __name__ == "__main__":
    out = Path(__file__).with_name("timeline.json")
    out.write_text(json.dumps(timeline(), indent=1) + "\n")
    print(f"wrote {out}")
