"""Render a Microduck Arena trace (from scripts/arena_stage_preview.py) on the visual robot model.

Each trace frame is a duck's trunk pose (position, quaternion) and its 14
joint angles in the Arena's joint order, which is this repo's order too.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path(__file__).resolve().parents[1] / "assets/robots/microduck/microduck_bam.xml"
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def build_model(num_ducks: int, width: int, height: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                           rgb1=[0.82, 0.78, 0.70], rgb2=[0.74, 0.70, 0.62])
    mat = spec.add_material(name="floor", texrepeat=[16, 16], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4.0, 4.0, 0.05],
                            pos=[0, 0, 0], material="floor", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1.5, 0, 3.0], dir=[-0.4, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)
    for k in range(num_ducks):
        robot = mujoco.MjSpec.from_file(str(ROBOT_XML))
        frame = spec.worldbody.add_frame()
        frame.attach_body(robot.body("trunk_base"), f"duck{k}/", "")
    return spec.compile()


def render(trace: np.ndarray, ctrl_dt: float, out: Path, width: int, height: int, stride: int,
           distance: float, elevation: float) -> None:
    model = build_model(trace.shape[0], width, height)
    data = mujoco.MjData(model)
    addresses = []
    for k in range(trace.shape[0]):
        free = int(model.joint(f"duck{k}/trunk_base_freejoint").qposadr[0])
        joints = np.array([int(model.joint(f"duck{k}/{name}").qposadr[0]) for name in JOINT_NAMES])
        addresses.append((free, joints))
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.12)
    camera.distance = distance
    camera.azimuth = 180.0
    camera.elevation = elevation
    fps = round(1.0 / (ctrl_dt * stride))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None
    for frame in range(0, trace.shape[1], stride):
        for k, (free, joints) in enumerate(addresses):
            data.qpos[free : free + 7] = trace[k, frame, :7]
            data.qpos[joints] = trace[k, frame, 7:]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        ffmpeg.stdin.write(renderer.render().tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=2, help="control steps per video frame")
    parser.add_argument("--distance", type=float, default=2.1)
    parser.add_argument("--elevation", type=float, default=-12.0)
    args = parser.parse_args()
    saved = np.load(args.trace)
    trace = np.concatenate((saved["settle"], saved["trace"]), axis=1)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    render(trace, float(saved["ctrl_dt"]), args.video, args.width, args.height, args.stride,
           args.distance, args.elevation)
    print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
