#!/usr/bin/env python
"""Live cv2 view to align a scene camera to its real reference. Flies a full 6-DOF camera
over episode 0's scene and prints its pose so you can copy it back into mujoco_env.

    jepa/scripts/mujoco-campose [top|side|wrist]     (default top)

top/side fly a fixed scene camera composited over its backdrop photo; the printed
pos/target/roll go into CAM_OVERHEAD / CAM_SIDE. wrist flies the gripper-mounted camera
blended 50/50 with the real recorded wrist frame; it prints the gripper-LOCAL pos + xyaxes
(what _add_wrist_mount's add_camera takes), since the wrist cam rides the arm.

Pan:    w/s in-out · a/d left-right · r/f up-down       (translate along the camera axes)
Rotate: i/k pitch  · j/l yaw       · u/o roll           (rotate about the camera axes)
FOV:    -/= · Esc quit. The pose line prints each keypress — copy the last one shown.
"""

import math
import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import mujoco_env as E
from mujoco_replay import build

WIN = (600, 800)  # (height, width)
_DATASETS = Path(os.environ.get("DATASETS_DIR", Path(__file__).resolve().parent / "datasets"))
ROOT = str(_DATASETS / "pick-cube-so101")
STEP_PAN = 0.005  # m per keypress
STEP_ROT = 0.75  # deg per keypress
STEP_FOV = 0.25  # deg per keypress
REF_FRAME = 110  # arm-over-desk frame to spawn at and reference against


def _fit(img, hw):
    """Centre-crop `img` (H,W,3 RGB) to the render aspect and resize to hw, returning BGR."""
    h, w = img.shape[:2]
    cw = int(h * hw[1] / hw[0])
    x = (w - cw) // 2
    return cv2.cvtColor(
        cv2.resize(img[:, x : x + cw], (hw[1], hw[0])), cv2.COLOR_RGB2BGR
    )


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roll_of(R):
    """The camera's roll (deg) relative to a level horizon, matching look_at_xyaxes."""
    up, back = R[:, 1], R[:, 2]
    world_up = np.array([0.0, 0.0, 1.0])
    lr = np.cross(world_up if abs(world_up @ back) < 0.99 else [0, 1, 0], back)
    lr /= np.linalg.norm(lr)
    lu = np.cross(back, lr)
    return math.degrees(math.atan2(-(up @ lr), up @ lu))


def main():
    which = {"top": "overhead", "side": "side", "wrist": "wrist"}.get(
        sys.argv[1] if len(sys.argv) > 1 else "top", "overhead"
    )
    scene, dataset, traj = build("baby_gewu/pick-cube-so101", ROOT, 0)
    scene.reset()
    for frame in traj[:REF_FRAME]:  # step in so the arm is over the desk
        scene.step(frame)
    m, d = scene.model, scene.data
    cid = m.camera(which).id
    pid = m.cam_bodyid[cid]  # parent body (worldbody for scene cams, gripper for wrist)
    r = mujoco.Renderer(m, *WIN)
    opt = mujoco.MjvOption()
    opt.geomgroup[E.COLLISION_GROUP] = 0

    if which == "wrist":  # no static backdrop — blend the real recorded wrist frame
        real = (dataset[REF_FRAME]["observation.images.camera2"].numpy() * 255).astype(
            np.uint8
        )
        bg = _fit(real.transpose(1, 2, 0), WIN)
    else:
        bg = _fit(np.asarray(Image.open(E.BACKDROP[which]))[:, :, :3], WIN)

    mujoco.mj_forward(m, d)
    pos = d.cam_xpos[cid].copy()
    R = d.cam_xmat[cid].reshape(3, 3).copy()  # [right | up | back]
    fovy = float(m.cam_fovy[cid])

    while True:
        pg, Rg = d.xpos[pid], d.xmat[pid].reshape(
            3, 3
        )  # place the camera in world space
        m.cam_pos[cid] = Rg.T @ (pos - pg)
        Rl = Rg.T @ R
        mujoco.mju_mat2Quat(m.cam_quat[cid], Rl.flatten())
        m.cam_fovy[cid] = fovy
        mujoco.mj_forward(m, d)

        r.update_scene(d, camera=which, scene_option=opt)
        sim = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        if which == "wrist":
            sim = cv2.addWeighted(sim, 0.5, bg, 0.5, 0.0)
        else:
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=which, scene_option=opt)
            seg = r.render()[:, :, 0] < 0
            r.disable_segmentation_rendering()
            sim[seg] = bg[seg]

        if which == "wrist":
            xy = np.round(np.concatenate([R[:, 0], R[:, 1]]), 4).tolist()
            line = f"wrist pos={np.round(m.cam_pos[cid], 4).tolist()} xyaxes={xy} fovy={fovy:.1f}"
        else:
            target = pos - R[:, 2]
            line = (
                f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                f"target=[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]  "
                f"roll={_roll_of(R):.2f} fovy={fovy:.1f}"
            )
        print(line, flush=True)
        cv2.putText(
            sim,
            line[:96],
            (8, WIN[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(
            f"campose [{which}]  ws/ad/rf pan  ik/jl/uo pitch-yaw-roll  -/= fov  Esc",
            sim,
        )

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        pans = {
            ord("w"): -R[:, 2],
            ord("s"): R[:, 2],
            ord("d"): R[:, 0],
            ord("a"): -R[:, 0],
            ord("r"): R[:, 1],
            ord("f"): -R[:, 1],
        }
        rots = {
            ord("i"): ("x", STEP_ROT),
            ord("k"): ("x", -STEP_ROT),
            ord("l"): ("y", STEP_ROT),
            ord("j"): ("y", -STEP_ROT),
            ord("u"): ("z", STEP_ROT),
            ord("o"): ("z", -STEP_ROT),
        }
        if k in pans:
            pos = pos + STEP_PAN * pans[k]
        elif k in rots:
            R = R @ _rot(*rots[k])
        elif k == ord("-"):
            fovy -= STEP_FOV
        elif k == ord("="):
            fovy += STEP_FOV


if __name__ == "__main__":
    main()
