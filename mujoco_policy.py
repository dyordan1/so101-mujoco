#!/usr/bin/env python
"""Roll out a trained SmolVLA policy in the MuJoCo SO-101 env.

Dumb consumer of mujoco_env: place the cube at a chosen grid spot + the tote at its
cluster, then let the policy drive. All scene construction + physics live in mujoco_env;
this file only samples placement, loads the policy, and does the sim↔policy glue (sim
renders → observation, policy action → scene.step).

    jepa/scripts/mujoco-policy <checkpoint> [--reach CM] [--azim DEG] [--view] [--seconds N]

--view shows the mjpython 3D viewer; default is headless. Non-realtime (CPU/MPS inference).
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

import mujoco_env as E

HERE = Path(__file__).resolve().parent
# Repo-root datasets/ (gitignored); DATASETS_DIR overrides it (the monorepo wrapper
# points it at the shared jepa/datasets/ tree).
DATASETS = Path(os.environ.get("DATASETS_DIR", HERE / "datasets"))
CALIBRATION = HERE / "calib" / "calibration.json"
TOTE_XY = (0.0, -0.384)  # measured release cluster centre (m)
RENDER_HW = (480, 640)  # match the recorded 4:3 frames; the policy preprocessor resizes
AZIM_LIMIT = (
    30.0  # cube kept within ±this of straight-ahead (real workspace is centred)
)
# --sweep eval grid: cube placements (cm reach × deg azimuth across the workspace).
# 15/25 cm are the trained rings, 20 the interpolation test.
SWEEP_REACHES = (15.0, 17.5, 20.0, 22.5, 25.0)
SWEEP_AZIMS = (-90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
SWEEP_TRIALS = 5  # samples per cell: 1 exact + jittered, so the rate isn't a coin flip
SWEEP_JITTER_M = 0.02  # ±2 cm cube placement jitter (matches the dataset's dense jitter)
SWEEP_YAW_DEG = 45.0  # ±this cube-yaw jitter; a cube is 4-fold symmetric so ±45° = all orientations


def load_policy(ckpt, device):
    """Load policy + saved pre/post processors, repinned to `device`. Mirrors
    eval-policy.py's load (the real-arm path), touches no hardware."""
    repo_id = json.loads(Path(ckpt, "train_config.json").read_text())["dataset"][
        "repo_id"
    ]
    # Use the checkpoint's own repo_id; read from a local datasets/<name> copy if one
    # exists (downloaded or monorepo-shared), else let LeRobot pull it from the Hub.
    local = DATASETS / repo_id.split("/")[-1]
    ds_meta = LeRobotDatasetMetadata(
        repo_id, root=str(local) if local.exists() else None
    )
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = str(device)
    cfg.compile_model = False
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map={})
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post, ds_meta, ds_meta.tasks.index.tolist()[0]


def cube_xy(pan_xy, reach_cm, azim_deg):
    """World (x, y) of a cube at (reach, azimuth) from the shoulder-pan axis: azimuth 0 =
    straight ahead (robot faces -Y), + = the robot's right (-X)."""
    az = math.radians(azim_deg)
    r = reach_cm / 100.0
    return (pan_xy[0] - r * math.sin(az), pan_xy[1] - r * math.cos(az))


def rollout(
    scene, policy, pre, post, ds_meta, task, device, joints, seconds, display=None
):
    """Drive the policy for `seconds` at 30 Hz: sim renders → observation → action →
    scene.step. `display(raw)` (viewer sync or camera grid) returns True to stop. Verdict.
    """
    policy.reset()
    for _ in range(int(seconds * 30)):
        raw = {
            "camera1": scene.render("overhead", RENDER_HW),
            "camera2": scene.render("wrist", RENDER_HW),
            "camera3": scene.render("side", RENDER_HW),
        }
        for name, val in zip(joints, scene.pos_state()):
            raw[f"{name}.pos"] = val
        obs = build_dataset_frame(ds_meta.features, raw, prefix=OBS_STR)
        action_t = predict_action(
            obs,
            policy,
            device,
            pre,
            post,
            use_amp=False,
            task=task,
            robot_type="so_follower",
        )
        action = action_t.detach().cpu().numpy().reshape(-1)
        if action.shape[0] != len(joints):
            raise SystemExit(
                f"policy action shape {action_t.shape}, expected {len(joints)}"
            )
        scene.step(action)
        if display and display(raw):
            return
        if scene.landed:
            print("SUCCESS: cube in tote")
            return
    print(f"done: success={scene.landed}")


def grid_display(raw):
    """cv2 row of the three sim cameras the policy sees (top | wrist | side). Esc stops."""
    import cv2

    tiles = []
    for label, _, key in E.CAM_GRID:
        rgb = raw[key.split(".")[-1]]
        bgr = cv2.cvtColor(
            cv2.resize(rgb, (E.CAM_TILE[1], E.CAM_TILE[0])), cv2.COLOR_RGB2BGR
        )
        cv2.rectangle(bgr, (0, 0), (11 * len(label) + 40, 26), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            f"{label} (policy)",
            (5, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        tiles.append(bgr)
    cv2.imshow("policy view (top | wrist | side)", np.hstack(tiles))
    return cv2.waitKey(1) == 27


def sweep(policy, pre, post, ds_meta, task, device, joints, home_deg, seconds, view=False):
    """Roll the policy out over a reach×azimuth grid of cube placements and print the
    success rate — the sim eval, vs a single --reach/--azim rollout. With `view`, each
    placement plays in the mjpython 3D viewer (a fresh window per cell)."""
    # Same jitter offsets reused for every cell (seeded → reproducible): the first is
    # the exact placement, the rest are ±SWEEP_JITTER_M so a cell's rate reflects
    # robustness to placement, not one lucky pose.
    rng = random.Random(0)
    yaw = math.radians(SWEEP_YAW_DEG)
    offsets = [(0.0, 0.0, 0.0)] + [
        (rng.uniform(-SWEEP_JITTER_M, SWEEP_JITTER_M),
         rng.uniform(-SWEEP_JITTER_M, SWEEP_JITTER_M),
         rng.uniform(-yaw, yaw))
        for _ in range(SWEEP_TRIALS - 1)
    ]
    res = {}
    for r in SWEEP_REACHES:
        for a in SWEEP_AZIMS:
            hits = 0
            for dx, dy, dyaw in offsets:
                robot = E.build_robot(joints)  # fresh spec per trial (Scene mutates it)
                x, y = cube_xy(robot.pan_xy, r, a)
                scene = E.Scene(
                    joints, (x + dx, y + dy), dyaw, TOTE_XY, home_deg, robot=robot
                )
                args = (scene, policy, pre, post, ds_meta, task, device, joints, seconds)
                if view:
                    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
                        viewer.opt.geomgroup[E.COLLISION_GROUP] = 0
                        rollout(
                            *args,
                            display=lambda _raw: (viewer.sync(), not viewer.is_running())[1],
                        )
                else:
                    rollout(*args)
                hits += bool(scene.landed)
            res[(r, a)] = hits
            print(f"  reach={r:.0f} azim={a:+.0f}: {hits}/{SWEEP_TRIALS}")
    total = len(res) * SWEEP_TRIALS
    s = sum(res.values())
    print(f"\nSUCCESS {s}/{total} = {100 * s / total:.0f}%")
    print("reach\\azim " + " ".join(f"{a:+5.0f}" for a in SWEEP_AZIMS))
    for r in SWEEP_REACHES:
        cells = " ".join(f"{res[(r, a)]}/{SWEEP_TRIALS}".rjust(5) for a in SWEEP_AZIMS)
        print(f"  {r:4.0f}cm  " + cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--reach", type=float, default=20.0, help="cube reach cm")
    ap.add_argument(
        "--azim", type=float, default=0.0, help="cube azimuth deg (+ right)"
    )
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--view", action="store_true", help="show the mjpython 3D viewer")
    ap.add_argument(
        "--grid", action="store_true", help="show the sim camera grid (cv2)"
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="eval: sweep a placement grid, report success rate",
    )
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    policy, pre, post, ds_meta, task = load_policy(args.checkpoint, device)
    joints = [n.removesuffix(".pos") for n in ds_meta.features[E.OBS_STATE]["names"]]
    home = json.loads(CALIBRATION.read_text())["baby_gewu_robot"]["home_pose"]
    home_deg = [home[f"{n}.pos"] for n in joints]
    if args.sweep:
        sweep(
            policy, pre, post, ds_meta, task, device, joints, home_deg,
            args.seconds, view=args.view,
        )
        return

    azim = max(-AZIM_LIMIT, min(AZIM_LIMIT, args.azim))
    if azim != args.azim:
        print(f"azim {args.azim} clamped to ±{AZIM_LIMIT}")

    robot = E.build_robot(joints)
    cxy = cube_xy(robot.pan_xy, args.reach, azim)
    scene = E.Scene(joints, cxy, 0.0, TOTE_XY, home_deg, robot=robot)
    print(
        f"loaded policy on {device}; task={task!r}; cube reach={args.reach} azim={azim}"
        f" -> world {tuple(round(v, 3) for v in cxy)}"
    )

    args_common = (
        scene,
        policy,
        pre,
        post,
        ds_meta,
        task,
        device,
        joints,
        args.seconds,
    )
    if args.view:
        with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
            viewer.opt.geomgroup[E.COLLISION_GROUP] = 0
            rollout(
                *args_common,
                display=lambda _raw: (viewer.sync(), not viewer.is_running())[1],
            )
    elif args.grid:
        import cv2

        try:
            rollout(*args_common, display=grid_display)
        finally:
            cv2.destroyAllWindows()
    else:
        rollout(*args_common)


if __name__ == "__main__":
    main()
