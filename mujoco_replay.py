#!/usr/bin/env python
"""Replay a recorded LeRobot episode in the MuJoCo SO-101 env.

Dumb consumer of mujoco_env: analyse the episode → place the cube at its grasp frame and
the tote at its release frame → drive the arm to the recorded trajectory. All scene
construction + physics live in mujoco_env; this file only decides placement (from the
recording) and control (the recorded joint targets), then displays the result.

    mjpython mujoco_replay.py <name> [episode]          # 3D viewer
    python   mujoco_replay.py <name> [episode] --grid   # camera grid

The 3D viewer runs under mjpython (macOS main-thread); the camera grid runs under plain
python (cv2's Cocoa GUI can't share mjpython's loop).
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from kinematics import (
    episode_distractors,
    grasp_frame,
    grasp_release_indices,
    release_frame,
)
from lerobot.datasets import LeRobotDataset

import mujoco_env as E


def build(repo_id, root, episode):
    """Load the episode and build its scene: cube at the grasp frame, tote at the release
    frame, arm homed to frame 0, plus the episode's recorded distractor_count of clutter
    objects sampled (seeded by episode) into the cube's fan. Returns (scene, dataset, traj).
    """
    dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
    states = dataset.select_columns(E.OBS_STATE)
    joints = [n.removesuffix(".pos") for n in dataset.features[E.OBS_STATE]["names"]]
    traj = np.array(
        [[float(v) for v in states[i][E.OBS_STATE]] for i in range(dataset.num_frames)]
    )
    robot = E.build_robot(joints)
    gx, gy, _, yaw = E.gripper_pose(
        robot.model, grasp_frame(traj, E.GRIP_WELD_MAX), robot.offsets
    )
    rx, ry, _, _ = E.gripper_pose(
        robot.model, release_frame(traj, E.GRIP_WELD_MAX), robot.offsets
    )
    count = episode_distractors(root).get(episode, 0)
    distractors = E.sample_distractors(
        count, (gx, gy), robot.pan_xy, np.random.default_rng(episode)
    )
    scene = E.Scene(
        joints,
        (gx, gy),
        yaw,
        (rx, ry),
        traj[0],
        dataset.fps,
        robot=robot,
        distractors=distractors,
    )
    return scene, dataset, traj


def run_headless(scene, traj):
    """Drive the recorded trajectory once, no display. Returns (success, welded_at_end):
    success if the released cube reached the tote floor; welded_at_end if still gripped.
    """
    scene.reset()
    for frame in traj:
        scene.step(frame)
    return scene.landed, scene.welded


def _label(img, text):
    import cv2

    cv2.rectangle(img, (0, 0), (11 * len(text) + 8, 26), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def _sim_row(scene, qpos):
    import cv2

    tiles = []
    for label, cam, _ in E.CAM_GRID:
        rgb = scene.render(cam, E.CAM_TILE, qpos=qpos if cam == "overhead" else None)
        tiles.append(_label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"{label} (sim)"))
    return np.hstack(tiles)


def _real_row(dataset, f):
    import cv2

    frame = dataset[f]
    tiles = []
    for label, _, key in E.CAM_GRID:
        rgb = (frame[key].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(
            cv2.resize(rgb, (E.CAM_TILE[1], E.CAM_TILE[0])), cv2.COLOR_RGB2BGR
        )
        tiles.append(_label(bgr, f"{label} (real)"))
    return np.hstack(tiles)


def run_grid(scene, dataset, traj):
    """Simulate once, snapshotting (qpos, weld) per frame, then scrub: play by default,
    space pauses, < / > step one frame, Esc quits. Overlays grasp/release frames + the jaw
    .pos and weld state, sim over real."""
    import cv2

    scene.reset()
    snaps = []
    for frame in traj:
        scene.step(frame)
        snaps.append((scene.data.qpos.copy(), scene.welded))
    grasp_i, release_i = grasp_release_indices(traj, E.GRIP_WELD_MAX)
    grip_i = scene.grip_i

    def draw(f, paused):
        qpos, welded = snaps[f]
        grid = np.vstack([_sim_row(scene, qpos), _real_row(dataset, f)])
        h, w = grid.shape[:2]
        here = " <GRASP>" if f == grasp_i else " <RELEASE>" if f == release_i else ""
        cv2.putText(
            grid,
            f"grasp {grasp_i}  release {release_i}",
            (8, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            grid,
            f"frame {f}/{len(snaps) - 1}{here}" + ("  PAUSED  < >" if paused else ""),
            (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        jaw = f"jaw {traj[f][grip_i]:.1f}  weld {'ON' if welded else 'OFF'}"
        (tw, _), _ = cv2.getTextSize(jaw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            grid,
            jaw,
            (w - tw - 8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if welded else (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        return grid

    f, paused = 0, False
    try:
        while True:
            cv2.imshow("cameras", draw(f, paused))
            key = cv2.waitKey(0 if paused else round(1000 / dataset.fps)) & 0xFF
            if key == 27:
                break
            elif key == 32:
                paused = not paused
            elif key in (ord("<"), ord(",")):
                paused, f = True, max(0, f - 1)
            elif key in (ord(">"), ord(".")):
                paused, f = True, min(len(snaps) - 1, f + 1)
            elif not paused:
                f = (f + 1) % len(snaps)
    finally:
        cv2.destroyAllWindows()


def run_viewer(scene, dataset, traj):
    """Loop the episode in the 3D viewer, real-time paced; close the window to stop."""
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.opt.geomgroup[E.COLLISION_GROUP] = 0
        while viewer.is_running():
            scene.reset()
            for frame in traj:
                t = time.perf_counter()
                scene.step(frame)
                viewer.sync()
                if not viewer.is_running():
                    return
                time.sleep(max(1 / dataset.fps - (time.perf_counter() - t), 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", dest="repo_id", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument(
        "--grid", action="store_true", help="camera grid instead of the viewer"
    )
    args = ap.parse_args()

    scene, dataset, traj = build(args.repo_id, args.root, args.episode)
    kind = "camera grid" if args.grid else "3D viewer"
    controls = (
        "space pauses · < / > step · Esc quits"
        if args.grid
        else "close the window to stop"
    )
    print(
        f"episode {args.episode}: {dataset.num_frames} frames @ {dataset.fps} fps — {kind}, {controls}"
    )
    (run_grid if args.grid else run_viewer)(scene, dataset, traj)


if __name__ == "__main__":
    main()
