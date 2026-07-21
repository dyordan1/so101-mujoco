#!/usr/bin/env python
"""Record a MuJoCo sim twin of a recorded LeRobot dataset.

Replays every episode of <name> through the sim (the validated mujoco_replay path — cube
welded at the grasp frame, tote at the release frame, the episode's distractors in the fan)
and writes the sim's three camera renders plus the recorded states/actions into a new
dataset <name>-sim. Proprioception (action + observation.state) is copied verbatim from the
source; only the pixels come from MuJoCo.

Each episode is rolled twice: once physics-only to check the cube lands in the tote, then —
only if it did — again with rendering to write the frames. Episodes that never land in sim
are DROPPED, so the twin has fewer than the source's episodes, by design (a sim-fidelity
filter). The kept episodes are renumbered 0..N-1.

    jepa/scripts/mujoco-record <name> [limit]   # limit>0: first N episodes; else all

Runs under plain python (no GUI). Requires the repo dev shell.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mujoco_replay import build, run_headless

HERE = Path(__file__).resolve().parent
# Repo-root datasets/ (gitignored). DATASETS_DIR points it elsewhere — the monorepo
# wrapper sets it to the shared jepa/datasets/ tree the real recordings live in.
DATASETS = Path(os.environ.get("DATASETS_DIR", HERE / "datasets"))
FPS = 30
RENDER_HW = (480, 640)
IMAGE_FEATURE = {
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"],
}


def features_for(joint_names):
    vec = {"dtype": "float32", "shape": [len(joint_names)], "names": joint_names}
    return {
        "action": dict(vec),
        "observation.state": dict(vec),
        "observation.images.camera1": dict(IMAGE_FEATURE),
        "observation.images.camera2": dict(IMAGE_FEATURE),
        "observation.images.camera3": dict(IMAGE_FEATURE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument(
        "limit", nargs="?", type=int, default=0, help="first N episodes (0 = all)"
    )
    args = ap.parse_args()

    src_root = str(DATASETS / args.name)
    src_repo = f"baby_gewu/{args.name}"
    out_name = f"{args.name}-sim"
    out_root = DATASETS / out_name
    info = json.loads((Path(src_root) / "meta/info.json").read_text())
    n = args.limit or info["total_episodes"]
    task = pd.read_parquet(Path(src_root) / "meta/tasks.parquet").index[0]

    if out_root.exists():
        shutil.rmtree(out_root)
    sim = LeRobotDataset.create(
        repo_id=f"baby_gewu/{out_name}",
        fps=FPS,
        features=features_for(info["features"]["observation.state"]["names"]),
        root=str(out_root),
        robot_type="so_follower",
    )

    kept, dropped = [], []
    for episode in range(n):
        scene, dataset, traj = build(src_repo, src_root, episode)
        landed, _ = run_headless(scene, traj)  # pass 1: physics only
        if not landed:
            dropped.append(episode)
            print(f"ep {episode:4d}: DROP (cube never landed)", flush=True)
            continue
        acts = dataset.select_columns("action")
        actions = np.array(
            [[float(v) for v in acts[i]["action"]] for i in range(dataset.num_frames)],
            dtype=np.float32,
        )
        scene.reset()  # pass 2: re-roll deterministically, this time rendering
        for t in range(len(traj)):
            scene.step(traj[t])
            sim.add_frame(
                {
                    "action": actions[t],
                    "observation.state": traj[t].astype(np.float32),
                    "observation.images.camera1": scene.render("overhead", RENDER_HW),
                    "observation.images.camera2": scene.render("wrist", RENDER_HW),
                    "observation.images.camera3": scene.render("side", RENDER_HW),
                    "task": task,
                }
            )
        sim.save_episode(
            parallel_encoding=False
        )  # in-process: fork'd encoders abort on macOS
        kept.append(episode)
        print(f"ep {episode:4d}: kept ({len(traj)} frames)", flush=True)

    print(
        f"\nsim twin '{out_name}': kept {len(kept)}/{n}, dropped {len(dropped)}"
        f"\ndropped: {dropped}"
    )


if __name__ == "__main__":
    main()
