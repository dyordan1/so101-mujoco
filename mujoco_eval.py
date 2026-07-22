#!/usr/bin/env python
"""Headless success/fail eval over recorded episodes in the physics sim.

Replays each recorded episode (arm position-controlled to the recording, cube + tote
under physics, deterministic weld grasp) and checks whether the cube ends up released
in the tote. Every recorded attempt succeeded on the real robot, so the pass rate here
measures SIM FIDELITY, not policy skill — a failure is the sim diverging from reality.

Failures split two ways, both worth knowing:
  - welded: the cube was still gripped on the final frame (the sim never released it —
    the weld rule or contact timing differs from the real hand-off).
  - missed: the cube was released but never reached the tote floor (it slipped early,
    was dropped short, or bounced out — the interesting, non-grasp divergences).

    python mujoco_eval.py <name> [sample]   # sample>0: random subset (seed 0); else all

Runs under plain python (no viewer/GUI).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mujoco_replay import build, run_headless

SAMPLE_SEED = 0  # fixed so the random subset is reproducible run to run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", dest="repo_id", required=True)
    parser.add_argument(
        "--sample", type=int, default=0, help="random subset size (0 = every episode)"
    )
    args = parser.parse_args()

    total = json.loads((Path(args.root) / "meta/info.json").read_text())[
        "total_episodes"
    ]
    if args.sample:
        rng = np.random.default_rng(SAMPLE_SEED)
        episodes = sorted(rng.choice(total, size=args.sample, replace=False).tolist())
        print(f"eval: {args.sample} random episodes of {total} (seed {SAMPLE_SEED})\n")
    else:
        episodes = list(range(total))
        print(f"eval: all {total} episodes\n")

    passed, welded, missed = 0, [], []
    for episode in episodes:
        scene, _, traj = build(args.repo_id, args.root, episode)
        success, welded_at_end = run_headless(scene, traj)
        if success:
            passed += 1
            verdict = "PASS"
        elif welded_at_end:
            welded.append(episode)
            verdict = "FAIL (welded)"
        else:
            missed.append(episode)
            verdict = "FAIL (missed)"
        print(f"ep {episode:4d}: {verdict}", flush=True)

    n = len(episodes)
    print(f"\nPASS         : {passed}/{n} ({100 * passed / n:.0f}%)")
    print(f"FAIL welded  : {len(welded)}")
    print(f"FAIL missed  : {len(missed)}  {missed}")


if __name__ == "__main__":
    main()
