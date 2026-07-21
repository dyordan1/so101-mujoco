#!/usr/bin/env python
"""Pull a LeRobot dataset from the Hugging Face Hub into datasets/ (gitignored),
where train + mujoco_policy read it. No robot required.

    python download.py [dataset-repo-id]   # default: dobri420/pick-cube-so101-sim
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
DATASETS = Path(os.environ.get("DATASETS_DIR", HERE / "datasets"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default="dobri420/pick-cube-so101-sim")
    args = ap.parse_args()
    dest = DATASETS / args.repo.split("/")[-1]
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(dest))
    print(f"downloaded {args.repo} -> {dest}")


if __name__ == "__main__":
    main()
