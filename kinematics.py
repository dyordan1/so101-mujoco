"""SO-101 grasp forward-kinematics + grasp/release detection — the calibration-free
subset the MuJoCo sim twin needs (mujoco_env, mujoco_replay).

The FK is hardcoded from the SO-101 URDF (so101_new_calib.urdf, TheRobotStudio
SO-ARM100): recorded degrees ARE that URDF's joint angles (lerobot's calibration
convention). The grasp point is the jaw center, on motor 6's axis.
"""

import glob
import math
import os

import numpy as np

D2R = math.pi / 180

# SO-101 chain to gripper_link: (origin xyz, origin rpy) per revolute joint (all
# about local z).
CHAIN = [
    ((0.0388353, 0.0, 0.0624), (3.14159, 0.0, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ((-0.11257, -0.028, 0.0), (0.0, 0.0, 1.5708)),
    ((-0.1349, 0.0052, 0.0), (0.0, 0.0, -1.5708)),
    ((0.0, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
]

# Grasp point = the jaw center, which lies on the wrist_roll (motor 5) rotation
# axis. That axis is gripper_link's local z (the gripper rolls about it), so the
# jaw center is laterally centered at x=y=0; z runs down the axis to the jaws (the
# gripper tip sits ~9.8 cm along it). For a downward grasp the axis is vertical, so
# this x,y is the object's position regardless of the exact z.
JAW_CENTER_XYZ = (0.0, 0.0, -0.0981)

# Local +X of gripper_link points along the jaws toward the upper jaw tip — the
# direction the arrows point; it rides the wrist roll, so it shows grasp yaw.
JAW_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _origin(xyz, r):
    M = np.eye(4)
    M[:3, :3] = _rpy(*r)
    M[:3, 3] = xyz
    return M


def _rotz(q):
    M = np.eye(4)
    c, s = math.cos(q), math.sin(q)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


_ORIGINS = [_origin(*j) for j in CHAIN]
_JAW = np.array([*JAW_CENTER_XYZ, 1.0])


def _gripper_pose(joint_deg):
    """(x, y, z, R) of the jaw center in base_link: position on the wrist_roll axis,
    R the gripper_link rotation carrying the jaw-yaw axis (metres + 3x3)."""
    M = np.eye(4)
    for O, q in zip(_ORIGINS, joint_deg):
        M = M @ O @ _rotz(q * D2R)
    x, y, z = (M @ _JAW)[:3]
    return x, y, z, M[:3, :3]


def _height_cm(joint_deg):
    """Jaw-center height in cm (base_link z) — the one FK output the grasp/release
    scan needs. (Reach/azimuth need the workspace calibration grid and live in the
    plotting module; the scan only cares how high the arm is.)"""
    return _gripper_pose(joint_deg)[2] * 100


# 5 cm margins in the two units the scan uses: height is cm, _gripper_pose() is m.
_LIFT_CM = 5.0
_LIFT_M = 0.05
_JAW_MOVE = 2.0  # gripper .pos units — a real open/close, above operator jitter


def grasp_release_indices(states, weld_threshold=16.0):
    """(grasp_i, release_i) via a deterministic forward scan. The height minimum alone is
    the reach-down, where the jaw is often still open (it opens WIDE to clear the cube,
    reaches down, THEN closes) — so grasp keys off the jaw, and release off a jaw open
    that is first walked clear of the grasp. states: (T, 6) degrees. Steps:

    0. Wait for the first real jaw OPEN (>2 .pos wider, not a close/jitter) — the approach
       opens the jaw to clear the cube. Directional + thresholded: skips operator jitter and
       episodes that twitch the jaw shut first (else phase 1 latches onto the home pose).
    1. Track the arm's lowest point; stop once it climbs >5 cm above the running min.
    2. Grasp = the first frame the jaw closes past `weld_threshold` in the reach window —
       where the weld actually engages. NOT the tightest grip a few frames later: the arm is
       still moving as the jaw finishes closing, so the tightest-grip pose is shifted off the
       grab point. Keep the tightest grip separately as the held value for the release scan.
    3. Wait for the arm to move >5 cm away from the grasp location (skip the carry).
    4. Wait for the jaw to leave its held grasp value (the open at the drop begins).
    5. Track the loosest grip from there; stop once the arm strays >5 cm from the running
       max (so a later home open can't steal it). The 5 cm margins absorb open/close jitter.

    Repairs e.g. ep 497's tote landing on the grasp spot, and start-jitter mis-grasps.
    """
    states = np.asarray(states)
    n = len(states)

    def loc(i):
        return np.array(_gripper_pose(states[i][:5])[:3])

    start = next((i for i in range(n) if states[i][5] - states[0][5] > _JAW_MOVE), 0)
    low_i, low_h = start, _height_cm(states[start][:5])
    for i in range(start + 1, n):
        h = _height_cm(states[i][:5])
        if h < low_h:
            low_h, low_i = h, i
        elif h > low_h + _LIFT_CM:
            break
    tightest, min_grip = low_i, states[low_i][5]
    for i in range(low_i + 1, n):
        if _height_cm(states[i][:5]) > low_h + _LIFT_CM:
            break
        if states[i][5] < min_grip:
            min_grip, tightest = states[i][5], i
    # Grasp = the closing crossing: the start of the contiguous ≤threshold run that holds
    # the tightest grip. Walk back from the tightest grip (the crossing can precede the
    # arm's low point — the jaw shuts while the arm is still descending) to where the jaw
    # was last above the threshold. That pose is where the weld engages; min_grip (the
    # tightest grip) stays the held value for the release scan.
    grasp_i = tightest
    while grasp_i > start and states[grasp_i - 1][5] <= weld_threshold:
        grasp_i -= 1
    grasp_loc = loc(grasp_i)
    moved = next(
        (
            i
            for i in range(grasp_i + 1, n)
            if np.linalg.norm(loc(i) - grasp_loc) > _LIFT_M
        ),
        n - 1,
    )
    rel_start = next(
        (i for i in range(moved, n) if states[i][5] - min_grip > _JAW_MOVE), moved
    )
    release_i, max_grip, anchor = rel_start, states[rel_start][5], loc(rel_start)
    for i in range(rel_start + 1, n):
        li = loc(i)
        if np.linalg.norm(li - anchor) > _LIFT_M:
            break
        if states[i][5] >= max_grip:
            max_grip, anchor, release_i = states[i][5], li, i
    return grasp_i, release_i


def grasp_frame(states, weld_threshold=16.0):
    """The 5-joint state where the jaw closes past the weld threshold. states: (T, 6)."""
    states = np.asarray(states)
    return states[grasp_release_indices(states, weld_threshold)[0]][:5]


def release_frame(states, weld_threshold=16.0):
    """The 5-joint state at the release: widest jaw opening after the grasp. states: (T, 6)."""
    states = np.asarray(states)
    return states[grasp_release_indices(states, weld_threshold)[1]][:5]


def episode_distractors(dataset_dir):
    """{episode_index: distractor_count} from meta/episodes/*.parquet, or {} if the
    dataset doesn't carry the column (older datasets, or freshly recorded episodes
    not yet annotated). Null counts read as 0."""
    import pyarrow.parquet as pq

    out = {}
    pattern = os.path.join(dataset_dir, "meta", "episodes", "*", "*.parquet")
    for f in sorted(glob.glob(pattern)):
        if "distractor_count" not in pq.read_schema(f).names:
            continue
        t = pq.read_table(f, columns=["episode_index", "distractor_count"])
        for e, c in zip(
            t.column("episode_index").to_pylist(),
            t.column("distractor_count").to_pylist(),
        ):
            out[int(e)] = int(c) if c is not None else 0
    return out
