#!/usr/bin/env python
"""The MuJoCo SO-101 pick-cube environment — owns ALL scene construction and physics.

Nothing else builds geoms, sets friction/collision masks, welds, or steps the sim: it all
lives here. Consumers (mujoco_replay, mujoco_policy) only decide WHERE the cube and tote go
and WHAT joint targets to feed each frame — placement + control are external, the sim is
this module's alone.

Public surface:
    build_robot(joints) -> Robot        # robot spec + FK model + .pos<->qpos machinery
    gripper_pose(model, arm_deg, off)   # world grasp point for an arm state (FK)
    calibration_offsets(joints)         # per-joint .pos -> model-frame offset
    Scene(joints, cube_xy, cube_yaw, tote_xy, home_deg, fps, robot=None)
        .reset() / .step(action_deg) / .pos_state() / .render(cam, hw)
        .welded / .landed  + .model .data for viewers

The grasp is a deterministic weld: the cube welds on a 2-point (face) contact with both
finger-only hulls pressing OPPOSING cube faces (never adjacent — see WELD_OPPOSING_COS)
while the gripper is closed past GRIP_WELD_MAX, and — once welded — stays
welded until the gripper opens back past it (weld latch: contact ignored while gripped, so a
firm grip survives a transient blip when the arm jerks). The threshold sits in the clean gap
between the tightest recorded grasp and the loosest release. The cube is 4 mm under the real
3 cm (compliance the rigid weld omits) and lightly lubed so it drops clean on release.
"""

import json
import math
import os
import struct
import sys
from collections import namedtuple
from pathlib import Path

# Headless Linux (GPU nodes) has no X display, so MuJoCo's default GLFW backend
# fails `gladLoadGL` the moment anything renders (the policy sweep / camera grid).
# Select the offscreen EGL backend automatically — set before `import mujoco`,
# guarded so an explicit MUJOCO_GL wins and the macOS interactive viewer (Cocoa,
# no DISPLAY) is untouched. Reproducible: no env var to remember at launch.
if sys.platform == "linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from kinematics import JAW_AXIS_LOCAL
from robot_descriptions import so_arm101_mj_description

OBS_STATE = "observation.state"
GRIPPER = "gripper"
GRIPPER_SITE = "gripper"  # jaw-center site on the wrist-roll axis == the grasp point
GRIPPER_ACTUATOR = "6"  # the gripper's actuator/joint name in the MJCF
# Both jaws' stock meshes carry chunky bodies behind the fingertip; their convex hulls
# register grasp contact (and spurious roll-off contact) against the body, not the
# fingertip. We disable each stock collider and collide against a hand-cleaned finger-only
# hull in the same coordinate frame instead.
MOVING_JAW = "moving_jaw_so101_v1"  # upper (actuated) jaw mesh
MOVING_FINGER = "moving_finger"
MOVING_FINGER_STL = (
    Path(__file__).resolve().parent / "assets/so101_moving_jaw_finger.stl"
)
FIXED_JAW_MESH = "wrist_roll_follower_so101_v1"  # lower jaw: L-shaped palm + finger
FIXED_FINGER = "fixed_finger"
FIXED_FINGER_STL = (
    Path(__file__).resolve().parent / "assets/so101_fixed_jaw_finger.stl"
)

# Wrist camera mount — a rigid gripper payload in the jaw frame; camera sits
# CAM_MOUNT_FORWARD in front of its face (4 CAD corners below), rolled to the real mount.
WRIST_MOUNT_STL = (
    Path(__file__).resolve().parent / "assets/so101_wrist_cam_mount.stl"
)
WRIST_MOUNT_MASS = 0.02
CAM_MOUNT_FORWARD = 0.02
CAM_FACE_CORNERS = np.array(
    [
        [0.021672, -0.053706, -0.008952],
        [-0.014328, -0.053794, -0.008833],
        [-0.014202, -0.085047, 0.005777],
        [0.021797, -0.084959, 0.005657],
    ]
)

CALIBRATION = Path(__file__).resolve().parent / "calib/so101_robot.json"
ENCODER_RES = 4096  # STS3215 12-bit encoder (lerobot feetech table)

CUBE_EDGE = 0.026  # m — 4 mm under the real 3 cm (stands in for grasp compliance)
CUBE_RGBA = [0.365, 0.439, 0.788, 1.0]  # #5D70C9
GRASP_FORWARD = 0.015  # grasp point sits 1.5 cm forward of the jaw-center site
DESK_SIZE = (1.00, 0.60, 0.025)  # m — long side (robot centred on it), depth, thickness
DESK_RGBA = [0.88, 0.83, 0.74, 1.0]  # pale light wood, matching the real desk
# Base plate world-axis-aligned: robot faces -Y, desk/tote extend -Y, long side along X.
# Plate mounts at world z=+0.030 (desk top + cube rest here), back wall y=+0.038, x=+0.021.
MOUNT_Z = 0.030
BASE_BACK_Y = 0.038
BASE_X = 0.021
ROBOT_RGBA = [1.0, 1.0, 1.0, 1.0]  # solid white
COLLISION_GROUP = 3  # model collision meshes; hidden in the viewer, active for physics

TOTE_OUTER = (0.125, 0.085, 0.043)  # m — outer L×W×H of the open-top box
TOTE_WALL = 0.003
TOTE_RGBA = [1.0, 1.0, 1.0, 1.0]

# Distractor clutter — blue objects at ~the cube's bounding-box volume (2.5–4 cm). Cylinder
# is a primitive; torus + monke are meshes (collision falls back to their convex hull).
DISTRACTOR_CYLINDER, DISTRACTOR_TORUS, DISTRACTOR_MONKE = "cylinder", "torus", "monke"
DISTRACTOR_KINDS = (DISTRACTOR_CYLINDER, DISTRACTOR_TORUS, DISTRACTOR_MONKE)
DISTRACTOR_RGBA = CUBE_RGBA  # same colour as the cube (#5D70C9)
DISTRACTOR_FRICTION = [1.0, 0.05, 0.01]
DISTRACTOR_REACH_CM = (15.0, 30.0)  # placed in the cube's polar fan off the pan axis
DISTRACTOR_AZIM_DEG = 90.0
CUBE_SAFE_RADIUS = 0.03  # 6 cm no-distractor zone around the cube
DISTRACTOR_SPACING = 0.045  # min gap between distractors
CYL_R, CYL_HALFH = 0.013, 0.013  # bbox 2.6³ cm, == the cube
TORUS_R, TORUS_TUBE = (
    0.012,
    0.0065,
)  # outer Ø 3.7 cm, height 1.3 cm, bbox vol ≈ the cube
MONKE_STL = Path(__file__).resolve().parent / "assets/monke.stl"
_DISTRACTOR_CACHE = {}

# Backdrop walls (white room): front of the robot (far -Y edge) + its left (+X), ground-to-
# 9ft. Static, non-colliding — a white camera background. Desk top is DESK_HEIGHT above the
# ground; sim desk top is MOUNT_Z, so the ground is MOUNT_Z - DESK_HEIGHT.
WALL_RGBA = [0.93, 0.93, 0.93, 1.0]
WALL_HEIGHT = 2.7
WALL_THICK = 0.05
DESK_HEIGHT = 0.745
WALL_FRONT_GAP = 0.02
WALL_LEFT_GAP = 0.18
# Headlight ambient+diffuse: moderate, so the pale desk reads as light wood without
# blowing out to flat white (matching the 0.83 mean directly saturated the desk). Ambient
# kept low (a high flat wash was the blowout); diffuse gives directional contrast.
HEADLIGHT_AMBIENT = 0.35
HEADLIGHT_DIFFUSE = 0.6

# Collision masks: arm links hit objects + desk but not each other; objects + desk hit all.
ARM_CONTYPE, ARM_CONAFFINITY = 1, 2
OBJ_CONTYPE, OBJ_CONAFFINITY = 2, 3
GRIP_FRICTION = [2.0, 0.05, 0.01]  # slide/spin/roll on the arm
# The weld holds now, so the cube's own friction is only in the way — kept low (with higher
# contact priority so it wins the cube↔jaw contact) so the cube drops clean on release.
CUBE_FRICTION = [0.6, 0.005, 0.0001]
CUBE_PRIORITY = 1
GRIP_KP, GRIP_FORCE = 50.0, 5.0  # gripper actuator stiffness + force cap (N·m)
GRIP_WELD_MAX = 16.0  # weld active only while the gripper .pos is at/below this
# The two jaws must pinch OPPOSING cube faces, not adjacent ones: require their
# pressed-face outward normals to point apart by ≥120° (dot ≤ this). A true
# parallel-jaw pinch is ~180° (dot ≈ -1); adjacent faces are ~90° (dot ≈ 0).
WELD_OPPOSING_COS = -0.5
SHOW_GRASP_MARKER = False  # debug: green sphere at the cube-spawn point on the gripper

# Camera vertical FOV (deg) matched to hardware: recorded 640×480 is a center-crop of the
# native 16:9, keeping the 16:9 vFOV. scene = 85° diag → 48.4; wrist = 70° diag → 37.9.
SCENE_FOVY = 48.4
WRIST_FOVY = 37.9
CAM_OVERHEAD = (
    [-0.021, -0.439, 0.805],
    [0.019, -0.222, -0.170],
    12.33,
)  # (pos, look-at, roll)
CAM_SIDE = ([-0.745, -0.379, 0.750], [-0.019, -0.347, 0.063], -2.22)
# (label, sim camera, real dataset image key), display order.
CAM_GRID = (
    ("top", "overhead", "observation.images.camera1"),
    ("wrist", "wrist", "observation.images.camera2"),
    ("side", "side", "observation.images.camera3"),
)
CAM_TILE = (240, 320)  # per-camera (height, width) for a grid tile
CUBE_DROP, TOTE_DROP = 0.005, 0.001  # start objects just above the desk so they settle
# Real empty-scene photos composited behind these cams (green-screen), matching the room.
_ASSETS = Path(__file__).resolve().parent / "assets"
BACKDROP = {
    "overhead": _ASSETS / "backdrop_top.png",
    "side": _ASSETS / "backdrop_side.png",
}

Robot = namedtuple(
    "Robot",
    "spec model joints grip_i grip_lo grip_hi offsets to_target pan_xy fixed_jaw",
)


# ---------------------------------------------------------------- construction helpers ---
def yaw_quat(a: float) -> list[float]:
    """MuJoCo (w,x,y,z) quaternion for a rotation of `a` about world z."""
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def polar_xy(pan_xy, reach_cm, azim_deg):
    """World (x, y) of a point at (reach, azimuth) off the shoulder-pan axis: azimuth 0 =
    straight ahead (robot faces -Y), + = the robot's right (-X). The workspace polar frame
    the cube and distractors are placed in."""
    az = math.radians(azim_deg)
    r = reach_cm / 100.0
    return (pan_xy[0] - r * math.sin(az), pan_xy[1] - r * math.cos(az))


def _look_at_xyaxes(pos, target, roll=0.0):
    forward = np.array(target, float) - np.array(pos, float)
    forward /= np.linalg.norm(forward)
    cam_z = -forward  # MuJoCo cameras look down their -Z
    up = np.array([0.0, 0.0, 1.0])
    if abs(up @ cam_z) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    cam_x = np.cross(up, cam_z)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    if roll:  # rotate the image axes about the view axis (roll degrees, CCW)
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        cam_x, cam_y = c * cam_x + s * cam_y, -s * cam_x + c * cam_y
    return [*cam_x, *cam_y]


def _add_scene_camera(body, name, pos_target):
    cam = body.add_camera(
        name=name, pos=pos_target[0], xyaxes=_look_at_xyaxes(*pos_target)
    )
    cam.fovy = SCENE_FOVY


def _load_stl(path):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    tris = np.frombuffer(
        raw[84 : 84 + 50 * n],
        dtype=np.dtype([("f", "<f4", (12,)), ("attr", "<u2")]),
        count=n,
    )
    corners = tris["f"].reshape(n, 4, 3)[:, 1:, :].reshape(-1, 3)
    verts, inverse = np.unique(corners, axis=0, return_inverse=True)
    return verts, inverse.reshape(-1).reshape(n, 3)


def _add_finger(spec, body_name, geom_name, stl_path, pos, quat):
    verts, faces = _load_stl(stl_path)
    mesh = spec.add_mesh()
    mesh.name = geom_name
    mesh.uservert = verts.flatten().tolist()
    mesh.userface = faces.flatten().tolist()
    body = next(b for b in spec.bodies if b.name == body_name)
    body.add_geom(
        name=geom_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=geom_name,
        pos=pos,
        quat=quat,
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
        friction=GRIP_FRICTION,
        mass=0,  # duplicate contact surface; mass stays on the stock visual geom
        group=COLLISION_GROUP,
    )


def _add_wrist_mount(spec, pos, quat, up_local, grasp_local):
    verts, faces = _load_stl(WRIST_MOUNT_STL)
    mesh = spec.add_mesh()
    mesh.name = "wrist_mount"
    mesh.uservert = verts.flatten().tolist()
    mesh.userface = faces.flatten().tolist()
    gripper = next(b for b in spec.bodies if b.name == GRIPPER)
    gripper.add_geom(
        name="wrist_mount",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="wrist_mount",
        pos=pos,
        quat=quat,
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
        friction=GRIP_FRICTION,
        mass=WRIST_MOUNT_MASS,
        rgba=ROBOT_RGBA,
    )
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.array(quat, float))
    corners = CAM_FACE_CORNERS @ rot.reshape(3, 3).T + np.array(pos)
    normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    normal /= np.linalg.norm(normal)
    if normal @ (grasp_local - corners.mean(0)) < 0:
        normal = -normal  # forward = toward the jaws
    cam_pos = corners.mean(0) + CAM_MOUNT_FORWARD * normal
    right = corners[1] - corners[0]
    right /= np.linalg.norm(right)
    up = np.cross(-normal, right)
    up /= np.linalg.norm(up)
    if up @ up_local < 0:
        up, right = -up, -right
    right, up = up, -right  # roll 90° to the camera's mounted orientation
    cam = gripper.add_camera(name="wrist", pos=cam_pos.tolist(), xyaxes=[*right, *up])
    cam.fovy = WRIST_FOVY


def _add_grasp_marker(spec, grasp_local):
    gripper = next(b for b in spec.bodies if b.name == GRIPPER)
    gripper.add_geom(
        name="grasp_marker",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.005, 0.0, 0.0],
        pos=list(grasp_local),
        rgba=[0.1, 1.0, 0.2, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.0,
    )


def _add_desk(spec):
    # Visible for now: with the desk drawn over the composited backdrop photo, its edges
    # show whether the camera pose/FOV lines up with the real desk. Hide it (group=
    # COLLISION_GROUP) once aligned, so the backdrop's own desk takes over.
    spec.worldbody.add_geom(
        name="desk",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BASE_X, BASE_BACK_Y - DESK_SIZE[1] / 2, MOUNT_Z - DESK_SIZE[2] / 2],
        size=[DESK_SIZE[0] / 2, DESK_SIZE[1] / 2, DESK_SIZE[2] / 2],
        rgba=DESK_RGBA,
        contype=OBJ_CONTYPE,
        conaffinity=OBJ_CONAFFINITY,
    )


def _add_walls(spec):
    cz = (MOUNT_Z - DESK_HEIGHT) + WALL_HEIGHT / 2
    y_far = BASE_BACK_Y - DESK_SIZE[1]  # far (front) desk edge, -Y
    x_left = BASE_X + DESK_SIZE[0] / 2  # left desk edge, +X
    for name, size, pos in (
        (
            "wall_front",
            [DESK_SIZE[0] / 2 + 0.3, WALL_THICK / 2, WALL_HEIGHT / 2],
            [BASE_X, y_far - WALL_FRONT_GAP - WALL_THICK / 2, cz],
        ),
        (
            "wall_left",  # ~5 desk depths long
            [WALL_THICK / 2, 2.5 * DESK_SIZE[1], WALL_HEIGHT / 2],
            [
                x_left + WALL_LEFT_GAP + WALL_THICK / 2,
                BASE_BACK_Y - DESK_SIZE[1] / 2,
                cz,
            ],
        ),
    ):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=WALL_RGBA,
            contype=0,
            conaffinity=0,
        )


def _add_tote(spec, x, y):
    length, width, height = TOTE_OUTER
    t = TOTE_WALL
    body = spec.worldbody.add_body(name="tote", pos=[x, y, MOUNT_Z + TOTE_DROP])
    body.add_freejoint()

    def slab(name, size, pos):
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=TOTE_RGBA,
            contype=OBJ_CONTYPE,
            conaffinity=OBJ_CONAFFINITY,
        )

    # Floor nested between the walls (|_|), so the only exposed floor face is the interior
    # top: a cube nudged against the outside hits a wall, so a cube↔floor contact = inside.
    slab("tote_floor", [length / 2 - t, width / 2 - t, t / 2], [0, 0, t / 2])
    slab("tote_x+", [t / 2, width / 2, height / 2], [length / 2 - t / 2, 0, height / 2])
    slab(
        "tote_x-",
        [t / 2, width / 2, height / 2],
        [-(length / 2 - t / 2), 0, height / 2],
    )
    slab(
        "tote_y+",
        [length / 2 - t, t / 2, height / 2],
        [0, width / 2 - t / 2, height / 2],
    )
    slab(
        "tote_y-",
        [length / 2 - t, t / 2, height / 2],
        [0, -(width / 2 - t / 2), height / 2],
    )


def _add_cube(spec, cube_pos, cube_yaw):
    cube = spec.worldbody.add_body(name="cube", pos=cube_pos, quat=yaw_quat(cube_yaw))
    cube.add_freejoint()
    cube.add_geom(
        name="grasp_cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_EDGE / 2] * 3,
        rgba=CUBE_RGBA,
        contype=OBJ_CONTYPE,
        conaffinity=OBJ_CONAFFINITY,
        friction=CUBE_FRICTION,
        priority=CUBE_PRIORITY,
    )


def _torus_mesh(major=48, minor=24):
    """Vertices + triangle faces of a torus (major radius TORUS_R, tube TORUS_TUBE)."""
    u = np.linspace(0, 2 * math.pi, major, endpoint=False)
    v = np.linspace(0, 2 * math.pi, minor, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = TORUS_R + TORUS_TUBE * np.cos(vv)
    verts = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), TORUS_TUBE * np.sin(vv)], -1
    )
    verts = verts.reshape(-1, 3)
    faces = []
    for i in range(major):
        for j in range(minor):
            a, d = i * minor + j, i * minor + (j + 1) % minor
            b, c = ((i + 1) % major) * minor + j, ((i + 1) % major) * minor + (
                j + 1
            ) % minor
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


def _distractor_geometry(kind):
    """(verts, faces, half_height) for a mesh distractor, scaled to the cube's bbox volume
    and centred on the origin so it rests flat. Cached; DISTRACTOR_CYLINDER is a primitive.
    """
    cached = _DISTRACTOR_CACHE.get(kind)
    if cached is not None:
        return cached
    if kind == DISTRACTOR_TORUS:
        verts, faces = _torus_mesh()
    elif kind == DISTRACTOR_MONKE:
        verts, faces = _load_stl(MONKE_STL)
        bb = verts.max(0) - verts.min(0)
        verts = (verts - (verts.max(0) + verts.min(0)) / 2) * (
            CUBE_EDGE**3 / (bb[0] * bb[1] * bb[2])
        ) ** (1 / 3)
    else:
        raise ValueError(kind)
    out = _DISTRACTOR_CACHE[kind] = (verts, faces, float(verts[:, 2].max()))
    return out


def _add_distractor(spec, name, kind, x, y, yaw):
    """A free-body clutter object (blue) resting on the desk at (x, y). Returns its init
    freejoint qpos (pos + quat) so reset() can reseat it."""
    if kind == DISTRACTOR_CYLINDER:
        half_h = CYL_HALFH
        geom = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CYL_R, CYL_HALFH])
    else:
        verts, faces, half_h = _distractor_geometry(kind)
        mesh = spec.add_mesh()
        mesh.name = f"{name}_mesh"
        mesh.uservert = verts.flatten().tolist()
        mesh.userface = faces.flatten().tolist()
        geom = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh")
    pos = [x, y, MOUNT_Z + half_h + CUBE_DROP]
    body = spec.worldbody.add_body(name=name, pos=pos, quat=yaw_quat(yaw))
    body.add_freejoint()
    body.add_geom(
        name=f"{name}_g",
        rgba=DISTRACTOR_RGBA,
        contype=OBJ_CONTYPE,
        conaffinity=OBJ_CONAFFINITY,
        friction=DISTRACTOR_FRICTION,
        **geom,
    )
    return [*pos, *yaw_quat(yaw)]


def sample_distractors(count, cube_xy, pan_xy, rng):
    """`count` distractors at random (reach, azimuth) in the cube's fan, avoiding a 6 cm
    zone around the cube and each other. Distinct kinds. Returns [(kind, x, y, yaw)]."""
    kinds = list(DISTRACTOR_KINDS)
    rng.shuffle(kinds)
    placed = []
    for kind in kinds[:count]:
        for _ in range(200):
            x, y = polar_xy(
                pan_xy,
                rng.uniform(*DISTRACTOR_REACH_CM),
                rng.uniform(-DISTRACTOR_AZIM_DEG, DISTRACTOR_AZIM_DEG),
            )
            if math.hypot(x - cube_xy[0], y - cube_xy[1]) < CUBE_SAFE_RADIUS:
                continue
            if any(math.hypot(x - p[1], y - p[2]) < DISTRACTOR_SPACING for p in placed):
                continue
            placed.append((kind, x, y, rng.uniform(0, 2 * math.pi)))
            break
    return placed


# --------------------------------------------------------------------- public FK / build ---
def calibration_offsets(joints):
    """Degrees to add to each recorded .pos to reach the model's joint frame. LeRobot's
    .pos is centred on the calibration midpoint; the model's qpos=0 is the homing reference
    (raw=res/2), so the fixed offset is (mid - res/2)·360/(res-1). Gripper offset is 0.
    """
    cal = json.loads(CALIBRATION.read_text())
    out = []
    for joint in joints:
        if joint == GRIPPER:
            out.append(0.0)
            continue
        c = cal[joint]
        mid = (c["range_min"] + c["range_max"]) / 2
        out.append((mid - ENCODER_RES / 2) * 360 / (ENCODER_RES - 1))
    return out


def gripper_pose(model, arm_deg, offsets):
    """World (x, y, z, yaw) of the grasp point for a 5-joint arm state (recorded degrees):
    the jaw-center site pushed GRASP_FORWARD along the jaws. FK only — no scene needed.
    """
    data = mujoco.MjData(model)
    for i, deg in enumerate(arm_deg):
        data.qpos[i] = math.radians(float(deg) + offsets[i])
    mujoco.mj_forward(model, data)
    jaw_world = data.body(GRIPPER).xmat.reshape(3, 3) @ JAW_AXIS_LOCAL
    x, y, z = data.site(GRIPPER_SITE).xpos + GRASP_FORWARD * jaw_world
    return float(x), float(y), float(z), math.atan2(jaw_world[1], jaw_world[0])


def build_robot(joints):
    """Robot spec + first compile: white paint, finger-only jaw colliders, tuned gripper
    actuator, headlight. Returns a Robot: the mutable spec (Scene finishes it), the FK model,
    the .pos↔qpos machinery, and the shoulder-pan axis xy (for reach/azim placement)."""
    spec = mujoco.MjSpec.from_file(so_arm101_mj_description.MJCF_PATH)
    spec.visual.headlight.ambient = [HEADLIGHT_AMBIENT] * 3
    spec.visual.headlight.diffuse = [HEADLIGHT_DIFFUSE] * 3
    spec.visual.global_.offwidth = 1280  # offscreen framebuffer for the larger renders
    spec.visual.global_.offheight = 960
    fixed_jaw = moving_jaw = None
    for geom in spec.geoms:
        geom.material = ""
        geom.rgba = ROBOT_RGBA
        if geom.classname and geom.classname.name == "collision":
            geom.contype, geom.conaffinity = ARM_CONTYPE, ARM_CONAFFINITY
            geom.friction = GRIP_FRICTION
            if geom.meshname == FIXED_JAW_MESH:
                geom.contype, geom.conaffinity = 0, 0
                fixed_jaw = (list(geom.pos), list(geom.quat))
            elif geom.meshname == MOVING_JAW:
                geom.contype, geom.conaffinity = 0, 0
                moving_jaw = (list(geom.pos), list(geom.quat))
    for actuator in spec.actuators:
        if actuator.name == GRIPPER_ACTUATOR:
            actuator.gainprm[0] = GRIP_KP
            actuator.biasprm[1] = -GRIP_KP
            actuator.forcerange = [-GRIP_FORCE, GRIP_FORCE]
    _add_finger(spec, GRIPPER, FIXED_FINGER, FIXED_FINGER_STL, *fixed_jaw)
    _add_finger(spec, MOVING_JAW, MOVING_FINGER, MOVING_FINGER_STL, *moving_jaw)
    model = spec.compile()
    if model.njnt != len(joints):
        raise SystemExit(f"model has {model.njnt} joints but {len(joints)} given")
    grip_i = joints.index(GRIPPER)
    grip_lo, grip_hi = (float(x) for x in model.jnt_range[grip_i])
    offsets = calibration_offsets(joints)

    def to_target(i, value):
        if i == grip_i:
            return grip_lo + (value / 100.0) * (grip_hi - grip_lo)
        return math.radians(value + offsets[i])

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    pan_xy = tuple(float(v) for v in d.xanchor[0][:2])  # shoulder-pan axis, world xy
    return Robot(
        spec,
        model,
        joints,
        grip_i,
        grip_lo,
        grip_hi,
        offsets,
        to_target,
        pan_xy,
        fixed_jaw,
    )


class Scene:
    """A compiled pick-cube scene + its physics. Placement (cube_xy, cube_yaw, tote_xy) and
    control (step's action) come from the caller; construction, the weld rule, stepping and
    rendering are the env's. `robot` reuses a build_robot() result to skip a recompile.
    """

    def __init__(
        self,
        joints,
        cube_xy,
        cube_yaw,
        tote_xy,
        home_deg,
        fps=30,
        robot=None,
        distractors=(),
    ):
        robot = robot or build_robot(joints)
        spec = robot.spec
        self.joints, self.grip_i = robot.joints, robot.grip_i
        self.grip_lo, self.grip_hi, self.offsets = (
            robot.grip_lo,
            robot.grip_hi,
            robot.offsets,
        )
        self._to_target = robot.to_target
        self.home_deg = [float(v) for v in home_deg]
        cube_pos = [cube_xy[0], cube_xy[1], MOUNT_Z + CUBE_EDGE / 2 + CUBE_DROP]
        self._cube_init = [*cube_pos, *yaw_quat(cube_yaw)]
        self._tote_init = [
            tote_xy[0],
            tote_xy[1],
            MOUNT_Z + TOTE_DROP,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        rest = mujoco.MjData(
            robot.model
        )  # gripper rest orientation for the wrist cam's up
        for i in range(len(joints)):
            rest.qpos[i] = self._to_target(i, self.home_deg[i])
        mujoco.mj_forward(robot.model, rest)
        up_local = rest.body(GRIPPER).xmat.reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        _add_wrist_mount(
            spec, *robot.fixed_jaw, up_local, robot.model.site(GRIPPER_SITE).pos
        )
        if SHOW_GRASP_MARKER:
            rot = rest.body(GRIPPER).xmat.reshape(3, 3)
            grasp_local = rot.T @ (
                rest.site(GRIPPER_SITE).xpos - rest.body(GRIPPER).xpos
            ) + GRASP_FORWARD * np.array(JAW_AXIS_LOCAL)
            _add_grasp_marker(spec, grasp_local)
        _add_desk(spec)
        _add_tote(spec, *tote_xy)
        _add_cube(spec, cube_pos, cube_yaw)
        distractor_inits = [
            (f"distractor{i}", _add_distractor(spec, f"distractor{i}", kind, x, y, yaw))
            for i, (kind, x, y, yaw) in enumerate(distractors)
        ]
        _add_scene_camera(spec.worldbody, "overhead", CAM_OVERHEAD)
        _add_scene_camera(spec.worldbody, "side", CAM_SIDE)
        weld = spec.add_equality()
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1, weld.name2, weld.active = GRIPPER, "cube", False

        self.model = spec.compile()
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self._substeps = round((1 / fps) / self.model.opt.timestep)
        self._grip_bid = self.model.body(GRIPPER).id
        self._cube_bid = self.model.body("cube").id
        self._upper = self.model.geom(MOVING_FINGER).id
        self._lower = self.model.geom(FIXED_FINGER).id
        self._cube_geom = next(
            g
            for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self._cube_bid
        )
        self._floor = self.model.geom("tote_floor").id
        self._grip_qadr = self.model.jnt_qposadr[self.grip_i]
        self._cube_qadr = self.model.jnt_qposadr[self.model.body("cube").jntadr[0]]
        self._tote_qadr = self.model.jnt_qposadr[self.model.body("tote").jntadr[0]]
        self._distractor_inits = [
            (self.model.jnt_qposadr[self.model.body(name).jntadr[0]], init)
            for name, init in distractor_inits
        ]
        self._renderers = {}
        self.reset()

    def place_distractors(self, placements):
        """Move the compiled distractor bodies to new (x, y, yaw); their kinds stay as
        built (geometry is baked at compile). Updates the stored init so a following
        reset() seats them there. `placements` must match the distractor count."""
        self._distractor_inits = [
            (qadr, [x, y, init[2], *yaw_quat(yaw)])
            for (qadr, init), (x, y, yaw) in zip(self._distractor_inits, placements)
        ]

    def reset(self):
        """Home the arm, put the cube + tote at their start, clear weld + landed state."""
        mujoco.mj_resetData(self.model, self.data)
        for i in range(len(self.joints)):
            self.data.qpos[i] = self.data.ctrl[i] = self._to_target(i, self.home_deg[i])
        self.data.qpos[self._cube_qadr : self._cube_qadr + 7] = self._cube_init
        self.data.qpos[self._tote_qadr : self._tote_qadr + 7] = self._tote_init
        for qadr, init in self._distractor_inits:
            self.data.qpos[qadr : qadr + 7] = init
        mujoco.mj_forward(self.model, self.data)
        self.welded, self._weld_jaw_pos, self.landed = False, 0.0, False

    def pos_state(self):
        """The 6 joint .pos (arm degrees, gripper 0-100) — inverse of the control mapping."""
        out = []
        for i in range(len(self.joints)):
            q = float(self.data.qpos[i])
            if i == self.grip_i:
                out.append((q - self.grip_lo) / (self.grip_hi - self.grip_lo) * 100)
            else:
                out.append(math.degrees(q) - self.offsets[i])
        return out

    def step(self, action_deg):
        """Drive one frame to `action_deg` (6 .pos) under physics + the weld latch. The
        gripper .pos gates the weld: closed past the threshold can grip, opening past it
        releases. Sets self.landed once the released cube rests on the tote floor."""
        gripping = float(action_deg[self.grip_i]) <= GRIP_WELD_MAX
        target = [
            self._to_target(i, float(action_deg[i])) for i in range(len(self.joints))
        ]
        for _ in range(self._substeps):
            for i in range(len(self.joints)):
                self.data.ctrl[i] = target[i]
            if (
                self.welded
            ):  # hold the jaw at its grip point so it can't sink into the cube
                self.data.ctrl[self.grip_i] = max(
                    target[self.grip_i], self._weld_jaw_pos
                )
            mujoco.mj_step(self.model, self.data)
            if not self.welded and gripping:
                cube_c = self.data.geom_xpos[self._cube_geom]
                up_n, up_c = _cube_face_normal(
                    self.data, self._cube_geom, self._upper, cube_c
                )
                lo_n, lo_c = _cube_face_normal(
                    self.data, self._cube_geom, self._lower, cube_c
                )
                # weld only on a real pinch: ≥2 points on one jaw, ≥1 on the other,
                # and the two jaws pressing OPPOSING faces (not adjacent — a corner)
                if up_c >= 2 and lo_c >= 1 and up_n @ lo_n <= WELD_OPPOSING_COS:
                    _weld_cube(self.model, self.data, self._grip_bid, self._cube_bid)
                    self._weld_jaw_pos = float(self.data.qpos[self._grip_qadr])
                    self.welded = True
            elif not gripping:  # weld latch: only the gripper opening releases
                self.data.eq_active[0], self.welded = 0, False
            if (
                not self.welded
                and not self.landed
                and _touching(self.data, self._cube_geom, self._floor)
            ):
                self.landed = True

    def render(self, cam, hw, qpos=None):
        """RGB render of a named camera at (height, width) hw, collision hull group hidden.
        `qpos` re-poses the scene first."""
        if qpos is not None:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
        r = self._renderers.get(hw)
        if r is None:
            r = self._renderers[hw] = mujoco.Renderer(self.model, hw[0], hw[1])
        opt = mujoco.MjvOption()
        opt.geomgroup[COLLISION_GROUP] = 0
        r.update_scene(self.data, camera=cam, scene_option=opt)
        return r.render()


def _contact_points(data, geom_a, geom_b):
    n = 0
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        if (g1 == geom_a and g2 == geom_b) or (g1 == geom_b and g2 == geom_a):
            n += 1
    return n


def _touching(data, geom_a, geom_b):
    return _contact_points(data, geom_a, geom_b) > 0


def _cube_face_normal(data, cube_geom, finger_geom, cube_center):
    """(mean outward face normal, contact count) for the cube↔finger contacts.
    Each contact normal is flipped to point from the cube center toward the
    contact, so it reads as the pressed cube face's outward normal — regardless
    of MuJoCo's geom-order sign convention. Returns (None, 0) if they don't touch."""
    acc, n = np.zeros(3), 0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if not (
            (g1 == cube_geom and g2 == finger_geom)
            or (g1 == finger_geom and g2 == cube_geom)
        ):
            continue
        normal = con.frame[:3].copy()
        if normal @ (con.pos - cube_center) < 0:
            normal = -normal
        acc += normal
        n += 1
    if n == 0:
        return None, 0
    return acc / np.linalg.norm(acc), n


def _weld_cube(model, data, grip_bid, cube_bid):
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, data.xquat[grip_bid])
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, data.xpos[cube_bid] - data.xpos[grip_bid], neg)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, neg, data.xquat[cube_bid])
    model.eq_data[0][:3] = 0
    model.eq_data[0][3:6] = relpos
    model.eq_data[0][6:10] = relquat
    model.eq_data[0][10] = 1
    data.eq_active[0] = 1
