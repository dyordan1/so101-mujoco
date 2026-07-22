#!/usr/bin/env python
"""MJX (MuJoCo-XLA) physics backend for the SO-101 pick-cube twin.

A parity-first port of mujoco_env.Scene's physics: same compiled model, same
contact-gated weld + latch, driven through mjx.step instead of mj_step. The goal
of this module is to answer "does GPU-capable MJX physics reproduce the C-MuJoCo
fidelity/sweep numbers" before investing in a fully in-graph vectorized rollout.

Scope (deliberate): SINGLE env, host-side weld gate. Each substep jits mjx.step,
then reads contacts back to the host to run the exact same gate as Scene.step
(≥2 points on the moving jaw, ≥1 on the fixed, opposing face normals) and toggles
the weld equality. This is CPU-slow (fine for a laptop parity check); the batched,
in-graph gate needed for the GPU-node speedup is a follow-on once parity holds.
"""

import numpy as np
import mujoco
from mujoco import mjx
import jax

import mujoco_env as E


class MjxScene:
    """Wraps a compiled Scene's model with an MJX rollout that mirrors Scene.step.

    Reuses the C-MuJoCo `scene` for construction (model, geom/body ids, control
    mapping, initial qpos) so the ONLY thing that changes is the physics engine.
    """

    def __init__(self, scene):
        self._s = scene
        self._mx = mjx.put_model(scene.model)
        self._step = jax.jit(mjx.step)
        self._dx = None

    def reset(self):
        """Home the arm + reseat cube/tote/distractors via the C scene, then lift that
        exact state into MJX. Clears the weld latch."""
        self._s.reset()
        self._dx = mjx.put_data(self._s.model, self._s.data)
        self.welded, self._weld_jaw_pos, self.landed = False, 0.0, False

    def _contacts(self):
        """Host-side snapshot of the active contacts: (geom pairs Nx2, normals Nx3,
        positions Nx3). Active = penetrating (dist < 0), matching mj_step's ncon set."""
        ct = self._dx._impl.contact
        dist = np.asarray(ct.dist)
        act = dist < 0.0
        geom = np.asarray(ct.geom)[act]
        # frame row 0 is the contact normal, same convention as C-MuJoCo con.frame[:3]
        normal = np.asarray(ct.frame)[act][:, 0, :]
        pos = np.asarray(ct.pos)[act]
        return geom, normal, pos

    def _face_normal(self, geom, normal, pos, finger_geom, cube_center):
        """(mean outward face normal, count) for cube<->finger contacts — the MJX twin of
        mujoco_env._cube_face_normal, over the host contact snapshot."""
        cube = self._s._cube_geom
        acc, n = np.zeros(3), 0
        for (g1, g2), nrm, p in zip(geom, normal, pos):
            if not (
                (g1 == cube and g2 == finger_geom)
                or (g1 == finger_geom and g2 == cube)
            ):
                continue
            nn = nrm.copy()
            if nn @ (p - cube_center) < 0:
                nn = -nn
            acc += nn
            n += 1
        if n == 0:
            return None, 0
        return acc / np.linalg.norm(acc), n

    def _touching(self, geom, a, b):
        return any(
            (g1 == a and g2 == b) or (g1 == b and g2 == a) for g1, g2 in geom
        )

    def _weld(self):
        """Capture the current cube-in-gripper transform into the model's weld eq and
        activate it — the MJX twin of mujoco_env._weld_cube, host-side on a single env."""
        s = self._s
        xpos = np.asarray(self._dx.xpos)
        xquat = np.asarray(self._dx.xquat)
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, xquat[s._grip_bid])
        relpos = np.zeros(3)
        mujoco.mju_rotVecQuat(relpos, xpos[s._cube_bid] - xpos[s._grip_bid], neg)
        relquat = np.zeros(4)
        mujoco.mju_mulQuat(relquat, neg, xquat[s._cube_bid])
        eq_data = np.array(self._mx.eq_data)
        eq_data[0][:3] = 0
        eq_data[0][3:6] = relpos
        eq_data[0][6:10] = relquat
        eq_data[0][10] = 1
        self._mx = self._mx.replace(eq_data=jax.numpy.asarray(eq_data))
        ea = np.array(self._dx.eq_active)
        ea[0] = 1
        self._dx = self._dx.replace(eq_active=jax.numpy.asarray(ea))

    def _set_eq_active(self, val):
        ea = np.array(self._dx.eq_active)
        ea[0] = val
        self._dx = self._dx.replace(eq_active=jax.numpy.asarray(ea))

    def step(self, action_deg):
        """One control frame under MJX physics + the weld latch — line-for-line the same
        logic as mujoco_env.Scene.step, with mjx.step for the substeps."""
        s = self._s
        gripping = float(action_deg[s.grip_i]) <= E.GRIP_WELD_MAX
        target = [s._to_target(i, float(action_deg[i])) for i in range(len(s.joints))]
        for _ in range(s._substeps):
            ctrl = np.array(self._dx.ctrl)
            for i in range(len(s.joints)):
                ctrl[i] = target[i]
            if self.welded:
                ctrl[s.grip_i] = max(target[s.grip_i], self._weld_jaw_pos)
            self._dx = self._dx.replace(ctrl=jax.numpy.asarray(ctrl))
            self._dx = self._step(self._mx, self._dx)
            geom, normal, pos = self._contacts()
            if not self.welded and gripping:
                cube_c = np.asarray(self._dx.geom_xpos)[s._cube_geom]
                up_n, up_c = self._face_normal(geom, normal, pos, s._upper, cube_c)
                lo_n, lo_c = self._face_normal(geom, normal, pos, s._lower, cube_c)
                if up_c >= 2 and lo_c >= 1 and up_n @ lo_n <= E.WELD_OPPOSING_COS:
                    self._weld()
                    self._weld_jaw_pos = float(np.asarray(self._dx.qpos)[s._grip_qadr])
                    self.welded = True
            elif not gripping:
                self._set_eq_active(0)
                self.welded = False
            if (
                not self.welded
                and not self.landed
                and self._touching(geom, s._cube_geom, s._floor)
            ):
                self.landed = True


def run_headless(scene, traj):
    """MJX twin of mujoco_replay.run_headless: drive the recorded trajectory once and
    return (landed, welded_at_end). Same success definition as the C-MuJoCo path."""
    mx = MjxScene(scene)
    mx.reset()
    for frame in traj:
        mx.step(frame)
    return mx.landed, mx.welded
