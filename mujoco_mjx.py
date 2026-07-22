#!/usr/bin/env python
"""MJX (MuJoCo-XLA) physics backend for the SO-101 pick-cube twin.

==============================================================================
⚠️  PARKED — CLEARLY BROKEN ON PERFORMANCE. GET BACK TO THIS.  (2026-07-22)
==============================================================================
Correctness is GOOD: single-env fidelity ep0 matches C-MuJoCo (grasp→weld→
release→land), and MJX's convex-mesh contacts drive the weld gate (over-counts
points 8 vs 4, but the gate is a lower-bound threshold so it fires either way).

PERFORMANCE IS A DISASTER: batched fidelity on a 4090 (9jian, mujoco 3.10 +
mujoco-mjx 3.10 + jax[cuda12]) ran **128 episodes in >73 min, GPU pegged**, vs
C-MuJoCo run_headless doing the same 128 in **84 s** on the laptop — ~50× SLOWER.
So the batching win does not exist yet; the naive port loses badly.

Prime suspects (fix before trusting any number):
  1. `base.replace(eq_data=eq_data)` PER SUBSTEP (see substep()) makes the whole
     Model pytree loop-varying through the scan — breaks MJX's core assumption
     that the model is a compile-time constant and only Data flows. Almost
     certainly the main killer. Fix: get the weld transform OUT of the model —
     set it in Data, or use a kinematic/mocap weld — so the model stays constant.
  2. Dense mesh collision over the arm's 16 convex meshes every step (mjx does
     GJK across all enabled pairs; C-MuJoCo has broadphase + specialised paths).
     Fix: disable collision on the non-finger arm links (they follow recorded
     joints, never touch the cube) to slash per-step cost.
  3. XLA autotuner OOMs at ~19 GB → forced --xla_gpu_autotune_level=0 → slow
     default kernels. Once (1)+(2) cut memory, turn autotune back on.

Also unresolved: this mjx build has NO cylinder-box collision (NotImplementedError),
so the cylinder distractor must become a mesh (or be excluded) for the full 540-ep
set — the batched harness currently runs only the 440 zero-distractor episodes.

The plan needs changes to the mujoco lib itself (weld representation, arm-link
collision flags), hence parked here until those land. Harness lives in
scratchpad (mjx_valbench.py: per-env qpos, vmap, chunked, parity vs C baseline).
==============================================================================

A full in-graph port of mujoco_env.Scene's physics: same compiled model, same
contact-gated weld + latch, driven through mjx.step — with the weld gate, the
grasp-transform capture, the latch, and landing detection ALL expressed in JAX
inside a single jitted lax.scan. Nothing leaves the device mid-episode: one host
sync per episode (the final landed/welded read). That's the whole point — it's
fast on CPU and the exact code that batches (vmap) on a GPU node.

The weld equality's captured transform lives in model.eq_data, so it's threaded
through the scan carry and re-bound each substep via model.replace — functional,
traced, no host round-trip. Success semantics match mujoco_replay.run_headless.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from mujoco import mjx

import mujoco_env as E


def _quat_conj(q):
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_rot(v, q):
    """Rotate vector v by quaternion q ([w,x,y,z]) — v + 2w(u×v) + 2u×(u×v)."""
    w, u = q[0], q[1:]
    t = 2.0 * jnp.cross(u, v)
    return v + w * t + jnp.cross(u, t)


def _face_normal(geom, normal, pos, active, cube, finger, cube_c):
    """(mean outward face normal, contact count) for cube<->finger contacts — the JAX
    twin of mujoco_env._cube_face_normal, over the padded contact arrays. Each contact
    normal is flipped to point from the cube center outward before averaging."""
    is_pair = active & (
        ((geom[:, 0] == cube) & (geom[:, 1] == finger))
        | ((geom[:, 0] == finger) & (geom[:, 1] == cube))
    )
    count = jnp.sum(is_pair)
    sign = jnp.sum(normal * (pos - cube_c), axis=1)
    flipped = jnp.where((sign < 0)[:, None], -normal, normal)
    acc = jnp.sum(jnp.where(is_pair[:, None], flipped, 0.0), axis=0)
    n = jnp.linalg.norm(acc)
    mean = acc / jnp.maximum(n, 1e-9)
    return mean, count


class MjxRollout:
    """Jitted single-env MJX rollout of a compiled Scene. Reuses the C-MuJoCo `scene`
    only for construction (model, geom/body ids, control mapping, initial state)."""

    def __init__(self, scene):
        s = self._s = scene
        self._base = mjx.put_model(scene.model)
        nj = len(s.joints)
        cube, upper, lower, floor = s._cube_geom, s._upper, s._lower, s._floor
        grip_i, grip_qadr = s.grip_i, s._grip_qadr
        grip_bid, cube_bid = s._grip_bid, s._cube_bid
        base = self._base
        cos = E.WELD_OPPOSING_COS

        def substep(carry, _, target, gripping, valid):
            dx, eq_data, welded, weld_jaw, landed = carry
            prev = (eq_data, welded, weld_jaw, landed)
            grip_ctrl = jnp.where(
                welded, jnp.maximum(target[grip_i], weld_jaw), target[grip_i]
            )
            ctrl = dx.ctrl.at[:nj].set(target).at[grip_i].set(grip_ctrl)
            mx = base.replace(eq_data=eq_data)
            dx = mjx.step(mx, dx.replace(ctrl=ctrl))

            ct = dx._impl.contact
            active = ct.dist < 0.0
            geom, normal, pos = ct.geom, ct.frame[:, 0, :], ct.pos
            cube_c = dx.geom_xpos[cube]
            up_n, up_c = _face_normal(geom, normal, pos, active, cube, upper, cube_c)
            lo_n, lo_c = _face_normal(geom, normal, pos, active, cube, lower, cube_c)
            fire = (
                (~welded)
                & gripping
                & (up_c >= 2)
                & (lo_c >= 1)
                & (jnp.dot(up_n, lo_n) <= cos)
            )

            neg = _quat_conj(dx.xquat[grip_bid])
            relpos = _quat_rot(dx.xpos[cube_bid] - dx.xpos[grip_bid], neg)
            relquat = _quat_mul(neg, dx.xquat[cube_bid])
            weld_row = jnp.concatenate(
                [jnp.zeros(3), relpos, relquat, jnp.ones(1)]
            )  # eq_data row: [anchor(3), relpos(3), relquat(4), torquescale(1)]
            eq_data = jnp.where(fire, eq_data.at[0].set(weld_row), eq_data)

            welded = jnp.where(gripping, welded | fire, False)
            weld_jaw = jnp.where(fire, dx.qpos[grip_qadr], weld_jaw)
            dx = dx.replace(
                eq_active=dx.eq_active.at[0].set(welded.astype(dx.eq_active.dtype))
            )

            floor_pair = active & (
                ((geom[:, 0] == cube) & (geom[:, 1] == floor))
                | ((geom[:, 0] == floor) & (geom[:, 1] == cube))
            )
            landed = landed | ((~welded) & jnp.any(floor_pair))
            # past an episode's true end (valid False) freeze the recorded weld/land state;
            # physics still steps (uniform scan) but the padded frames can't flip results
            eq_data, welded, weld_jaw, landed = jax.tree_util.tree_map(
                lambda new, old: jnp.where(valid, new, old),
                (eq_data, welded, weld_jaw, landed),
                prev,
            )
            return (dx, eq_data, welded, weld_jaw, landed), None

        def frame(carry, inp):
            target, gripping, valid = inp
            body = lambda c, x: substep(c, x, target, gripping, valid)
            carry, _ = lax.scan(body, carry, None, length=s._substeps)
            return carry, None

        def rollout(dx0, targets, grippings, valids):
            carry = (dx0, base.eq_data, jnp.bool_(False), 0.0, jnp.bool_(False))
            carry, _ = lax.scan(frame, carry, (targets, grippings, valids))
            _, _, welded, _, landed = carry
            return landed, welded

        # expose the pure fn so callers can vmap it across a batch of envs (the GPU win),
        # plus a jitted single-env entry for parity checks
        self.rollout_fn = rollout
        self._rollout = jax.jit(rollout)

    def _targets(self, traj):
        """Host-side, once per episode: recorded actions → ctrl targets + per-frame
        grip gate. Not in the hot loop; the scan consumes these arrays."""
        s = self._s
        nj = len(s.joints)
        targets = np.array(
            [[s._to_target(i, float(f[i])) for i in range(nj)] for f in traj]
        )
        grippings = traj[:, s.grip_i] <= E.GRIP_WELD_MAX
        return jnp.asarray(targets), jnp.asarray(grippings)

    def run(self, traj):
        """Drive the recorded trajectory once; return (landed, welded_at_end) as python
        bools. One device sync, at the end."""
        self._s.reset()
        dx = mjx.put_data(self._s.model, self._s.data)
        dx = dx.replace(eq_active=dx.eq_active.at[0].set(False))
        targets, grippings = self._targets(traj)
        valids = jnp.ones(len(traj), dtype=bool)
        landed, welded = self._rollout(dx, targets, grippings, valids)
        return bool(landed), bool(welded)


def run_headless(scene, traj):
    """MJX twin of mujoco_replay.run_headless: (landed, welded_at_end), same success
    definition as the C-MuJoCo path."""
    return MjxRollout(scene).run(traj)
