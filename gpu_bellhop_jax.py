#!/usr/bin/env python3
"""
GPU-native differentiable underwater ray / beam tracer in JAX.

This is a performance-oriented refactor of the core 2D BELLHOP-style workflow:
- fully batched over rays (no Python loops over rays)
- mask-based reflections at the free surface and sloping bottom
- JAX jit + lax.scan keeps the full forward pass on device
- differentiable with respect to sound-speed field, bathymetry, source parameters,
  and scalar physical coefficients

Notes
-----
This script is intentionally GPU-first and autodiff-native. It preserves the
high-frequency ray/beam structure of the original Fortran solver, but it does
not attempt to reproduce every legacy BELLHOP option or every file format.
Instead it implements a clean, vectorized 2D beam tracer suitable for SciML,
optimization, and gradient-based inversion.

Inputs supported
----------------
1) Built-in synthetic environment (default).
2) NPZ file with keys:
   r_env, z_env, c_grid, bathy, source_r, source_z, receiver_r, receiver_z

Array conventions
-----------------
- c_grid shape: [Nr_env, Nz_env]
- bathy shape:  [Nr_env]
- receiver grid is rectilinear: receiver_r [Nr_rx], receiver_z [Nz_rx]

Output
------
- transmission loss field TL [Nr_rx, Nz_rx]
- complex pressure field U [Nr_rx, Nz_rx]
- optional ray history for plotting
- gradient example showing end-to-end autodiff
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array


class Env(NamedTuple):
    # Uniform range-depth grid for sound speed and its derivatives.
    c: Array
    c_r: Array
    c_z: Array
    c_rr: Array
    c_rz: Array
    c_zz: Array
    bathy: Array
    bathy_r: Array
    bathy_rr: Array
    r0: float
    z0: float
    dr_env: float
    dz_env: float
    b0: float
    db_env: float
    rho_w: float
    bottom_cp: float
    bottom_rho: float
    cimag: float


class SimOutputs(NamedTuple):
    U: Array
    TL: Array
    ray_r_hist: Optional[Array]
    ray_z_hist: Optional[Array]
    launch_angles: Array


@dataclass(frozen=True)
class SimConfig:
    ds: float = 5.0                # arclength-like integration step [m]
    n_steps: int = 1400            # total integration steps
    n_rays: int = 32768            # batched rays
    max_bounces: int = 64          # termination cap for repeated reflections
    amp_kill: float = 1.0e-4       # terminate negligible rays
    sigma_min: float = 2.0         # minimum beam width for receiver deposition [m]
    store_n_rays: int = 256        # only a small subset for plotting
    store_stride: int = 4          # save every k-th integration step for plots


# -----------------------------------------------------------------------------
# Finite differences and interpolation (all differentiable w.r.t. grid values)
# -----------------------------------------------------------------------------

def first_diff(arr: Array, dx: float, axis: int) -> Array:
    sl_all = [slice(None)] * arr.ndim
    sl_c_l = list(sl_all)
    sl_c_r = list(sl_all)
    sl_m_l = list(sl_all)
    sl_m_r = list(sl_all)
    sl_c_l[axis] = slice(0, -2)
    sl_c_r[axis] = slice(2, None)
    sl_m_l[axis] = slice(0, 1)
    sl_m_r[axis] = slice(-1, None)

    center = (arr[tuple(sl_c_r)] - arr[tuple(sl_c_l)]) / (2.0 * dx)
    left = (jnp.take(arr, 1, axis=axis) - jnp.take(arr, 0, axis=axis)) / dx
    right = (jnp.take(arr, -1, axis=axis) - jnp.take(arr, -2, axis=axis)) / dx
    left = jnp.expand_dims(left, axis=axis)
    right = jnp.expand_dims(right, axis=axis)
    return jnp.concatenate([left, center, right], axis=axis)


def second_diff(arr: Array, dx: float, axis: int) -> Array:
    center = (
        jnp.take(arr, jnp.arange(2, arr.shape[axis]), axis=axis)
        - 2.0 * jnp.take(arr, jnp.arange(1, arr.shape[axis] - 1), axis=axis)
        + jnp.take(arr, jnp.arange(0, arr.shape[axis] - 2), axis=axis)
    ) / (dx * dx)
    left = jnp.expand_dims(jnp.take(center, 0, axis=axis), axis=axis)
    right = jnp.expand_dims(jnp.take(center, -1, axis=axis), axis=axis)
    return jnp.concatenate([left, center, right], axis=axis)


def mixed_diff(arr: Array, dx: float, dz: float) -> Array:
    return first_diff(first_diff(arr, dx, axis=0), dz, axis=1)


def interp1_uniform(values: Array, x: Array, x0: float, dx: float) -> Array:
    nx = values.shape[0]
    xf = (x - x0) / dx
    i0 = jnp.clip(jnp.floor(xf).astype(jnp.int32), 0, nx - 2)
    w = jnp.clip(xf - i0, 0.0, 1.0)
    v0 = values[i0]
    v1 = values[i0 + 1]
    return (1.0 - w) * v0 + w * v1


def interp2_uniform(field: Array, r: Array, z: Array, r0: float, z0: float, dr: float, dz: float) -> Array:
    nr, nz = field.shape
    rf = (r - r0) / dr
    zf = (z - z0) / dz
    ir0 = jnp.clip(jnp.floor(rf).astype(jnp.int32), 0, nr - 2)
    iz0 = jnp.clip(jnp.floor(zf).astype(jnp.int32), 0, nz - 2)
    wr = jnp.clip(rf - ir0, 0.0, 1.0)
    wz = jnp.clip(zf - iz0, 0.0, 1.0)

    f00 = field[ir0, iz0]
    f10 = field[ir0 + 1, iz0]
    f01 = field[ir0, iz0 + 1]
    f11 = field[ir0 + 1, iz0 + 1]

    return (
        (1.0 - wr) * (1.0 - wz) * f00
        + wr * (1.0 - wz) * f10
        + (1.0 - wr) * wz * f01
        + wr * wz * f11
    )


# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------

def build_env(
    r_env: Array,
    z_env: Array,
    c_grid: Array,
    bathy: Array,
    *,
    rho_w: float = 1.0,
    bottom_cp: float = 1700.0,
    bottom_rho: float = 1.8,
    cimag: float = 0.0,
) -> Env:
    r_env = jnp.asarray(r_env)
    z_env = jnp.asarray(z_env)
    c_grid = jnp.asarray(c_grid)
    bathy = jnp.asarray(bathy)

    dr_env = float(r_env[1] - r_env[0])
    dz_env = float(z_env[1] - z_env[0])
    db_env = dr_env

    c_r = first_diff(c_grid, dr_env, axis=0)
    c_z = first_diff(c_grid, dz_env, axis=1)
    c_rr = second_diff(c_grid, dr_env, axis=0)
    c_zz = second_diff(c_grid, dz_env, axis=1)
    c_rz = mixed_diff(c_grid, dr_env, dz_env)
    bathy_r = first_diff(bathy, db_env, axis=0)
    bathy_rr = second_diff(bathy, db_env, axis=0)

    return Env(
        c=c_grid,
        c_r=c_r,
        c_z=c_z,
        c_rr=c_rr,
        c_rz=c_rz,
        c_zz=c_zz,
        bathy=bathy,
        bathy_r=bathy_r,
        bathy_rr=bathy_rr,
        r0=float(r_env[0]),
        z0=float(z_env[0]),
        dr_env=dr_env,
        dz_env=dz_env,
        b0=float(r_env[0]),
        db_env=db_env,
        rho_w=rho_w,
        bottom_cp=bottom_cp,
        bottom_rho=bottom_rho,
        cimag=cimag,
    )


def make_synthetic_environment(
    nr_env: int = 2048,
    nz_env: int = 512,
    r_max: float = 20_000.0,
    z_max: float = 250.0,
    thermocline_strength: float = 14.0,
    bathy_slope: float = 0.0035,
) -> tuple[Array, Array, Array, Array]:
    r = jnp.linspace(0.0, r_max, nr_env)
    z = jnp.linspace(0.0, z_max, nz_env)
    R, Z = jnp.meshgrid(r, z, indexing="ij")

    # Smooth range-depth sound-speed field.
    c0 = 1500.0
    thermocline = thermocline_strength * jnp.tanh((Z - 70.0) / 18.0)
    range_pert = 7.5 * jnp.sin(2.0 * jnp.pi * R / 12_000.0) * jnp.exp(-Z / 190.0)
    deep_grad = 0.045 * Z
    c = c0 + deep_grad + thermocline + range_pert

    bathy = 180.0 + bathy_slope * r + 7.0 * jnp.sin(2.0 * jnp.pi * r / 9_000.0)
    bathy = jnp.clip(bathy, 120.0, z_max - 4.0)
    return r, z, c, bathy


# -----------------------------------------------------------------------------
# Vectorized physics kernels
# -----------------------------------------------------------------------------

def sample_env(env: Env, r: Array, z: Array):
    r_clip = jnp.clip(r, env.r0, env.r0 + env.dr_env * (env.c.shape[0] - 1.0001))
    z_clip = jnp.clip(z, env.z0, env.z0 + env.dz_env * (env.c.shape[1] - 1.0001))
    c = interp2_uniform(env.c, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    c_r = interp2_uniform(env.c_r, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    c_z = interp2_uniform(env.c_z, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    c_rr = interp2_uniform(env.c_rr, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    c_rz = interp2_uniform(env.c_rz, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    c_zz = interp2_uniform(env.c_zz, r_clip, z_clip, env.r0, env.z0, env.dr_env, env.dz_env)
    b = interp1_uniform(env.bathy, r_clip, env.b0, env.db_env)
    b_r = interp1_uniform(env.bathy_r, r_clip, env.b0, env.db_env)
    b_rr = interp1_uniform(env.bathy_rr, r_clip, env.b0, env.db_env)
    return c, c_r, c_z, c_rr, c_rz, c_zz, b, b_r, b_rr


def midpoint_step(r, z, tr, tz, p, q, tau, env: Env, ds: float):
    c0, cr0, cz0, crr0, crz0, czz0, _, _, _ = sample_env(env, r, z)
    cnn0 = crr0 * tz * tz - 2.0 * crz0 * tr * tz + czz0 * tr * tr

    dr0 = c0 * tr
    dz0 = c0 * tz
    dtr0 = -cr0 / (c0 * c0)
    dtz0 = -cz0 / (c0 * c0)
    dp0 = -cnn0 * q
    dq0 = c0 * p
    dtau0 = 1.0 / (c0 + 1j * env.cimag)

    r_mid = r + 0.5 * ds * dr0
    z_mid = z + 0.5 * ds * dz0
    tr_mid = tr + 0.5 * ds * dtr0
    tz_mid = tz + 0.5 * ds * dtz0
    p_mid = p + 0.5 * ds * dp0
    q_mid = q + 0.5 * ds * dq0
    tau_mid = tau + 0.5 * ds * dtau0

    c1, cr1, cz1, crr1, crz1, czz1, _, _, _ = sample_env(env, r_mid, z_mid)
    cnn1 = crr1 * tz_mid * tz_mid - 2.0 * crz1 * tr_mid * tz_mid + czz1 * tr_mid * tr_mid

    r_new = r + ds * (c1 * tr_mid)
    z_new = z + ds * (c1 * tz_mid)
    tr_new = tr + ds * (-cr1 / (c1 * c1))
    tz_new = tz + ds * (-cz1 / (c1 * c1))
    p_new = p + ds * (-cnn1 * q_mid)
    q_new = q + ds * (c1 * p_mid)
    tau_new = tau + ds * (1.0 / (c1 + 1j * env.cimag))

    # Normalize the tangent so that ||c*t|| = 1, which controls drift.
    u_r = c1 * tr_new
    u_z = c1 * tz_new
    norm_u = jnp.sqrt(u_r * u_r + u_z * u_z + 1.0e-18)
    u_r = u_r / norm_u
    u_z = u_z / norm_u
    tr_new = u_r / c1
    tz_new = u_z / c1
    return r_new, z_new, tr_new, tz_new, p_new, q_new, tau_new


# -----------------------------------------------------------------------------
# Specialized simulation builder (injects omega into the reflection kernel)
# -----------------------------------------------------------------------------

def make_simulator(env: Env, receiver_r: Array, receiver_z: Array, freq_hz: float, cfg: SimConfig):
    receiver_r = jnp.asarray(receiver_r)
    receiver_z = jnp.asarray(receiver_z)
    nr_rx = receiver_r.shape[0]
    nz_rx = receiver_z.shape[0]
    dr_rx = float(receiver_r[1] - receiver_r[0])
    rr0 = float(receiver_r[0])
    omega = 2.0 * jnp.pi * freq_hz

    @partial(jax.checkpoint, prevent_cse=False)
    def scan_step(carry, _):
        r, z, tr, tz, p, q, tau, gain, active, top_b, bot_b, U = carry

        # Batched midpoint step for every ray.
        r1, z1, tr1, tz1, p1, q1, tau1 = midpoint_step(r, z, tr, tz, p, q, tau, env, cfg.ds)

        # Reflection handling using pure masking.
        eps = 1.0e-15
        _, _, _, _, _, _, b_old, _, _ = sample_env(env, r, z)
        _, _, _, _, _, _, b_new, _, _ = sample_env(env, r1, z1)
        dist_top_old = z
        dist_top_new = z1
        dist_bot_old = b_old - z
        dist_bot_new = b_new - z1

        hit_top = active & (dist_top_old > 0.0) & (dist_top_new <= 0.0)
        hit_bot = active & (~hit_top) & (dist_bot_old > 0.0) & (dist_bot_new <= 0.0)

        lam_top = jnp.clip(dist_top_old / (dist_top_old - dist_top_new + eps), 0.0, 1.0)
        lam_bot = jnp.clip(dist_bot_old / (dist_bot_old - dist_bot_new + eps), 0.0, 1.0)

        r_hit_top = r + lam_top * (r1 - r)
        z_hit_top = jnp.zeros_like(r_hit_top)

        r_hit_bot = r + lam_bot * (r1 - r)
        _, _, _, _, _, _, b_hit_bot, br_hit_bot, brr_hit_bot = sample_env(
            env,
            r_hit_bot,
            jnp.clip(z, env.z0, env.z0 + env.dz_env * (env.c.shape[1] - 1.0001)),
        )
        z_hit_bot = b_hit_bot

        t_top = jnp.stack([jnp.ones_like(r1), jnp.zeros_like(r1)], axis=-1)
        n_top = jnp.stack([jnp.zeros_like(r1), -jnp.ones_like(r1)], axis=-1)
        k_top = jnp.zeros_like(r1)

        denom = jnp.sqrt(1.0 + br_hit_bot * br_hit_bot)
        t_bot = jnp.stack([1.0 / denom, br_hit_bot / denom], axis=-1)
        n_bot = jnp.stack([-br_hit_bot / denom, 1.0 / denom], axis=-1)
        k_bot = brr_hit_bot / jnp.power(1.0 + br_hit_bot * br_hit_bot, 1.5)

        r_hit = jnp.where(hit_top, r_hit_top, r_hit_bot)
        z_hit = jnp.where(hit_top, z_hit_top, z_hit_bot)
        t_b = jnp.where(hit_top[:, None], t_top, t_bot)
        n_b = jnp.where(hit_top[:, None], n_top, n_bot)
        kappa = jnp.where(hit_top, k_top, k_bot)

        c_hit, cr_hit, cz_hit, _, _, _, _, _, _ = sample_env(env, r_hit, z_hit)
        gradc_hit = jnp.stack([cr_hit, cz_hit], axis=-1)

        t_pre = jnp.stack([tr1, tz1], axis=-1)
        Tg = jnp.sum(t_pre * t_b, axis=-1)
        Th = jnp.sum(t_pre * n_b, axis=-1)
        t_ref = t_pre - 2.0 * Th[:, None] * n_b

        rayt = c_hit[:, None] * t_pre
        rayn = jnp.stack([-rayt[:, 1], rayt[:, 0]], axis=-1)
        rayt_tilde = c_hit[:, None] * t_ref
        rayn_tilde = -jnp.stack([-rayt_tilde[:, 1], rayt_tilde[:, 0]], axis=-1)

        cnjump = -jnp.sum(gradc_hit * (rayn_tilde - rayn), axis=-1)
        csjump = -jnp.sum(gradc_hit * (rayt_tilde - rayt), axis=-1)
        cnjump = jnp.where(hit_top, -cnjump, cnjump)

        RN = 2.0 * kappa / (c_hit * c_hit * (Th + eps))
        RN = jnp.where(hit_top, -RN, RN)
        RM = Tg / (Th + eps)
        RN = RN + RM * (2.0 * cnjump - RM * csjump) / (c_hit * c_hit)

        p_ref = p1 + q1 * RN
        q_ref = q1

        # Reflection coefficients.
        refl_top = -jnp.ones_like(gain, dtype=jnp.complex128)
        kx = omega * Tg
        kz = omega * Th
        kzP = jnp.sqrt((kx * kx - (omega / env.bottom_cp) ** 2).astype(jnp.complex128))
        refl_bot = -(
            env.rho_w * kzP - 1j * kz * env.bottom_rho
        ) / (
            env.rho_w * kzP + 1j * kz * env.bottom_rho + 1.0e-18j
        )
        refl = jnp.where(hit_top, refl_top, refl_bot)

        # Mirror the endpoint through the local tangent plane so the ray remains inside.
        plane_dist = -jnp.sum((jnp.stack([r1 - r_hit, z1 - z_hit], axis=-1)) * n_b, axis=-1)
        x1_ref = jnp.stack([r1, z1], axis=-1) + 2.0 * plane_dist[:, None] * n_b

        r1 = jnp.where(hit_top | hit_bot, x1_ref[:, 0], r1)
        z1 = jnp.where(hit_top | hit_bot, x1_ref[:, 1], z1)
        tr1 = jnp.where(hit_top | hit_bot, t_ref[:, 0], tr1)
        tz1 = jnp.where(hit_top | hit_bot, t_ref[:, 1], tz1)
        p1 = jnp.where(hit_top | hit_bot, p_ref, p1)
        q1 = jnp.where(hit_top | hit_bot, q_ref, q1)
        gain = jnp.where(hit_top | hit_bot, gain * refl, gain)
        top_b = top_b + hit_top.astype(jnp.int32)
        bot_b = bot_b + hit_bot.astype(jnp.int32)

        # Termination mask.
        _, _, _, _, _, _, b_post, _, _ = sample_env(env, r1, z1)
        inside = (z1 >= -1.0e-8) & (z1 <= b_post + 1.0e-8)
        active = active & inside & (r1 <= receiver_r[-1] + cfg.ds) & (jnp.abs(gain) > cfg.amp_kill)
        active = active & ((top_b + bot_b) <= cfg.max_bounces)

        # Range-plane deposition to build TL on a receiver grid.
        ir0 = jnp.floor((r - rr0) / dr_rx).astype(jnp.int32)
        ir1 = jnp.floor((r1 - rr0) / dr_rx).astype(jnp.int32)
        crossed = active & (ir1 > ir0) & (ir1 >= 0) & (ir1 < nr_rx)

        rr_hit = rr0 + ir1.astype(jnp.float64) * dr_rx
        lam_r = jnp.clip((rr_hit - r) / (r1 - r + 1.0e-18), 0.0, 1.0)
        z_cross = z + lam_r * (z1 - z)
        tau_cross = tau + lam_r * (tau1 - tau)
        q_cross = q + lam_r * (q1 - q)
        c_cross, _, _, _, _, _, _, _, _ = sample_env(env, rr_hit, z_cross)

        delta_alpha = jnp.pi / max(cfg.n_rays - 1, 1)
        q0_ref = c_cross / max(delta_alpha, 1.0e-8)
        sigma_z = jnp.maximum(jnp.abs(q_cross) / (q0_ref + 1.0e-12), cfg.sigma_min)

        depth_offsets = receiver_z[None, :] - z_cross[:, None]
        depth_kernel = jnp.exp(-0.5 * (depth_offsets / sigma_z[:, None]) ** 2)
        beam_prefac = gain[:, None] * jnp.sqrt(c_cross[:, None] / (jnp.abs(q_cross)[:, None] + 1.0e-12))
        contrib = beam_prefac * depth_kernel * jnp.exp(-1j * omega * tau_cross)[:, None]
        contrib = jnp.where(crossed[:, None], contrib, 0.0 + 0.0j)
        U = U.at[ir1, :].add(contrib)

        carry = (r1, z1, tr1, tz1, p1, q1, tau1, gain, active, top_b, bot_b, U)
        hist = (r1[: cfg.store_n_rays], z1[: cfg.store_n_rays])
        return carry, hist

    @jax.jit
    def simulate(source_r: Array, source_z: Array, launch_angles: Array) -> SimOutputs:
        source_r = jnp.asarray(source_r)
        source_z = jnp.asarray(source_z)
        launch_angles = jnp.asarray(launch_angles)

        n_sources = source_r.shape[0]
        n_angles = launch_angles.shape[0]
        src_r = jnp.repeat(source_r, n_angles)
        src_z = jnp.repeat(source_z, n_angles)
        alpha = jnp.tile(launch_angles, n_sources)

        c0, _, _, _, _, _, _, _, _ = sample_env(env, src_r, src_z)
        tr0 = jnp.cos(alpha) / c0
        tz0 = jnp.sin(alpha) / c0
        p0 = jnp.ones_like(src_r, dtype=jnp.complex128)
        q0 = 1j * jnp.ones_like(src_r, dtype=jnp.complex128)
        tau0 = jnp.zeros_like(src_r, dtype=jnp.complex128)
        gain0 = jnp.ones_like(src_r, dtype=jnp.complex128)
        active0 = jnp.ones_like(src_r, dtype=bool)
        top_b0 = jnp.zeros_like(src_r, dtype=jnp.int32)
        bot_b0 = jnp.zeros_like(src_r, dtype=jnp.int32)
        U0 = jnp.zeros((nr_rx, nz_rx), dtype=jnp.complex128)

        carry0 = (src_r, src_z, tr0, tz0, p0, q0, tau0, gain0, active0, top_b0, bot_b0, U0)
        carry, hist = jax.lax.scan(scan_step, carry0, xs=None, length=cfg.n_steps)

        r, z, tr, tz, p, q, tau, gain, active, top_b, bot_b, U = carry
        del r, z, tr, tz, p, q, tau, gain, active, top_b, bot_b

        intensity = jnp.abs(U)
        TL = -20.0 * jnp.log10(intensity + 1.0e-12)
        ray_r_hist = hist[0][:: cfg.store_stride]
        ray_z_hist = hist[1][:: cfg.store_stride]
        return SimOutputs(U=U, TL=TL, ray_r_hist=ray_r_hist, ray_z_hist=ray_z_hist, launch_angles=alpha[:n_angles])

    return simulate


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def load_npz_environment(path: Path):
    data = np.load(path)
    required = ["r_env", "z_env", "c_grid", "bathy", "source_r", "source_z", "receiver_r", "receiver_z"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return {k: jnp.asarray(data[k]) for k in required}


def save_results_png(outdir: Path, receiver_r: Array, receiver_z: Array, TL: Array, ray_r_hist: Optional[Array], ray_z_hist: Optional[Array]):
    outdir.mkdir(parents=True, exist_ok=True)
    rr = np.asarray(receiver_r) / 1000.0
    zz = np.asarray(receiver_z)
    TL_np = np.asarray(TL).T

    plt.figure(figsize=(10, 5))
    extent = [rr[0], rr[-1], zz[-1], zz[0]]
    plt.imshow(TL_np, aspect="auto", extent=extent)
    plt.colorbar(label="Transmission Loss [dB]")
    plt.xlabel("Range [km]")
    plt.ylabel("Depth [m]")
    plt.title("GPU-differentiable JAX ray/beam TL field")
    plt.tight_layout()
    plt.savefig(outdir / "tl_field.png", dpi=180)
    plt.close()

    if ray_r_hist is not None and ray_z_hist is not None:
        plt.figure(figsize=(10, 5))
        r_hist = np.asarray(ray_r_hist) / 1000.0
        z_hist = np.asarray(ray_z_hist)
        nplot = min(r_hist.shape[1], 64)
        for i in range(nplot):
            plt.plot(r_hist[:, i], z_hist[:, i], linewidth=0.8)
        plt.gca().invert_yaxis()
        plt.xlabel("Range [km]")
        plt.ylabel("Depth [m]")
        plt.title("Sampled ray trajectories")
        plt.tight_layout()
        plt.savefig(outdir / "ray_paths.png", dpi=180)
        plt.close()


# -----------------------------------------------------------------------------
# Autodiff demo
# -----------------------------------------------------------------------------

def gradient_demo(receiver_r: Array, receiver_z: Array, freq_hz: float, cfg: SimConfig):
    src_r = jnp.array([0.0])
    src_z = jnp.array([60.0])
    launch_angles = jnp.linspace(jnp.deg2rad(-35.0), jnp.deg2rad(35.0), 1024)

    def loss_from_strength(thermocline_strength: float):
        r_env, z_env, c_grid, bathy = make_synthetic_environment(
            nr_env=1024,
            nz_env=256,
            r_max=float(receiver_r[-1]),
            z_max=float(receiver_z[-1]) + 20.0,
            thermocline_strength=thermocline_strength,
            bathy_slope=0.0035,
        )
        env = build_env(r_env, z_env, c_grid, bathy)
        sim = make_simulator(env, receiver_r, receiver_z, freq_hz, cfg)
        outputs = sim(src_r, src_z, launch_angles)
        # Smooth scalar objective: maximize intensity near the center of the receiver grid.
        mid_r = outputs.U.shape[0] // 2
        mid_z = outputs.U.shape[1] // 2
        return -jnp.log(jnp.abs(outputs.U[mid_r, mid_z]) + 1.0e-12)

    grad_val = jax.grad(loss_from_strength)(14.0)
    return float(grad_val)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU-native differentiable underwater ray tracer in JAX")
    parser.add_argument("--env-npz", type=Path, default=None, help="Optional NPZ file with environment arrays")
    parser.add_argument("--outdir", type=Path, default=Path("jax_gpu_bellhop_out"))
    parser.add_argument("--freq", type=float, default=250.0, help="Acoustic frequency [Hz]")
    parser.add_argument("--n-rays", type=int, default=32768)
    parser.add_argument("--n-steps", type=int, default=1400)
    parser.add_argument("--ds", type=float, default=5.0)
    args = parser.parse_args()

    cfg = SimConfig(n_rays=args.n_rays, n_steps=args.n_steps, ds=args.ds)

    if args.env_npz is None:
        r_env, z_env, c_grid, bathy = make_synthetic_environment()
        source_r = jnp.array([0.0])
        source_z = jnp.array([60.0])
        receiver_r = jnp.linspace(100.0, 20_000.0, 300)
        receiver_z = jnp.linspace(1.0, 220.0, 220)
    else:
        data = load_npz_environment(args.env_npz)
        r_env = data["r_env"]
        z_env = data["z_env"]
        c_grid = data["c_grid"]
        bathy = data["bathy"]
        source_r = data["source_r"]
        source_z = data["source_z"]
        receiver_r = data["receiver_r"]
        receiver_z = data["receiver_z"]

    env = build_env(r_env, z_env, c_grid, bathy)
    launch_angles = jnp.linspace(jnp.deg2rad(-70.0), jnp.deg2rad(70.0), cfg.n_rays)
    simulate = make_simulator(env, receiver_r, receiver_z, args.freq, cfg)

    outputs = simulate(source_r, source_z, launch_angles)
    save_results_png(args.outdir, receiver_r, receiver_z, outputs.TL, outputs.ray_r_hist, outputs.ray_z_hist)

    np.savez(
        args.outdir / "results.npz",
        receiver_r=np.asarray(receiver_r),
        receiver_z=np.asarray(receiver_z),
        U=np.asarray(outputs.U),
        TL=np.asarray(outputs.TL),
        ray_r_hist=np.asarray(outputs.ray_r_hist),
        ray_z_hist=np.asarray(outputs.ray_z_hist),
    )

    grad_example = gradient_demo(receiver_r, receiver_z, args.freq, SimConfig(n_rays=1024, n_steps=700, ds=args.ds, store_n_rays=64))
    print("Saved results to:", args.outdir)
    print("Gradient demo d(loss)/d(thermocline_strength) =", grad_example)


if __name__ == "__main__":
    main()
