"""Differentiable 2D underwater acoustic ray and beam tracing in JAX.

This module contains the main forward solvers used by the repository:

- ``solve_transmission_loss(...)``:
  Bellhop-oriented tracing and field accumulation path used for fidelity
  studies and Bellhop/JAX validation.
- ``solve_transmission_loss_autodiff(...)``:
  autodiff-safe path for SciML workloads where smooth gradients are more
  important than exact Bellhop feature parity.

Design overview
---------------
The implementation is organized as a three-stage pipeline:

1. Acoustic environment configuration
   ``configure_acoustic_operators(...)`` installs sound-speed, bathymetry,
   altimetry, and reflection operators used by the solver.
2. Batched ray tracing
   Rays are traced as a single batched state tensor using JAX-native control
   flow and masking. This keeps the rollout GPU/TPU-friendly and avoids
   Python loops over rays.
3. Receiver-grid field accumulation
   Traced beam states are mapped to a receiver grid to produce complex
   pressure and transmission loss fields.

Two solver modes are intentionally exposed because the repository serves two
different engineering goals:

- Bellhop compatibility for benchmarking and validation.
- end-to-end differentiability for inverse problems and SciML training.
"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import core as jax_core

import sys
import os
import functools
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Literal

sys.path.append('.')
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import functions from other modules.
from sound_speed import c, c_r, c_z, c_rr, c_zz, c_rz, DEFAULT_OPERATORS
from boundary import (
    bathymetry as bty, altimetry as ati, normal_vector_to_altimetry, normal_vector_to_bathymetry as n_bty,
    normal_vector_to_altimetry as n_ati, tangent_vector_to_altimetry as t_ati, tangent_vector_to_bathymetry as t_bty,
    DEFAULT_BOUNDARY_OPERATORS,
)

RunMode = Literal["coherent", "incoherent", "semicoherent"]
AccumulationModel = Literal["bellhop", "gaussian", "hat"]
BeamType = Literal["geometric"]


@dataclass(frozen=True)
class ReflectionModelConfig:
    source_type: str = "point"
    water_density_gcc: float = 1.0
    top_boundary_condition: str = "vacuum"
    bottom_boundary_condition: str = "rigid"
    kill_backward_rays: bool = False
    bottom_alpha_r_mps: float = 1600.0
    bottom_alpha_i_user: float = 0.0
    bottom_beta_r_mps: float = 0.0
    bottom_beta_i_user: float = 0.0
    bottom_density_gcc: float = 1.8
    attenuation_units: str = "W"


@dataclass(frozen=True)
class StepControlConfig:
    ssp_interface_depths_m: jnp.ndarray = field(default_factory=lambda: jnp.asarray([], dtype=jnp.float64))
    bathymetry_range_breaks_m: jnp.ndarray = field(default_factory=lambda: jnp.asarray([], dtype=jnp.float64))
    altimetry_range_breaks_m: jnp.ndarray = field(default_factory=lambda: jnp.asarray([], dtype=jnp.float64))


DEFAULT_REFLECTION_MODEL = ReflectionModelConfig()
DEFAULT_STEP_CONTROL = StepControlConfig()


def _reflection_model_to_dict(reflection_model: ReflectionModelConfig | Mapping | None) -> dict:
    if reflection_model is None:
        return asdict(DEFAULT_REFLECTION_MODEL)
    if isinstance(reflection_model, ReflectionModelConfig):
        return asdict(reflection_model)
    merged = asdict(DEFAULT_REFLECTION_MODEL)
    merged.update(dict(reflection_model))
    return merged


def _step_control_from_operators(ssp_operators: Mapping, boundary_operators: Mapping) -> StepControlConfig:
    return StepControlConfig(
        ssp_interface_depths_m=jnp.asarray(ssp_operators.get("interface_depths_m", jnp.asarray([], dtype=jnp.float64))),
        bathymetry_range_breaks_m=jnp.asarray(boundary_operators.get("bathymetry_range_breaks_m", jnp.asarray([], dtype=jnp.float64))),
        altimetry_range_breaks_m=jnp.asarray(boundary_operators.get("altimetry_range_breaks_m", jnp.asarray([], dtype=jnp.float64))),
    )


def configure_acoustic_operators(sound_speed_operators=None, boundary_operators=None, reflection_model=None):
    """Configure globally active acoustic operators for tracing and field solve."""
    global c, c_r, c_z, c_rr, c_zz, c_rz
    global bty, ati, n_bty, n_ati, t_ati, t_bty
    global REFLECTION_MODEL, STEP_CONTROL

    ssp = DEFAULT_OPERATORS if sound_speed_operators is None else sound_speed_operators
    bdry = DEFAULT_BOUNDARY_OPERATORS if boundary_operators is None else boundary_operators

    c = ssp["c"]
    c_r = ssp["c_r"]
    c_z = ssp["c_z"]
    c_rr = ssp["c_rr"]
    c_rz = ssp["c_rz"]
    c_zz = ssp["c_zz"]

    bty = bdry["bathymetry"]
    ati = bdry["altimetry"]
    n_bty = bdry["normal_vector_to_bathymetry"]
    n_ati = bdry["normal_vector_to_altimetry"]
    t_bty = bdry["tangent_vector_to_bathymetry"]
    t_ati = bdry["tangent_vector_to_altimetry"]
    REFLECTION_MODEL = _reflection_model_to_dict(reflection_model)
    STEP_CONTROL = asdict(_step_control_from_operators(ssp, bdry))


REFLECTION_MODEL = asdict(DEFAULT_REFLECTION_MODEL)
STEP_CONTROL = asdict(DEFAULT_STEP_CONTROL)


def bellhop_recommended_nbeams(freq, c0, r_max, theta_min, theta_max):
    d_alpha_opt = jnp.sqrt(c0 / (6.0 * freq * jnp.maximum(r_max, 1.0)))
    span = jnp.maximum(theta_max - theta_min, 0.0)
    n_opt = jnp.round(2.0 + span / (d_alpha_opt + 1e-12))
    return jnp.maximum(1, n_opt.astype(jnp.int32))


def evaluate_source_beam_pattern(theta_rad, pattern_angles_deg=None, pattern_db=None):
    if pattern_angles_deg is None or pattern_db is None:
        return jnp.ones_like(theta_rad, dtype=jnp.float64)

    angles_deg = jnp.asarray(pattern_angles_deg, dtype=jnp.float64)
    gains_db = jnp.asarray(pattern_db, dtype=jnp.float64)
    if angles_deg.ndim != 1 or gains_db.ndim != 1:
        raise ValueError("Source beam pattern angles and dB arrays must be one-dimensional.")
    if angles_deg.shape[0] != gains_db.shape[0]:
        raise ValueError("Source beam pattern angles and dB arrays must have the same length.")
    if angles_deg.shape[0] < 2:
        raise ValueError("Source beam pattern must contain at least two points.")

    launch_angles_deg = jnp.rad2deg(theta_rad)
    gains_linear = 10.0 ** (gains_db / 20.0)
    return jnp.interp(launch_angles_deg, angles_deg, gains_linear)


def _complex_sound_speed(real_speed, attenuation_value, freq, atten_unit):
    omega = 2.0 * jnp.pi * freq
    alpha_nepers_per_m = jnp.where(
        atten_unit == 'W',
        attenuation_value * freq / (8.6858896 * real_speed + 1e-12),
        attenuation_value / 8.6858896,
    )
    imag_speed = alpha_nepers_per_m * real_speed * real_speed / (omega + 1e-12)
    return real_speed + 1j * imag_speed


def _acoustic_halfspace_reflection_coefficient(freq, c_water, tangential_component, normal_component):
    omega = 2.0 * jnp.pi * freq
    cp_bottom = _complex_sound_speed(
        REFLECTION_MODEL["bottom_alpha_r_mps"],
        REFLECTION_MODEL["bottom_alpha_i_user"],
        freq,
        REFLECTION_MODEL["attenuation_units"],
    )
    rho_water = REFLECTION_MODEL["water_density_gcc"]
    rho_bottom = REFLECTION_MODEL["bottom_density_gcc"]

    kx = omega * tangential_component
    kz = omega * normal_component
    kz_p = jnp.sqrt(kx * kx - (omega / cp_bottom) ** 2)
    kz_p = jnp.where(
        jnp.logical_and(jnp.isclose(jnp.real(kz_p), 0.0), jnp.imag(kz_p) < 0.0),
        -kz_p,
        kz_p,
    )
    return -(rho_water * kz_p - 1j * kz * rho_bottom) / (rho_water * kz_p + 1j * kz * rho_bottom + 1e-12)


def _apply_boundary_reflection(Y_new, coeff, bounce_index):
    amp = jnp.abs(coeff)
    phase = jnp.arctan2(jnp.imag(coeff), jnp.real(coeff))
    if Y_new.shape[0] > bounce_index:
        Y_new = Y_new.at[bounce_index].set(Y_new[bounce_index] + 1.0)
    if Y_new.shape[0] > 11:
        Y_new = Y_new.at[11].set(Y_new[11] + jnp.log(jnp.maximum(amp, 1e-12)))
    if Y_new.shape[0] > 12:
        Y_new = Y_new.at[12].set(Y_new[12] + phase)
    return Y_new



@jax.jit
def dynamic_ray_eqns(Y):
    """
    Defines the ODE system for 'dynamic' ray tracing, which includes
    (q, p) for beam spreading / amplitude calculation.
    
    State: Y = [r, z, rho, zeta, q_r, q_i, p_r, p_i, tau]
    """
    r, z, rho, zeta, q_r, q_i, p_r, p_i, tau = Y[:9]

    # Sound speed and derivatives
    c_val = c(r, z)
    cr = c_r(r, z)
    cz = c_z(r, z)
    crr = c_rr(r, z)
    crz = c_rz(r, z)
    czz = c_zz(r, z)
    
    c2 = c_val**2
    # c_{nn} = c^2 * (c_rr * zeta^2 - 2*c_rz*rho*zeta + c_zz*rho^2)
    c_nn = c2 * (crr * zeta**2 - 2.0 * crz * rho * zeta + czz * rho**2)
    
    # ODEs
    # dr/ds   = c_val * rho, 0
    # dz/ds   = c_val * zeta, 1
    # drho/ds = - (cr / c^2), 2
    # dzeta/ds= - (cz / c^2), 3
    # dq_r/ds   = c_val * p_r, 4
    # dq_i/ds   = c_val * p_i, 5
    # dp_r/ds   = -(c_nn / c^2) * q_r, 6
    # dp_i/ds   = -(c_nn / c^2) * q_i, 7
    # dtau/ds = 1/c_val, 8

    Y_dot = jnp.zeros_like(Y)
    Y_dot = Y_dot.at[0].set(c_val * rho)
    Y_dot = Y_dot.at[1].set(c_val * zeta)
    Y_dot = Y_dot.at[2].set(-cr / c2)
    Y_dot = Y_dot.at[3].set(-cz / c2)
    Y_dot = Y_dot.at[4].set(c_val * p_r)
    Y_dot = Y_dot.at[5].set(c_val * p_i)
    Y_dot = Y_dot.at[6].set(- (c_nn / c2) * q_r)
    Y_dot = Y_dot.at[7].set(- (c_nn / c2) * q_i)
    Y_dot = Y_dot.at[8].set(1 / c_val)

    return Y_dot

# Modified RK2 stepper with reflection counting.
class RK2_stepper:
    def __init__(self, function, dt, freq=None, ati=None, bty=None, t_ati=None, n_ati=None, t_bty=None, n_bty=None):
        self.function = function  # Expecting extended_ray_eqns.
        self.dt = dt
        self.freq = 0.0 if freq is None else freq
        self.ati = ati if ati is not None else globals()["ati"]
        self.bty = bty if bty is not None else globals()["bty"]
        self.n_bty = n_bty if n_bty is not None else globals()["n_bty"]
        self.t_bty = t_bty if t_bty is not None else globals()["t_bty"]
        self.t_ati = t_ati if t_ati is not None else globals()["t_ati"]
        self.n_ati = n_ati if n_ati is not None else globals()["n_ati"]

    def _crossing_step_to_depths(self, z0, dzds, h, depth_breaks):
        if depth_breaks.size == 0:
            return jnp.inf
        z1 = z0 + h * dzds
        between = jnp.logical_or(
            jnp.logical_and(depth_breaks > z0, depth_breaks <= z1),
            jnp.logical_and(depth_breaks < z0, depth_breaks >= z1),
        )
        cand = (depth_breaks - z0) / (dzds + 1e-12)
        cand = jnp.where(jnp.logical_and(between, cand > 0.0), cand, jnp.inf)
        return jnp.min(cand)

    def _crossing_step_to_range_breaks(self, r0, drds, h, range_breaks):
        if range_breaks.size == 0:
            return jnp.inf
        r1 = r0 + h * drds
        between = jnp.logical_or(
            jnp.logical_and(range_breaks > r0, range_breaks <= r1),
            jnp.logical_and(range_breaks < r0, range_breaks >= r1),
        )
        cand = (range_breaks - r0) / (drds + 1e-12)
        cand = jnp.where(jnp.logical_and(between, cand > 0.0), cand, jnp.inf)
        return jnp.min(cand)

    def _reduce_step(self, Y, c_local, h):
        c_tray = c_local * jnp.array([Y[2], Y[3]])
        r0, z0 = Y[0], Y[1]
        x_trial = jnp.array([r0, z0]) + h * c_tray

        h_layer = self._crossing_step_to_depths(z0, c_tray[1], h, STEP_CONTROL["ssp_interface_depths_m"])

        f_top0 = z0 - self.ati(r0)
        f_top1 = x_trial[1] - self.ati(x_trial[0])
        dtop_dr = jax.grad(self.ati)(r0)
        denom_top = c_tray[1] - dtop_dr * c_tray[0]
        h_top = jax.lax.cond(
            jnp.logical_and(f_top0 >= 0.0, f_top1 < 0.0),
            lambda _: -f_top0 / (denom_top + 1e-12),
            lambda _: jnp.inf,
            operand=None,
        )

        f_bot0 = z0 - self.bty(r0)
        f_bot1 = x_trial[1] - self.bty(x_trial[0])
        dbot_dr = jax.grad(self.bty)(r0)
        denom_bot = c_tray[1] - dbot_dr * c_tray[0]
        h_bot = jax.lax.cond(
            jnp.logical_and(f_bot0 <= 0.0, f_bot1 > 0.0),
            lambda _: -f_bot0 / (denom_bot + 1e-12),
            lambda _: jnp.inf,
            operand=None,
        )

        h_top_seg = self._crossing_step_to_range_breaks(r0, c_tray[0], h, STEP_CONTROL["altimetry_range_breaks_m"])
        h_bot_seg = self._crossing_step_to_range_breaks(r0, c_tray[0], h, STEP_CONTROL["bathymetry_range_breaks_m"])

        h_new = jnp.min(jnp.asarray([h, h_layer, h_top, h_bot, h_top_seg, h_bot_seg]))
        h_new = jnp.where(jnp.logical_or(~jnp.isfinite(h_new), h_new <= 0.0), h, h_new)
        return jnp.where(h_new < 1.0e-4 * self.dt, 1.0e-5 * self.dt, h_new)

    def bottom_reflection(self, Y, Y_new):
        # Reflection handling for bottom (bathymetry).
        t_ray = jnp.array([Y[2], Y[3]])
        n_bdry = self.n_bty(Y[0])
        f_current = Y[1] - self.bty(Y[0])
        f_new = Y_new[1] - self.bty(Y_new[0])
        d0 = jnp.array([Y[0], Y[1]]) - jnp.array([Y[0], self.bty(Y[0])])
        dt_proj = jnp.abs(jnp.dot(-d0, n_bdry)) / (jnp.abs(jnp.dot(c(Y[0], Y[1]) * t_ray, n_bdry)) + 1e-12)
        dt_linear = self.dt * jnp.abs(f_current) / (jnp.abs(f_current) + jnp.abs(f_new) + 1e-12)
        dt_tilde = (dt_proj + dt_linear) / 2.0
        
        K = Y + (dt_tilde / 2.0) * self.function(Y)
        Y_new = Y + dt_tilde * self.function(K)
        Y_new = Y_new.at[1].set(self.bty(Y_new[0]))
        
        # alpha = jnp.arctan(jax.grad(self.bty)(Y_new[0]))
        # theta = jnp.arctan(Y_new[3] / Y_new[2])
        # theta_new = -theta + 2 * alpha
        # C_new = c(Y_new[0], Y_new[1])
        # Y_new = Y_new.at[2].set(jnp.cos(theta_new) / C_new)
        # Y_new = Y_new.at[3].set(jnp.sin(theta_new) / C_new)
        
        # # dynamic ray equations update
        # t_ray_new = jnp.array([Y_new[2], Y_new[3]])
        # n_bdry_new = self.n_bty(Y_new[0])
        # t_bdry_new = jnp.array([Y_new[0], self.bty(Y_new[0])])
        # Alpha = jnp.dot(t_ray_new, n_bdry_new)
        # Beta = jnp.dot(t_ray_new, t_bdry_new)
        # M = Beta / Alpha
        # rho_new = Y_new[2]
        # zeta_new = Y_new[3]
        # cr_new = c_r(Y_new[0], Y_new[1])
        # cz_new = c_z(Y_new[0], Y_new[1])
        # cn = -1*C_new*(cr_new * zeta_new - cz_new * rho_new)
        # cs = C_new*(cr_new * rho_new + cz_new * zeta_new)
        # N = M*(4*cn -2*M*cs)/C_new**2
        # # set p = p +q*N
        # Y_new = Y_new.at[6].set(Y_new[6] + Y_new[4] * N)
        # Y_new = Y_new.at[7].set(Y_new[7] + Y_new[5] * N)
        # return Y_new
        
        # --- Specular reflection: vector form (chapter-faithful) ---
        C_new = c(Y_new[0], Y_new[1])

        # Incident unit tangent (dr/ds, dz/ds) = c * (rho, zeta)
        tI = C_new * jnp.array([Y_new[2], Y_new[3]])
        rhoI  = tI[0] / C_new
        zetaI = tI[1] / C_new

        # Boundary unit normal / tangent at hit point
        n = self.n_bty(Y_new[0])          # should be unit
        tB = self.t_bty(Y_new[0])         # MUST be unit tangent (see note below)

        # Reflect ray direction: tR = tI - 2 (tI·n) n
        tR = tI - 2.0 * jnp.dot(tI, n) * n

        # Convert back to slowness components
        Y_new = Y_new.at[2].set(tR[0] / C_new)
        Y_new = Y_new.at[3].set(tR[1] / C_new)

        # --- Dynamic reflection update (uses correct α,β decomposition) ---
        # α = component along boundary normal, β = component along boundary tangent
        # Use incident direction for α,β (this is standard in the reflection jump)
        alpha = jnp.dot(tI / (jnp.linalg.norm(tI) + 1e-12), n)
        beta  = jnp.dot(tI / (jnp.linalg.norm(tI) + 1e-12), tB)
        M = beta / (alpha + 1e-12)

        # Sound-speed gradients at the hit point
        cr_new = c_r(Y_new[0], Y_new[1])
        cz_new = c_z(Y_new[0], Y_new[1])

        # Keep your original cn/cs definitions (but now reflection geometry is correct)
        # rho_new  = Y_new[2]
        # zeta_new = Y_new[3]
        rho_new  = rhoI
        zeta_new = zetaI
        cn = -1.0 * C_new * (cr_new * zeta_new - cz_new * rho_new)
        cs =  1.0 * C_new * (cr_new * rho_new + cz_new * zeta_new)

        N = M * (4.0 * cn - 2.0 * M * cs) / (C_new**2)

        # p = p + q*N  (your state: q_r,q_i at [4],[5]; p_r,p_i at [6],[7])
        Y_new = Y_new.at[6].set(Y_new[6] + Y_new[4] * N)
        Y_new = Y_new.at[7].set(Y_new[7] + Y_new[5] * N)
        bottom_coeff = jax.lax.cond(
            REFLECTION_MODEL["bottom_boundary_condition"] == "acoustic_halfspace",
            lambda _: _acoustic_halfspace_reflection_coefficient(
                self.freq,
                C_new,
                beta,
                alpha,
            ),
            lambda _: 1.0 + 0.0j,
            operand=None,
        )
        Y_new = _apply_boundary_reflection(Y_new, bottom_coeff, 10)

        return Y_new


    def top_reflection(self, Y, Y_new):
        # Reflection handling for top (altimetry).
        t_ray = jnp.array([Y[2], Y[3]])
        n_bdry = self.n_ati(Y[0])
        f_current = Y[1] - self.ati(Y[0])
        f_new = Y_new[1] - self.ati(Y_new[0])
        dt_proj = jnp.abs(jnp.dot(- (jnp.array([Y[0], Y[1]]) - jnp.array([Y[0], self.ati(Y[0])])), n_bdry)) \
                    / (jnp.abs(jnp.dot(c(Y[0], Y[1]) * t_ray, n_bdry)) + 1e-12)
        dt_linear = self.dt * jnp.abs(f_current) / (jnp.abs(f_current) + jnp.abs(f_new) + 1e-12)
        dt_tilde = (dt_proj + dt_linear) / 2.0
        
        K = Y + (dt_tilde / 2.0) * self.function(Y)
        Y_new = Y + dt_tilde * self.function(K)
        Y_new = Y_new.at[1].set(self.ati(Y_new[0]))
        
        # alpha = jnp.arctan(jax.grad(self.ati)(Y_new[0]))
        # theta = jnp.arctan(Y_new[3] / Y_new[2])
        # theta_new = -theta - 2 * alpha
        # C_new = c(Y_new[0], Y_new[1])
        # Y_new = Y_new.at[2].set(jnp.cos(theta_new) / C_new)
        # Y_new = Y_new.at[3].set(jnp.sin(theta_new) / C_new)
        
        # # dynamic ray equations update
        # t_ray_new = jnp.array([Y_new[2], Y_new[3]])
        # n_bdry_new = self.n_bty(Y_new[0])
        # t_bdry_new = jnp.array([Y_new[0], self.bty(Y_new[0])])
        # Alpha = jnp.dot(t_ray_new, n_bdry_new)
        # Beta = jnp.dot(t_ray_new, t_bdry_new)
        # M = Beta / Alpha
        # rho_new = Y_new[2]
        # zeta_new = Y_new[3]
        # cr_new = c_r(Y_new[0], Y_new[1])
        # cz_new = c_z(Y_new[0], Y_new[1])
        # cn = -1*C_new*(cr_new * zeta_new - cz_new * rho_new)
        # cs = C_new*(cr_new * rho_new + cz_new * zeta_new)
        # N = M*(4*cn -2*M*cs)/C_new**2
        # # set p = p +q*N
        # Y_new = Y_new.at[6].set(Y_new[6] + Y_new[4] * N)
        # Y_new = Y_new.at[7].set(Y_new[7] + Y_new[5] * N)
        
        # return Y_new

        # --- Specular reflection: vector form (chapter-faithful) ---
        C_new = c(Y_new[0], Y_new[1])

        tI = C_new * jnp.array([Y_new[2], Y_new[3]])
        rhoI  = tI[0] / C_new
        zetaI = tI[1] / C_new

        # IMPORTANT: use TOP boundary normal/tangent (not bottom)   
        n = self.n_ati(Y_new[0])
        tB = self.t_ati(Y_new[0])

        tR = tI - 2.0 * jnp.dot(tI, n) * n

        Y_new = Y_new.at[2].set(tR[0] / C_new)
        Y_new = Y_new.at[3].set(tR[1] / C_new)

        # --- Dynamic reflection update (correct α,β decomposition w.r.t TOP boundary) ---
        alpha = jnp.dot(tI / (jnp.linalg.norm(tI) + 1e-12), n)
        beta  = jnp.dot(tI / (jnp.linalg.norm(tI) + 1e-12), tB)
        M = beta / (alpha + 1e-12)

        cr_new = c_r(Y_new[0], Y_new[1])
        cz_new = c_z(Y_new[0], Y_new[1])

        # rho_new  = Y_new[2]
        # zeta_new = Y_new[3]
        rho_new  = rhoI
        zeta_new = zetaI
        cn = -1.0 * C_new * (cr_new * zeta_new - cz_new * rho_new)
        cs =  1.0 * C_new * (cr_new * rho_new + cz_new * zeta_new)

        N = M * (4.0 * cn - 2.0 * M * cs) / (C_new**2)

        Y_new = Y_new.at[6].set(Y_new[6] + Y_new[4] * N)
        Y_new = Y_new.at[7].set(Y_new[7] + Y_new[5] * N)
        top_coeff = jax.lax.cond(
            REFLECTION_MODEL["top_boundary_condition"] == "vacuum",
            lambda _: -1.0 + 0.0j,
            lambda _: 1.0 + 0.0j,
            operand=None,
        )
        Y_new = _apply_boundary_reflection(Y_new, top_coeff, 9)

        return Y_new


    def __call__(self, Y):
        X0 = jnp.array([Y[0], Y[1]])
        Y_pass = jnp.copy(Y)
        c0 = c(Y[0], Y[1])
        h0 = self._reduce_step(Y, c0, self.dt)

        f0 = self.function(Y)
        K = Y + 0.5 * h0 * f0
        c1 = c(K[0], K[1])
        h1 = self._reduce_step(Y, c1, h0)
        halfh = 0.5 * h0
        w1 = h1 / (2.0 * halfh + 1e-12)
        w0 = 1.0 - w1
        f1 = self.function(K)
        Y_new = Y + h1 * (w0 * f0 + w1 * f1)
        X = jnp.array([Y_new[0], Y_new[1]])
        
        n_bty = self.n_bty(Y[0])
        n_ati = self.n_ati(Y[0])
        
        do_bty = X0 - jnp.array([Y[0], self.bty(Y[0])])
        del0_bty = jnp.dot(-do_bty, n_bty)
        do_ati = X0 - jnp.array([Y[0], self.ati(Y[0])])
        del0_ati = jnp.dot(-do_ati, n_ati)
        d_bty = X - jnp.array([Y_new[0], self.bty(Y_new[0])])
        del_bty = jnp.dot(-d_bty, n_bty)
        d_ati = X - jnp.array([Y_new[0], self.ati(Y_new[0])])
        del_ati = jnp.dot(-d_ati, n_ati)
        
        Y_new = jax.lax.cond(
            jnp.logical_and((del0_bty <= 0.0), (del_bty >= 0.0)),
            lambda Y_pass, Y_new: self.bottom_reflection(Y_pass, Y_new),
            lambda Y_pass, Y_new:  Y_new,
            Y_pass, Y_new
        )
        Y_new = jax.lax.cond(
            jnp.logical_and((del0_ati <= 0.0), (del_ati >= 0.0)),
            lambda Y_pass, Y_new: self.top_reflection(Y_pass, Y_new),
            lambda Y_pass, Y_new: Y_new,
            Y_pass, Y_new
        )
        return Y_new
    
# Rollout function to integrate the ray path.
def rollout(stepper, n, *, include_init: bool = True):
    def scan_fn(Y, _):
        Y_next = stepper(Y)
        theta_logic = jnp.isnan(Y_next).any()
        r = Y_next[0]
        z = Y_next[1]
        top_z = ati(r)
        bottom_z = bty(r)
        r_logic = jnp.logical_or(r < 0.0, r >= 100000.0)
        z_logic = jnp.logical_or(z < top_z - 1.0e-6, z > bottom_z + 1.0e-6)
        backward_logic = jnp.logical_and(REFLECTION_MODEL["kill_backward_rays"], Y_next[2] < 0.0)
        should_break = jnp.logical_or(backward_logic, jnp.logical_or(r_logic, jnp.logical_or(z_logic, theta_logic)))
        return jax.lax.cond(
            should_break,
            lambda _: (Y, Y),
            lambda _: (Y_next, Y_next),
            operand=None
        )
    def rollout_fn(Y_0):
        _, trj = jax.lax.scan(scan_fn, Y_0, None, length=n)
        if include_init:
            return jnp.concatenate([jnp.expand_dims(Y_0, axis=0), trj], axis=0)
        return trj
    return rollout_fn

def compute_single_ray_path(freq, r_s, z_s, theta_0, delta_alpha_opt, ray_eqns=dynamic_ray_eqns, ds=10.0, R_max=100000, Z_max=5000):
    """
    Computes a single ray trajectory using the extended state.
    The state is [r, z, rho, zeta, n_top, n_bottom].
    """
    omega = 2 * jnp.pi * freq
    C0 = c(r_s, z_s)
    rho = jnp.cos(theta_0) / C0
    zeta = jnp.sin(theta_0) / C0
    epsilon = (2*C0**2)/(omega * delta_alpha_opt**2)
    q0_r = 0.0
    q0_i = epsilon
    p0_r = 1.0
    p0_i = 0.0
    tau_0 = 0.0

    Y0 = jnp.array([r_s, z_s, rho, zeta, q0_r, q0_i, p0_r, p0_i, tau_0], dtype=jnp.float64)
    num_steps = int(R_max / ds)
    stepper = jax.jit(RK2_stepper(ray_eqns, ds))
    trj = rollout(stepper, num_steps, include_init=True)(Y0)
    return trj


def _initialize_ray_fan_state(freq, r_s, z_s, theta, *, beam_type='paraxial'):
    omega = 2 * jnp.pi * freq
    delta_alpha = jnp.where(theta.shape[0] > 1, theta[1] - theta[0], 1.0)
    n_theta = len(theta)
    Y0_set = jnp.zeros((n_theta, 13), dtype=jnp.float64)
    C0 = c(r_s, z_s)
    rho = jnp.cos(theta) / C0
    zeta = jnp.sin(theta) / C0
    epsilon = (2 * C0**2) / (omega * delta_alpha**2 + 1e-12)
    bellhop_beam = jnp.logical_or(beam_type == 'bellhop', beam_type == 'geometric')
    q0_r = 0.0
    q0_i = jnp.where(bellhop_beam, 1.0, epsilon)
    p0_r = 1.0
    p0_i = 0.0
    tau_0 = 0.0

    Y0_set = Y0_set.at[:, 0].set(r_s)
    Y0_set = Y0_set.at[:, 1].set(z_s)
    Y0_set = Y0_set.at[:, 2].set(rho)
    Y0_set = Y0_set.at[:, 3].set(zeta)
    Y0_set = Y0_set.at[:, 4].set(q0_r)
    Y0_set = Y0_set.at[:, 5].set(q0_i)
    Y0_set = Y0_set.at[:, 6].set(p0_r)
    Y0_set = Y0_set.at[:, 7].set(p0_i)
    Y0_set = Y0_set.at[:, 8].set(tau_0)
    Y0_set = Y0_set.at[:, 9].set(0.0)
    Y0_set = Y0_set.at[:, 10].set(0.0)
    Y0_set = Y0_set.at[:, 11].set(0.0)
    Y0_set = Y0_set.at[:, 12].set(0.0)
    return Y0_set


def rollout_batched(stepper, n, *, include_init: bool = True):
    batched_step = jax.jit(jax.vmap(lambda y: stepper(y), in_axes=0, out_axes=0))

    def scan_fn(carry, _):
        Y_batch, active = carry
        Y_next_all = batched_step(Y_batch)

        r = Y_next_all[:, 0]
        z = Y_next_all[:, 1]
        top_z = jax.vmap(stepper.ati)(r)
        bottom_z = jax.vmap(stepper.bty)(r)
        nan_logic = jnp.any(jnp.isnan(Y_next_all), axis=1)
        r_logic = jnp.logical_or(r < 0.0, r >= 100000.0)
        z_logic = jnp.logical_or(z < top_z - 1.0e-6, z > bottom_z + 1.0e-6)
        backward_logic = jnp.logical_and(REFLECTION_MODEL["kill_backward_rays"], Y_next_all[:, 2] < 0.0)
        should_break = jnp.logical_and(active, jnp.logical_or(backward_logic, jnp.logical_or(r_logic, jnp.logical_or(z_logic, nan_logic))))

        next_active = jnp.logical_and(active, jnp.logical_not(should_break))
        Y_next = jnp.where((active[:, None]), Y_next_all, Y_batch)
        Y_next = jnp.where(should_break[:, None], Y_batch, Y_next)
        return (Y_next, next_active), Y_next

    def rollout_fn(Y0_batch):
        init_active = jnp.ones((Y0_batch.shape[0],), dtype=bool)
        (_, _), trj = jax.lax.scan(scan_fn, (Y0_batch, init_active), None, length=n)
        if include_init:
            return jnp.concatenate([Y0_batch[None, :, :], trj], axis=0)
        return trj

    return rollout_fn


def compute_multiple_ray_paths_batched(
    freq,
    r_s,
    z_s,
    theta,
    ray_eqns=dynamic_ray_eqns,
    ds=10.0,
    R_max=100000,
    Z_max=5000,
    ati=None,
    bty=None,
    beam_type='paraxial',
):
    """
    HPC-oriented batched tracer.

    This updates the full ray fan as one contiguous tensor with a single
    `lax.scan` over steps and per-ray masking for termination, avoiding the
    older `vmap(scan(single_ray))` structure.
    """
    Y0_set = _initialize_ray_fan_state(freq, r_s, z_s, theta, beam_type=beam_type)
    stepper = RK2_stepper(ray_eqns, ds, freq=freq, ati=ati, bty=bty)
    trj = rollout_batched(stepper, int(R_max / ds), include_init=True)(Y0_set)
    return jnp.swapaxes(trj, 0, 1)

def compute_multiple_ray_paths(
    freq,
    r_s,
    z_s,
    theta,
    ray_eqns=dynamic_ray_eqns,
    ds=10.0,
    R_max=100000,
    Z_max=5000,
    ati=None,
    bty=None,
    beam_type='paraxial',
):
    """
    Computes trajectories for an array of initial angles using vectorized integration.
    """
    return compute_multiple_ray_paths_batched(
        freq,
        r_s,
        z_s,
        theta,
        ray_eqns=ray_eqns,
        ds=ds,
        R_max=R_max,
        Z_max=Z_max,
        ati=ati,
        bty=bty,
        beam_type=beam_type,
    )


def _segment_reflection_factor(
    single_trj,
    is_,
    *,
    surface_reflection_coeff=-1.0 + 0.0j,
    bottom_reflection_coeff=1.0 + 0.0j,
):
    if single_trj.shape[1] >= 13:
        return jnp.exp(single_trj[is_, 11] + 1j * single_trj[is_, 12])
    if single_trj.shape[1] < 11:
        return 1.0 + 0.0j

    n_surface = single_trj[:, 9]
    n_bottom = single_trj[:, 10]
    surface_bounces = jnp.rint(n_surface[is_]).astype(jnp.int32)
    bottom_bounces = jnp.rint(n_bottom[is_]).astype(jnp.int32)
    return (surface_reflection_coeff ** surface_bounces) * (bottom_reflection_coeff ** bottom_bounces)

def compute_caustic(ray_q: jnp.ndarray) -> jnp.ndarray:
    """
    Given ray_q of shape (Nsteps,), where ray_q[i] is the q(1) at step i,
    returns an array caustic_vals of shape (Nsteps,) containing the caustic
    multiplier at each step (starting from 1 at step 0).
    """
    N = ray_q.shape[0]

    def step(carry, i):
        caustic_prev = carry
        # detect zero‐crossing between ray_q[i-1] and ray_q[i]:
        crossed = ((ray_q[i] <= 0.0) & (ray_q[i-1] > 0.0)) | \
                  ((ray_q[i] >= 0.0) & (ray_q[i-1] < 0.0))
        # apply caustic phase change if crossed:
        caustic = jnp.where(crossed, 1j * caustic_prev, caustic_prev)
        return caustic, caustic

    # initialize with caustic = 1 at step 0, then scan from i=1…N-1
    init = 1 + 0j
    _, hist = jax.lax.scan(step, init, jnp.arange(1, N))
    # prepend the initial value
    return jnp.concatenate([jnp.asarray([init], dtype=hist.dtype), hist], axis=0)


def bracket_indices(x_curr, x_next, r_grid):
    """Compute 0‑based bracketed indices [i1, i2] for x_curr→x_next over uniform r_grid."""
    N = r_grid.shape[0]
    delta_r = (r_grid[-1] - r_grid[0]) / (N - 1)
    # MATLAB eps(x) ≈ smallest representable increment above x
    eps = jnp.nextafter(x_curr, jnp.inf) - x_curr
    f_curr = (x_curr - r_grid[0]) / delta_r
    f_next = (x_next - r_grid[0]) / delta_r

    def right(_):
        i1 = jnp.ceil(f_curr + eps)
        i2 = jnp.floor(f_next)
        return i1, i2

    def left(_):
        i1 = jnp.ceil(f_next + eps)
        i2 = jnp.floor(f_curr)
        return i1, i2

    stepping_right = x_next >= x_curr
    i1, i2 = jax.lax.cond(stepping_right, right, left, operand=None)
    i1 = jnp.clip(i1, 0, N-1).astype(jnp.int32)
    i2 = jnp.clip(i2, 0, N-1).astype(jnp.int32)
    return i1, i2


def _resolve_run_mode(run_mode=None, coherent=None):
    if run_mode is not None:
        normalized = str(run_mode).strip().lower()
        if normalized not in ("coherent", "incoherent", "semicoherent"):
            raise ValueError("run_mode must be one of: coherent, incoherent, semicoherent.")
        return normalized
    if coherent is None:
        return "coherent"
    return "coherent" if coherent else "incoherent"


def _validate_solver_inputs(
    freq,
    theta_min,
    theta_max,
    n_beams,
    rr_grid,
    rz_grid,
    ds,
    accumulation_model,
    beam_type,
):
    if not isinstance(freq, jax_core.Tracer) and float(freq) <= 0.0:
        raise ValueError("freq must be positive.")
    if int(n_beams) < 1:
        raise ValueError("n_beams must be at least 1.")
    if not isinstance(ds, jax_core.Tracer) and float(ds) <= 0.0:
        raise ValueError("ds must be positive.")
    if (
        not isinstance(theta_min, jax_core.Tracer)
        and not isinstance(theta_max, jax_core.Tracer)
        and float(theta_max) <= float(theta_min)
    ):
        raise ValueError("theta_max must be greater than theta_min.")
    if rr_grid.ndim != 1 or rz_grid.ndim != 1:
        raise ValueError("rr_grid and rz_grid must be one-dimensional arrays.")
    if rr_grid.shape[0] < 2 or rz_grid.shape[0] < 2:
        raise ValueError("rr_grid and rz_grid must contain at least two points.")
    if not jnp.all(rr_grid[1:] > rr_grid[:-1]):
        raise ValueError("rr_grid must be strictly increasing.")
    if not jnp.all(rz_grid[1:] > rz_grid[:-1]):
        raise ValueError("rz_grid must be strictly increasing.")

    valid_accumulation_models = {"bellhop", "gaussian", "hat"}
    if accumulation_model not in valid_accumulation_models:
        raise ValueError(
            f"accumulation_model must be one of {sorted(valid_accumulation_models)}, got {accumulation_model!r}."
        )
    valid_beam_types = {"geometric"}
    if beam_type not in valid_beam_types:
        raise ValueError(f"beam_type must be one of {sorted(valid_beam_types)}, got {beam_type!r}.")


def _resolve_propagation_limits(rr_grid, rz_grid, R_max, Z_max):
    resolved_r_max = float(rr_grid[-1]) if R_max is None else float(R_max)
    resolved_z_max = float(rz_grid[-1]) if Z_max is None else float(Z_max)
    return resolved_r_max, resolved_z_max


def _resolve_precision(precision: str):
    normalized = str(precision).strip().lower()
    if normalized == "float32":
        return normalized, jnp.float32, jnp.complex64
    if normalized == "float64":
        return normalized, jnp.float64, jnp.complex128
    raise ValueError("precision must be 'float32' or 'float64'.")


def _prepare_launch_fan(
    freq,
    r_s,
    z_s,
    theta_min,
    theta_max,
    n_beams,
    *,
    auto_beam_count,
    source_beam_pattern_angles_deg,
    source_beam_pattern_db,
    dtype,
    r_max,
):
    c0 = c(r_s, z_s)
    n_beams_recommended = bellhop_recommended_nbeams(freq, c0, r_max, theta_min, theta_max)
    n_beams_eff = int(n_beams_recommended) if auto_beam_count else int(n_beams)
    theta = jnp.linspace(theta_min, theta_max, n_beams_eff, dtype=dtype)
    source_amplitudes = evaluate_source_beam_pattern(theta, source_beam_pattern_angles_deg, source_beam_pattern_db).astype(dtype)
    weights = jnp.ones_like(theta)
    if n_beams_eff > 1:
        weights = weights.at[0].set(0.5).at[-1].set(0.5)
    return theta, weights.astype(dtype), source_amplitudes, n_beams_recommended


def _trace_beam_chunk(
    freq,
    r_s,
    z_s,
    theta_chunk,
    *,
    ds,
    r_max,
    z_max,
    ati,
    bty,
    beam_type,
):
    return compute_multiple_ray_paths(
        freq,
        r_s,
        z_s,
        theta_chunk,
        ds=ds,
        R_max=r_max,
        Z_max=z_max,
        ati=ati,
        bty=bty,
        beam_type=beam_type,
    )


def _make_bellhop_beam_solver(
    accumulation_model,
    *,
    freq,
    z_s,
    delta_alpha,
    rr_grid,
    rz_grid,
    run_mode,
    min_width_m,
):
    run_type_e = 'C' if run_mode == 'coherent' else 'S'
    bellhop_style = accumulation_model in ('hat', 'gaussian')

    if accumulation_model == 'gaussian':
        solver = jax.vmap(
            lambda trj, launch_angle, source_amplitude: InfluenceGeoGaussian(
                freq, trj, z_s, launch_angle, source_amplitude, delta_alpha, rr_grid, rz_grid, RunTypeE=run_type_e
            ),
            in_axes=(0, 0, 0),
        )
    elif accumulation_model == 'hat':
        solver = jax.vmap(
            lambda trj, launch_angle, source_amplitude: InfluenceGeoHat(
                freq, trj, z_s, launch_angle, source_amplitude, delta_alpha, rr_grid, rz_grid, RunTypeE=run_type_e
            ),
            in_axes=(0, 0, 0),
        )
    else:
        solver = jax.vmap(
            lambda trj, launch_angle, source_amplitude: accumulate_geometric_gaussian_field(
                freq,
                trj,
                launch_angle,
                source_amplitude,
                delta_alpha,
                rr_grid,
                rz_grid,
                min_width_m,
                coherent=run_mode == 'coherent',
            ),
            in_axes=(0, 0, 0),
        )
    return solver, bellhop_style

@functools.partial(jax.jit, static_argnames=('RunTypeE',))
def InfluenceGeoGaussian(freq, single_trj, z_s, source_takeoff_angle, source_amplitude, delta_alpha, rr_grid, rz_grid, RunTypeE='S'):
    
    omega = 2 * jnp.pi * freq
    IBWin = 4
    DS  = jnp.sqrt(2.0)* jnp.sin( omega * z_s * single_trj[0, 3])
    C =  c(single_trj[0, 0], single_trj[0, 1])
    q0  = C/ delta_alpha
    lambda_ = C / freq
    launch_cosine = jnp.sqrt(jnp.maximum(jnp.abs(jnp.cos(source_takeoff_angle)), 1e-12))
    Rat1 = jax.lax.cond(
        REFLECTION_MODEL["source_type"] == "point",
        lambda _: launch_cosine / jnp.sqrt(2 * jnp.pi),
        lambda _: 1 / jnp.sqrt(2 * jnp.pi),
        operand=None,
    )
    Nsteps = single_trj.shape[0]
    Nrr = rr_grid.shape[0]
    Nrz = rz_grid.shape[0]
    
    ray_q = jnp.real(single_trj[:, 4])

    ray_x = jnp.real(single_trj[:, :2])
    ray_tau = jnp.real(single_trj[:, 8])
    
    init_c = 1 + 0j
    init_U = jnp.zeros((Nrz, Nrr), dtype=jnp.complex128)
    
    def body_is(carry, is_):
        caustic, U = carry

        # --- caustic detection & phase flip ---
        def flip(c):
            return 1j * c
        # print("dtype of is_:", is_.dtype)
        # crossed = ((ray_q[is_] <= 0) & (ray_q[is_-1] > 0)) | \
        #           ((ray_q[is_] >= 0) & (ray_q[is_-1] < 0))
        
        crossed = ((ray_q[is_] <= 0) & (ray_q[is_-1] > 0)) | \
                  ((ray_q[is_] >= 0) & (ray_q[is_-1] < 0))
        caustic = jax.lax.cond((is_ > 0) & crossed, flip, lambda c: c, caustic)

        # --- bracket receiver ranges ---
        x_curr = ray_x[is_, 0]
        x_next = ray_x[is_+1, 0]
        ir1, ir2 = bracket_indices(x_curr, x_next, rr_grid)

        # print("ir1 dtype:", ir1.dtype, "ir2:", ir2.dtype)

        # Only proceed if there is at least one bracketed point
        def no_op(carry):
            return carry
        
        def process_ranges(carry):
            caustic_loc, U_loc = carry
            
            # print("U loc dtype:", U_loc.dtype)

            # beam tangent & normal
            C_curr = c(ray_x[is_, 0], ray_x[is_, 1])
            t_ray = jnp.array([C_curr * single_trj[is_, 2], C_curr * single_trj[is_, 3]])
            rlen = jnp.linalg.norm(t_ray)
            t_ray_unit = t_ray / rlen
            t_ray_scaled = t_ray_unit / rlen
            n_ray = jnp.array([-t_ray_unit[1], t_ray_unit[0]])

            # iterate over receiver indices
            def body_ir(ir, U_inner):
                # print(" U inner dtype:", U_inner.dtype)
                # print("ir dtype:", ir.dtype)
                # receiver point (z varies along second dim)
                z_pts = rz_grid
                # xrcvr: shape (Nz,2)
                xrcvr = jnp.stack([rr_grid[ir] * jnp.ones_like(z_pts), z_pts], axis=-1)
                # beam point xray: shape (Nz,2)
                xray = ray_x[is_]  # broadcast by vectorization

                # along‐ray and normal distances
                delta = xrcvr - xray
                s = delta @ t_ray_scaled
                n = jnp.abs(delta @ n_ray)

                # # interpolated q and radius sigma
                # q0 = ray_q[0]
                # q_i = ray_q[is_] + s * (ray_q[is_+1] - ray_q[is_])
                # sigma = jnp.abs(q_i / q0)
                # sigma_lim = jnp.minimum(0.2 * freq * ray_tau[is_] / lambda_, jnp.pi * lambda_)
                # sigma = jnp.where(sigma < sigma_lim, sigma_lim, sigma)
                
                # interpolated complex q and beam radius sigma
                q_seg = ray_q[is_] + s * (ray_q[is_+1] - ray_q[is_])
                sigma = jnp.abs(q_seg) / (q0 + 1e-12)
                sigma_lim = jnp.minimum(0.2 * freq * ray_tau[is_] / lambda_, jnp.pi * lambda_)
                sigma = jnp.where(sigma < sigma_lim, sigma_lim, sigma)

                # select depth‑indices where beam contributes
                mask = n < IBWin * sigma

                def no_contrib(Uc):
                    return Uc

                def contrib(Uc):
                    A = jnp.abs(q0 / (q_seg + 1e-12))
                    delay = ray_tau[is_] + s * (ray_tau[is_+1] - ray_tau[is_])

                    # caustic phase per sub‐beam
                    caust = caustic_loc * jnp.ones_like(q_seg, dtype=jnp.complex128)
                    flip_sub = ((q_seg <= 0) & (ray_q[is_] > 0)) | ((q_seg >= 0) & (ray_q[is_] < 0))
                    caust = jnp.where(flip_sub, 1j * caust, caust)
                    const = (
                        source_amplitude
                        * Rat1
                        * jnp.sqrt(c(ray_x[is_, 0], ray_x[is_, 1]) / (jnp.abs(q_seg) + 1e-12))
                        * caust
                    )
                    const = const * _segment_reflection_factor(single_trj, is_)

                    if RunTypeE == 'S':
                        const = DS * const

                    amp = const * jnp.exp(-0.5 * (n / sigma)**2) / (sigma * A)

                    if RunTypeE == 'C':  # coherent TL
                        contri = amp * jnp.exp(-1j * 2*jnp.pi*freq * delay)
                        return Uc.at[:, ir].add(jnp.where(mask, contri, 0.0 + 0.0j))
                    else:               # incoherent / semi‐coherent
                        W = jnp.exp(-0.5 * (n / sigma)**2) / (2.*sigma*A)
                        W_safe = jnp.where(mask, W, 1.0)
                        contri = (jnp.abs(amp) * jnp.exp(2*jnp.pi*freq * jnp.imag(delay)) / W_safe)**2 * W_safe
                        return Uc.at[:, ir].add(jnp.where(mask, contri + 0.0j, 0.0 + 0.0j))

                return jax.lax.cond(mask.any(), contrib, no_contrib, U_inner)
            
            # run inner loop
            U_processed = jax.lax.fori_loop(ir1, ir2+1, body_ir, U_loc)
            return (caustic_loc, U_processed)

        caustic, U = jax.lax.cond(ir2 >= ir1,
                                  process_ranges,
                                  no_op,
                              operand=(caustic, U))

        return (caustic, U), None

    # Scan over steps 0…Nsteps-2
    (final_c, final_U), _ = jax.lax.scan(body_is, (init_c, init_U), jnp.arange(Nsteps-1))
    
    return final_U


@functools.partial(jax.jit, static_argnames=('RunTypeE',))
def InfluenceGeoHat(freq, single_trj, z_s, source_takeoff_angle, source_amplitude, delta_alpha, rr_grid, rz_grid, RunTypeE='S'):
    omega = 2 * jnp.pi * freq
    DS = jnp.sqrt(2.0) * jnp.sin(omega * z_s * single_trj[0, 3])
    C = c(single_trj[0, 0], single_trj[0, 1])
    q0 = C / delta_alpha
    Rat1 = jax.lax.cond(
        REFLECTION_MODEL["source_type"] == "point",
        lambda _: jnp.sqrt(jnp.maximum(jnp.abs(jnp.cos(source_takeoff_angle)), 1e-12)),
        lambda _: 1.0,
        operand=None,
    )

    n_steps = single_trj.shape[0]
    n_rr = rr_grid.shape[0]
    n_rz = rz_grid.shape[0]
    ray_q = jnp.real(single_trj[:, 4])
    ray_tau = jnp.real(single_trj[:, 8])
    ray_x = jnp.real(single_trj[:, :2])

    init_c = 1.0 + 0.0j
    init_u = jnp.zeros((n_rz, n_rr), dtype=jnp.complex128)

    def body_is(carry, is_):
        caustic, field = carry

        crossed = ((ray_q[is_] <= 0.0) & (ray_q[is_ - 1] > 0.0)) | ((ray_q[is_] >= 0.0) & (ray_q[is_ - 1] < 0.0))
        caustic = jax.lax.cond(is_ > 0, lambda c0: jnp.where(crossed, 1j * c0, c0), lambda c0: c0, caustic)

        x_curr = ray_x[is_, 0]
        x_next = ray_x[is_ + 1, 0]
        ir1, ir2 = bracket_indices(x_curr, x_next, rr_grid)

        def no_op(state):
            return state

        def process_ranges(state):
            caustic_loc, field_loc = state
            c_curr = c(ray_x[is_, 0], ray_x[is_, 1])
            tray = ray_x[is_ + 1] - ray_x[is_]
            rlen = jnp.linalg.norm(tray) + 1e-12
            tray = tray / rlen
            tray_scaled = tray / rlen
            nray = jnp.array([-tray[1], tray[0]])

            def body_ir(ir, field_inner):
                xrcvr = jnp.stack([jnp.full_like(rz_grid, rr_grid[ir]), rz_grid], axis=-1)
                delta = xrcvr - ray_x[is_]
                s = delta @ tray_scaled
                n = jnp.abs(delta @ nray)

                q = ray_q[is_] + s * (ray_q[is_ + 1] - ray_q[is_])
                rad_max = jnp.abs(q / (q0 + 1e-12))
                mask = n < rad_max

                def no_contrib(fc):
                    return fc

                def add_contrib(fc):
                    A = 1.0 / (rad_max + 1e-12)
                    delay = ray_tau[is_] + s * (ray_tau[is_ + 1] - ray_tau[is_])
                    caust = caustic_loc * jnp.ones_like(q, dtype=jnp.complex128)
                    flip_sub = ((q <= 0.0) & (ray_q[is_] > 0.0)) | ((q >= 0.0) & (ray_q[is_] < 0.0))
                    caust = jnp.where(flip_sub, 1j * caust, caust)
                    const = source_amplitude * Rat1 * jnp.sqrt(c_curr / (jnp.abs(q) + 1e-12)) * A * caust
                    const = const * _segment_reflection_factor(single_trj, is_)
                    const = jax.lax.cond(RunTypeE == 'S', lambda cc: DS * cc, lambda cc: cc, const)
                    amp = const * (rad_max - n)

                    if RunTypeE == 'C':
                        contri = amp * jnp.exp(-1j * omega * delay)
                        return fc.at[:, ir].add(jnp.where(mask, contri, 0.0 + 0.0j))

                    W = (rad_max - n) / (rad_max + 1e-12)
                    W_safe = jnp.where(mask, W, 1.0)
                    contri = (jnp.abs(amp) * jnp.exp(omega * jnp.imag(delay)) / (W_safe + 1e-12)) ** 2 * W_safe
                    return fc.at[:, ir].add(jnp.where(mask, contri + 0.0j, 0.0 + 0.0j))

                return jax.lax.cond(jnp.any(mask), add_contrib, no_contrib, field_inner)

            field_loc = jax.lax.fori_loop(ir1, ir2 + 1, body_ir, field_loc)
            return caustic_loc, field_loc

        active = _segment_alive(single_trj, is_)
        should_process = jnp.logical_and(active, ir2 >= ir1)
        return jax.lax.cond(should_process, process_ranges, no_op, (caustic, field)), None

    (_, final_u), _ = jax.lax.scan(body_is, (init_c, init_u), jnp.arange(n_steps - 1))
    return final_u


@functools.partial(jax.jit, static_argnames=('run_mode', 'line_source'))
def scale_pressure_bellhop(freq, delta_alpha, c0, rr_grid, field, run_mode='coherent', line_source=False):
    coherent = run_mode == 'coherent'
    const = jax.lax.cond(
        coherent,
        lambda _: -delta_alpha * jnp.sqrt(freq) / (c0 + 1e-12),
        lambda _: -1.0,
        operand=None,
    )
    safe_r = jnp.maximum(rr_grid, 1e-12)
    scaled = field
    if not coherent:
        scaled = jnp.sqrt(jnp.maximum(jnp.real(scaled), 0.0)) + 0.0j
    factor = jax.lax.cond(
        line_source,
        lambda _: -4.0 * jnp.sqrt(jnp.pi) * const * jnp.ones_like(rr_grid),
        lambda _: jnp.where(rr_grid > 0.0, const / jnp.sqrt(safe_r), 0.0),
        operand=None,
    )
    return scaled * factor[None, :]


def _segment_alive(single_trj, is_):
    curr = single_trj[is_]
    nxt = single_trj[is_ + 1]
    moved = jnp.logical_not(jnp.all(curr[:2] == nxt[:2]))
    finite = jnp.logical_and(jnp.isfinite(curr).all(), jnp.isfinite(nxt).all())
    return jnp.logical_and(moved, finite)


@functools.partial(jax.jit, static_argnames=('coherent',))
def accumulate_geometric_gaussian_field(
    freq,
    single_trj,
    source_takeoff_angle,
    source_amplitude,
    delta_alpha,
    rr_grid,
    rz_grid,
    min_width_m,
    coherent=True,
):
    """
    Bellhop-style 2D geometric Gaussian beam accumulation.

    This follows Porter (2019), Sec. II.C:
    - geometric beam width: W(s) = |q(s) * delta_alpha / c(0)|
    - geometric Gaussian shape: exp(-0.5 * (n / W)^2)
    - cylindrical spreading amplitude: 1/sqrt(2*pi) * sqrt(cos(alpha) * c(s) / (r * q(s)))

    The implementation adds a configurable beam-width "stent" to regularize
    caustics while remaining differentiable.
    """
    omega = 2.0 * jnp.pi * freq
    c0 = c(single_trj[0, 0], single_trj[0, 1])
    cos_alpha = jnp.maximum(jnp.cos(source_takeoff_angle), 1e-12)
    n_steps = single_trj.shape[0]
    n_rr = rr_grid.shape[0]
    n_rz = rz_grid.shape[0]
    ray_q = jnp.real(single_trj[:, 4])
    ray_tau = jnp.real(single_trj[:, 8])
    ray_x = jnp.real(single_trj[:, :2])

    init_c = 1.0 + 0.0j
    init_u = jnp.zeros((n_rz, n_rr), dtype=jnp.complex128)

    def body_is(carry, is_):
        caustic, field = carry

        crossed = jnp.logical_or(
            jnp.logical_and(ray_q[is_] <= 0.0, ray_q[is_ - 1] > 0.0),
            jnp.logical_and(ray_q[is_] >= 0.0, ray_q[is_ - 1] < 0.0),
        )
        caustic = jax.lax.cond(is_ > 0, lambda c: jnp.where(crossed, 1j * c, c), lambda c: c, caustic)

        x_curr = ray_x[is_, 0]
        x_next = ray_x[is_ + 1, 0]
        ir1, ir2 = bracket_indices(x_curr, x_next, rr_grid)

        def no_op(state):
            return state

        def process_ranges(state):
            caustic_loc, field_loc = state
            c_curr = c(ray_x[is_, 0], ray_x[is_, 1])
            tangent = c_curr * jnp.array([single_trj[is_, 2], single_trj[is_, 3]])
            t_norm = jnp.linalg.norm(tangent) + 1e-12
            t_hat = tangent / t_norm
            n_hat = jnp.array([-t_hat[1], t_hat[0]])

            def body_ir(ir, field_inner):
                x_receiver = jnp.stack(
                    [jnp.full_like(rz_grid, rr_grid[ir]), rz_grid],
                    axis=-1,
                )
                delta = x_receiver - ray_x[is_]
                s_local = delta @ t_hat
                n_local = delta @ n_hat

                q_interp = ray_q[is_] + s_local * (ray_q[is_ + 1] - ray_q[is_])
                tau_interp = ray_tau[is_] + s_local * (ray_tau[is_ + 1] - ray_tau[is_])
                width = jnp.abs(q_interp) * delta_alpha / (c0 + 1e-12)
                width_eff = jnp.maximum(width, min_width_m)
                beam_mask = jnp.abs(n_local) <= 4.0 * width_eff

                def no_contrib(field_current):
                    return field_current

                def add_contrib(field_current):
                    range_safe = jnp.maximum(rr_grid[ir], 1.0)
                    amp0 = source_amplitude * (1.0 / jnp.sqrt(2.0 * jnp.pi)) * caustic_loc
                    amp0 = amp0 * jnp.sqrt(cos_alpha * c_curr / (range_safe * (jnp.abs(q_interp) + 1e-12)))
                    shape = jnp.exp(-0.5 * (n_local / width_eff) ** 2)
                    phase = jnp.exp(-1j * omega * tau_interp)
                    contrib = amp0 * shape * phase
                    contrib = jnp.where(beam_mask, contrib, 0.0 + 0.0j)
                    if coherent:
                        return field_current.at[:, ir].add(contrib)
                    return field_current.at[:, ir].add(jnp.abs(contrib) ** 2 + 0.0j)

                return jax.lax.cond(jnp.any(beam_mask), add_contrib, no_contrib, field_inner)

            field_loc = jax.lax.fori_loop(ir1, ir2 + 1, body_ir, field_loc)
            return caustic_loc, field_loc

        is_active = _segment_alive(single_trj, is_)
        should_process = jnp.logical_and(is_active, ir2 >= ir1)
        return jax.lax.cond(should_process, process_ranges, no_op, (caustic, field)), None

    (final_c, final_u), _ = jax.lax.scan(body_is, (init_c, init_u), jnp.arange(n_steps - 1))
    return final_u


def trace_beam_fan(
    freq,
    r_s,
    z_s,
    theta_min,
    theta_max,
    n_beams,
    *,
    ds=10.0,
    R_max=100000.0,
    Z_max=5000.0,
    beam_type='geometric',
    auto_beam_count=False,
    source_beam_pattern_angles_deg=None,
    source_beam_pattern_db=None,
    ati=ati,
    bty=bty,
):
    """
    Trace a launch fan suitable for Bellhop-style beam summation.
    """
    c0 = c(r_s, z_s)
    if auto_beam_count:
        n_beams_eff = int(bellhop_recommended_nbeams(freq, c0, R_max, theta_min, theta_max))
    else:
        n_beams_eff = int(n_beams)
    theta = jnp.linspace(theta_min, theta_max, n_beams_eff, dtype=jnp.float64)
    source_amplitudes = evaluate_source_beam_pattern(theta, source_beam_pattern_angles_deg, source_beam_pattern_db)
    weights = jnp.ones_like(theta)
    weights = jax.lax.cond(
        theta.shape[0] > 1,
        lambda w: w.at[0].set(0.5).at[-1].set(0.5),
        lambda w: w,
        weights,
    )
    trj_set = compute_multiple_ray_paths(
        freq,
        r_s,
        z_s,
        theta,
        ds=ds,
        R_max=R_max,
        Z_max=Z_max,
        ati=ati,
        bty=bty,
        beam_type=beam_type,
    )
    return theta, weights, source_amplitudes, trj_set


def solve_transmission_loss(
    freq,
    r_s,
    z_s,
    theta_min,
    theta_max,
    n_beams,
    rr_grid,
    rz_grid,
    *,
    ds=10.0,
    beam_type='geometric',
    coherent=True,
    run_mode=None,
    min_width_wavelengths=0.5,
    accumulation_model='bellhop',
    auto_beam_count=False,
    source_beam_pattern_angles_deg=None,
    source_beam_pattern_db=None,
    store_field_per_beam=False,
    store_trajectories=True,
    beam_chunk_size=None,
    accumulation_backend="windowed",
    precision="float64",
    R_max=None,
    Z_max=None,
    ati=ati,
    bty=bty,
):
    """
    Solve a 2D transmission-loss field using the Bellhop-oriented path.

    This entry point is intended for validation and Bellhop-comparison
    workflows. It performs:

    1. launch-fan construction
    2. batched ray tracing
    3. beam influence accumulation on the receiver grid
    4. Bellhop-style pressure scaling and TL conversion

    Parameters
    ----------
    freq, r_s, z_s:
        Source frequency and position.
    theta_min, theta_max, n_beams:
        Launch-fan definition.
    rr_grid, rz_grid:
        Receiver range/depth grids. Both must be strictly increasing.
    ds:
        Arc-length step size.
    beam_type:
        Currently ``"geometric"``.
    coherent, run_mode:
        Legacy boolean and explicit run-mode interface. ``run_mode`` is
        preferred and may be ``"coherent"``, ``"incoherent"``, or
        ``"semicoherent"``.
    accumulation_model:
        ``"bellhop"``, ``"gaussian"``, or ``"hat"``.

    Returns
    -------
    dict
        Dictionary containing launch angles, trajectories, complex field
        arrays, and TL in dB.
    """
    run_mode = _resolve_run_mode(run_mode, coherent)
    _validate_solver_inputs(
        freq,
        theta_min,
        theta_max,
        n_beams,
        rr_grid,
        rz_grid,
        ds,
        accumulation_model,
        beam_type,
    )
    R_max, Z_max = _resolve_propagation_limits(rr_grid, rz_grid, R_max, Z_max)
    precision_mode, real_dtype, complex_dtype = _resolve_precision(precision)
    freq = real_dtype(freq)
    r_s = real_dtype(r_s)
    z_s = real_dtype(z_s)
    theta_min = real_dtype(theta_min)
    theta_max = real_dtype(theta_max)
    rr_grid = jnp.asarray(rr_grid, dtype=real_dtype)
    rz_grid = jnp.asarray(rz_grid, dtype=real_dtype)

    t_launch_start = time.perf_counter()
    theta, launch_weights, source_amplitudes, n_beams_recommended = _prepare_launch_fan(
        freq,
        r_s,
        z_s,
        theta_min,
        theta_max,
        n_beams,
        auto_beam_count=auto_beam_count,
        source_beam_pattern_angles_deg=source_beam_pattern_angles_deg,
        source_beam_pattern_db=source_beam_pattern_db,
        dtype=real_dtype,
        r_max=R_max,
    )
    launch_fan_s = time.perf_counter() - t_launch_start

    c0 = c(r_s, z_s)
    delta_alpha = jnp.where(theta.shape[0] > 1, theta[1] - theta[0], 1.0)
    wavelength = c0 / freq
    min_width_m = min_width_wavelengths * wavelength
    if accumulation_backend not in {"windowed", "dense"}:
        raise ValueError("accumulation_backend must be 'windowed' or 'dense'.")

    beam_solver, bellhop_style = _make_bellhop_beam_solver(
        accumulation_model,
        freq=freq,
        z_s=z_s,
        delta_alpha=delta_alpha,
        rr_grid=rr_grid,
        rz_grid=rz_grid,
        run_mode=run_mode,
        min_width_m=min_width_m,
    )
    n_total_beams = int(theta.shape[0])
    effective_beam_chunk_size = n_total_beams if beam_chunk_size is None else max(1, min(int(beam_chunk_size), n_total_beams))
    trace_time_s = 0.0
    accumulation_time_s = 0.0

    if accumulation_backend == "dense":
        t_trace_start = time.perf_counter()
        trajectories = _trace_beam_chunk(
            freq,
            r_s,
            z_s,
            theta,
            ds=ds,
            r_max=R_max,
            z_max=Z_max,
            ati=ati,
            bty=bty,
            beam_type=beam_type,
        )
        trace_time_s = time.perf_counter() - t_trace_start

        t_accum_start = time.perf_counter()
        field_per_beam = beam_solver(trajectories, theta, source_amplitudes)
        if not bellhop_style:
            field_per_beam = field_per_beam * launch_weights[:, None, None]
        field_total_raw = jnp.sum(field_per_beam, axis=0)
        accumulation_time_s = time.perf_counter() - t_accum_start
        if not store_field_per_beam:
            field_per_beam = None
        if not store_trajectories:
            trajectories = None
    else:
        field_total_raw = jnp.zeros((rz_grid.shape[0], rr_grid.shape[0]), dtype=complex_dtype)
        stored_field_chunks = [] if store_field_per_beam else None
        stored_trj_chunks = [] if store_trajectories else None

        for start in range(0, n_total_beams, effective_beam_chunk_size):
            stop = min(start + effective_beam_chunk_size, n_total_beams)
            theta_chunk = theta[start:stop]
            weight_chunk = launch_weights[start:stop]
            amplitude_chunk = source_amplitudes[start:stop]

            t_trace_start = time.perf_counter()
            trj_chunk = _trace_beam_chunk(
                freq,
                r_s,
                z_s,
                theta_chunk,
                ds=ds,
                r_max=R_max,
                z_max=Z_max,
                ati=ati,
                bty=bty,
                beam_type=beam_type,
            )
            trace_time_s += time.perf_counter() - t_trace_start

            t_accum_start = time.perf_counter()
            field_chunk = beam_solver(trj_chunk, theta_chunk, amplitude_chunk)
            if not bellhop_style:
                field_chunk = field_chunk * weight_chunk[:, None, None]
            field_total_raw = field_total_raw + jnp.sum(field_chunk, axis=0)
            accumulation_time_s += time.perf_counter() - t_accum_start

            if store_field_per_beam:
                stored_field_chunks.append(field_chunk)
            if store_trajectories:
                stored_trj_chunks.append(trj_chunk)

        field_per_beam = None
        if store_field_per_beam:
            field_per_beam = jnp.concatenate(stored_field_chunks, axis=0)

        trajectories = None
        if store_trajectories:
            trajectories = jnp.concatenate(stored_trj_chunks, axis=0)

    field_total = scale_pressure_bellhop(freq, delta_alpha, c0, rr_grid, field_total_raw, run_mode=run_mode)
    energy = jnp.abs(field_total)
    tl_db = -20.0 * jnp.log10(energy + 1e-16)

    return {
        'theta': theta,
        'launch_weights': launch_weights,
        'source_amplitudes': source_amplitudes,
        'n_beams_recommended': n_beams_recommended,
        'trajectories': trajectories,
        'field_per_beam': field_per_beam,
        'field_total_raw': field_total_raw,
        'field_total': field_total,
        'tl_db': tl_db,
        'timings_s': {
            'launch_fan': launch_fan_s,
            'ray_rollout': trace_time_s,
            'accumulation': accumulation_time_s,
            'total_solver': launch_fan_s + trace_time_s + accumulation_time_s,
        },
        'storage': {
            'store_field_per_beam': bool(store_field_per_beam),
            'store_trajectories': bool(store_trajectories),
            'beam_chunk_size_requested': None if beam_chunk_size is None else int(beam_chunk_size),
            'beam_chunk_size_used': int(effective_beam_chunk_size),
            'accumulation_backend': accumulation_backend,
            'precision': precision_mode,
        },
    }


def _smooth_segment_window(rr_grid, x0, x1, softness_m):
    x_lo = jnp.minimum(x0, x1)
    x_hi = jnp.maximum(x0, x1)
    left = jax.nn.sigmoid((rr_grid - x_lo) / (softness_m + 1e-12))
    right = jax.nn.sigmoid((x_hi - rr_grid) / (softness_m + 1e-12))
    return left * right


@functools.partial(jax.jit, static_argnames=("run_mode",))
def accumulate_geometric_gaussian_field_autodiff(
    freq,
    single_trj,
    source_takeoff_angle,
    source_amplitude,
    delta_alpha,
    rr_grid,
    rz_grid,
    min_width_m,
    range_window_softness_m,
    run_mode="coherent",
):
    omega = 2.0 * jnp.pi * freq
    c0 = c(single_trj[0, 0], single_trj[0, 1])
    cos_alpha = jnp.maximum(jnp.cos(source_takeoff_angle), 1e-12)
    ray_q = jnp.real(single_trj[:, 4])
    ray_tau = jnp.real(single_trj[:, 8])
    ray_x = jnp.real(single_trj[:, :2])
    n_steps = single_trj.shape[0]
    rr_mat = jnp.broadcast_to(rr_grid[None, :], (rz_grid.shape[0], rr_grid.shape[0]))
    rz_mat = jnp.broadcast_to(rz_grid[:, None], (rz_grid.shape[0], rr_grid.shape[0]))
    coherent = run_mode == "coherent"

    def body_fn(carry, is_):
        field, caustic = carry
        curr = ray_x[is_]
        nxt = ray_x[is_ + 1]
        delta = nxt - curr
        seg_len = jnp.linalg.norm(delta) + 1e-12
        t_hat = delta / seg_len
        t_scaled = t_hat / seg_len
        n_hat = jnp.array([-t_hat[1], t_hat[0]])

        crossed = jnp.logical_or(
            jnp.logical_and(ray_q[is_] <= 0.0, ray_q[jnp.maximum(is_ - 1, 0)] > 0.0),
            jnp.logical_and(ray_q[is_] >= 0.0, ray_q[jnp.maximum(is_ - 1, 0)] < 0.0),
        )
        caustic = jax.lax.cond(is_ > 0, lambda c: jnp.where(crossed, 1j * c, c), lambda c: c, caustic)

        delta_r = rr_mat - curr[0]
        delta_z = rz_mat - curr[1]
        s_local = delta_r * t_scaled[0] + delta_z * t_scaled[1]
        n_local = delta_r * n_hat[0] + delta_z * n_hat[1]

        q_interp = ray_q[is_] + s_local * (ray_q[is_ + 1] - ray_q[is_])
        tau_interp = ray_tau[is_] + s_local * (ray_tau[is_ + 1] - ray_tau[is_])
        width = jnp.abs(q_interp) * delta_alpha / (c0 + 1e-12)
        width_eff = jnp.maximum(width, min_width_m)
        range_weight = _smooth_segment_window(rr_grid, curr[0], nxt[0], range_window_softness_m)[None, :]

        amp0 = source_amplitude * (1.0 / jnp.sqrt(2.0 * jnp.pi)) * caustic
        amp0 = amp0 * jnp.sqrt(cos_alpha * c(curr[0], curr[1]) / (jnp.maximum(rr_mat, 1.0) * (jnp.abs(q_interp) + 1e-12)))
        amp0 = amp0 * _segment_reflection_factor(single_trj, is_)
        shape = jnp.exp(-0.5 * (n_local / width_eff) ** 2)
        phase = jnp.exp(-1j * omega * tau_interp)
        contrib = amp0 * shape * phase * range_weight

        field = jax.lax.cond(
            coherent,
            lambda fld: fld + contrib,
            lambda fld: fld + (jnp.abs(contrib) ** 2 + 0.0j),
            field,
        )
        return (field, caustic), None

    init_field = jnp.zeros((rz_grid.shape[0], rr_grid.shape[0]), dtype=jnp.complex128)
    (field, _), _ = jax.lax.scan(body_fn, (init_field, 1.0 + 0.0j), jnp.arange(n_steps - 1))
    return field


def solve_transmission_loss_autodiff(
    freq,
    r_s,
    z_s,
    theta_min,
    theta_max,
    n_beams,
    rr_grid,
    rz_grid,
    *,
    ds=10.0,
    beam_type="geometric",
    run_mode="coherent",
    auto_beam_count=False,
    min_width_wavelengths=0.5,
    range_window_softness_m=None,
    source_beam_pattern_angles_deg=None,
    source_beam_pattern_db=None,
    R_max=None,
    Z_max=None,
    ati=ati,
    bty=bty,
):
    """
    Solve a 2D transmission-loss field with an autodiff-safe accumulation path.

    This path avoids discrete receiver bracketing and hard receiver masks by
    using smooth segment windows and full-grid Gaussian influence evaluation.
    It is intended for gradient-based inversion/training workloads.

    Compared with ``solve_transmission_loss(...)``, this routine prioritizes:

    - stable reverse-mode differentiation
    - JIT/vmap friendliness
    - smooth receiver-grid accumulation

    over exact Bellhop feature parity.
    """
    run_mode = _resolve_run_mode(run_mode, None)
    _validate_solver_inputs(
        freq,
        theta_min,
        theta_max,
        n_beams,
        rr_grid,
        rz_grid,
        ds,
        "bellhop",
        beam_type,
    )
    R_max, Z_max = _resolve_propagation_limits(rr_grid, rz_grid, R_max, Z_max)
    theta, launch_weights, source_amplitudes, trj_set = trace_beam_fan(
        freq,
        r_s,
        z_s,
        theta_min,
        theta_max,
        n_beams,
        ds=ds,
        R_max=R_max,
        Z_max=Z_max,
        beam_type=beam_type,
        auto_beam_count=auto_beam_count,
        source_beam_pattern_angles_deg=source_beam_pattern_angles_deg,
        source_beam_pattern_db=source_beam_pattern_db,
        ati=ati,
        bty=bty,
    )
    c0 = c(r_s, z_s)
    delta_alpha = jnp.where(theta.shape[0] > 1, theta[1] - theta[0], 1.0)
    wavelength = c0 / freq
    min_width_m = min_width_wavelengths * wavelength
    if range_window_softness_m is None:
        range_window_softness_m = jnp.maximum(ds, (rr_grid[-1] - rr_grid[0]) / jnp.maximum(rr_grid.shape[0] - 1, 1))

    beam_solver = jax.vmap(
        lambda trj, launch_angle, source_amplitude: accumulate_geometric_gaussian_field_autodiff(
            freq,
            trj,
            launch_angle,
            source_amplitude,
            delta_alpha,
            rr_grid,
            rz_grid,
            min_width_m,
            range_window_softness_m,
            run_mode=run_mode,
        ),
        in_axes=(0, 0, 0),
    )

    field_per_beam = beam_solver(trj_set, theta, source_amplitudes) * launch_weights[:, None, None]
    field_total_raw = jnp.sum(field_per_beam, axis=0)
    field_total = scale_pressure_bellhop(freq, delta_alpha, c0, rr_grid, field_total_raw, run_mode=run_mode)
    energy = jnp.abs(field_total)
    tl_db = -20.0 * jnp.log10(energy + 1e-16)

    return {
        "theta": theta,
        "launch_weights": launch_weights,
        "source_amplitudes": source_amplitudes,
        "trajectories": trj_set,
        "field_per_beam": field_per_beam,
        "field_total_raw": field_total_raw,
        "field_total": field_total,
        "tl_db": tl_db,
    }


# @functools.partial(jax.jit, static_argnames=('RunTypeE',))
# def InfluenceGeoGaussian(
#     freq, single_trj, z_s, delta_alpha, rr_grid, rz_grid, RunTypeE='S'
# ):
#     # --- force all position‐and‐time columns to real floats at once ---
#     ray_r    = jnp.real(single_trj[:, 0])   # [100]   ranges
#     ray_z    = jnp.real(single_trj[:, 1])   # [100]   depths
#     ray_rho  = jnp.real(single_trj[:, 2])
#     ray_zeta = jnp.real(single_trj[:, 3])
#     ray_tau  = jnp.real(single_trj[:, 8])

#     # --- now rebuild the *complex* beam amplitude ---
#     q_r      = single_trj[:, 4]             # real part
#     q_i      = single_trj[:, 5]             # imag part
#     ray_q    = q_r + 1j * q_i                # shape [100], dtype=complex128

#     omega    = 2 * jnp.pi * freq
#     IBWin    = 4
#     DS       = jnp.sqrt(2.0) * jnp.sin(omega * z_s * ray_zeta[0])
#     C0       = c(ray_r[0], ray_z[0])
#     q0       = C0 / delta_alpha
#     lambda_  = C0 / freq
#     Rat1     = 1 / jnp.sqrt(2 * jnp.pi)

#     Nsteps   = ray_r.shape[0]
#     Nrr      = rr_grid.shape[0]
#     Nrz      = rz_grid.shape[0]

#     init_U   = jnp.zeros((Nrz, Nrr), dtype=jnp.complex128)
#     init_c   = 1.0 + 0j

#     def body_is(carry, is_):
#         caustic, U = carry

#         # caustic flip on zero‐crossing of ray_q
#         crossed = ((ray_q[is_] <= 0) & (ray_q[is_-1] > 0)) | \
#                   ((ray_q[is_] >= 0) & (ray_q[is_-1] < 0))
#         caustic = jax.lax.cond(is_>0 & crossed,
#                                lambda c: 1j*c,
#                                lambda c: c,
#                                caustic)

#         # bracket real ranges only
#         x_curr = ray_r[is_]
#         x_next = ray_r[is_+1]
#         ir1, ir2 = bracket_indices(x_curr, x_next, rr_grid)

#         def no_op(carry): return carry
#         def work(carry):
#             ca, Uloc = carry

#             # build real tangent and normal
#             dr = ray_r[is_+1] - ray_r[is_]
#             dz = ray_z[is_+1] - ray_z[is_]
#             tray = jnp.array([dr, dz])
#             rlen = jnp.linalg.norm(tray)
#             t_unit = tray / rlen
#             t_scaled = t_unit / rlen
#             n_ray = jnp.array([-t_unit[1], t_unit[0]])

#             def body_ir(Uc, ir):
#                 # all real operations until we get to amplitudes
#                 z_pts = rz_grid
#                 xrcvr = jnp.stack([jnp.full_like(z_pts, rr_grid[ir]), z_pts], axis=-1)
#                 xray  = jnp.stack([jnp.full_like(z_pts, ray_r[is_]),
#                                    jnp.full_like(z_pts, ray_z[is_])], axis=-1)
#                 delta = xrcvr - xray
#                 s     = delta @ t_scaled
#                 n     = jnp.abs(delta @ n_ray)

#                 # complex amplitude interpolation
#                 q_i   = ray_q[is_] + s * (ray_q[is_+1] - ray_q[is_])
#                 sigma = jnp.abs(q_i / q0)
#                 sigma_lim = jnp.minimum(0.2 * freq * ray_tau[is_] / lambda_,
#                                         jnp.pi * lambda_)
#                 sigma = jnp.where(sigma < sigma_lim, sigma_lim, sigma)

#                 mask = n < IBWin * sigma
#                 # irz is now guaranteed to be integer dtype
#                 irz  = jnp.nonzero(mask, size=Nrz)[0]

#                 def no_contrib(Uc): return Uc
#                 def contrib(Uc):
#                     A     = jnp.abs(q0 / q_i[irz])
#                     delay = ray_tau[is_] + s[irz] * (ray_tau[is_+1] - ray_tau[is_])
#                     ca    = jnp.where(
#                                 ((q_i[irz] <= 0) & (ray_q[is_] > 0)) |
#                                 ((q_i[irz] >= 0) & (ray_q[is_] < 0)),
#                                 1j * ca,
#                                 ca
#                              )
#                     const = Rat1 * ca * jnp.sqrt(C0 / jnp.abs(q_i[irz]))
#                     if RunTypeE == 'S':
#                         const = DS * const
#                     amp = const * jnp.exp(-0.5*(n[irz]/sigma[irz])**2)/(sigma[irz]*A)

#                     if RunTypeE == 'C':
#                         contri = amp * jnp.exp(-1j*omega*delay)
#                     else:
#                         W = jnp.exp(-0.5*(n[irz]/sigma[irz])**2)/(2*sigma[irz]*A)
#                         contri = (jnp.abs(amp)*jnp.exp(omega*jnp.imag(delay))/W)**2 * W

#                     # **now** both irz and ir are integer, so this is legal
#                     return Uc.at[irz, ir].add(contri)

#                 return jax.lax.cond(mask.any(), contrib, no_contrib, Uc)

#             Unew = jax.lax.fori_loop(ir1, ir2+1, body_ir, Uloc)
#             return (ca, Unew)

#         caustic, U = jax.lax.cond(ir2>=ir1, work, no_op, (caustic, U))
#         return (caustic, U), None

#     (_, final_U), _ = jax.lax.scan(body_is, (init_c, init_U), jnp.arange(Nsteps-1))
#     return final_U



#### Gemini Version ####
# @functools.partial(jax.jit, static_argnames=('RunTypeE',))
# def InfluenceGeoGaussian(freq, single_trj, z_s, delta_alpha, rr_grid, rz_grid, RunTypeE='S'):
#     """
#     Corrected version of the Gaussian beam influence function.
#     """
#     # --- Extract trajectory components ---
#     ray_r    = jnp.real(single_trj[:, 0])
#     ray_z    = jnp.real(single_trj[:, 1])
#     ray_zeta = jnp.real(single_trj[:, 3])
#     ray_tau  = jnp.real(single_trj[:, 8])
#     # --- Rebuild the complex beam parameter q ---
#     ray_q    = single_trj[:, 4] + 1j * single_trj[:, 5]
    
#     # CRITICAL FIX for TypeError: Use the REAL part of q for any comparisons.
#     # Comparing complex numbers (e.g., `ray_q <= 0`) is undefined and causes the error.
#     ray_q_real = jnp.real(ray_q)

#     # --- Constants and initializations ---
#     omega    = 2 * jnp.pi * freq
#     IBWin    = 4.0
#     C0       = c(ray_r[0], ray_z[0])
#     DS       = jnp.sqrt(2.0) * jnp.sin(omega * z_s * ray_zeta[0])
#     q0       = C0 / delta_alpha
#     lambda_  = C0 / freq
#     Rat1     = 1 / jnp.sqrt(2 * jnp.pi)

#     Nsteps   = ray_r.shape[0]
#     Nrr      = rr_grid.shape[0]
#     Nrz      = rz_grid.shape[0]

#     init_U   = jnp.zeros((Nrz, Nrr), dtype=jnp.complex128)
#     init_c   = 1.0 + 0j

#     # --- Main loop over ray segments using lax.scan ---
#     def body_is(carry, is_):
#         caustic, U = carry

#         # --- Caustic detection: check for zero-crossing of REAL(q) ---
#         crossed = ((ray_q_real[is_] <= 0) & (ray_q_real[is_-1] > 0)) | \
#                   ((ray_q_real[is_] >= 0) & (ray_q_real[is_-1] < 0))
        
#         caustic = jax.lax.cond(jnp.logical_and(is_ > 0, crossed),
#                                lambda c: 1j * c,
#                                lambda c: c,
#                                caustic)

#         # --- Bracket receiver ranges for the current ray segment ---
#         x_curr = ray_r[is_]
#         x_next = ray_r[is_+1]
#         ir1, ir2 = bracket_indices(x_curr, x_next, rr_grid)

#         def no_op(carry):
#             return carry

#         def work(carry):
#             ca, Uloc = carry

#             # --- Geometry for the ray segment ---
#             dr = ray_r[is_+1] - ray_r[is_]
#             dz = ray_z[is_+1] - ray_z[is_]
#             tray = jnp.array([dr, dz])
#             rlen = jnp.linalg.norm(tray) + 1e-12
#             t_unit = tray / rlen
#             t_scaled = t_unit / rlen
#             n_ray = jnp.array([-t_unit[1], t_unit[0]])

#             # --- Loop over bracketed range cells ---
#             def body_ir(U_inner, ir):
#                 # --- All operations here are vectorized over depth (z) ---
#                 z_pts = rz_grid
#                 xrcvr = jnp.stack([jnp.full_like(z_pts, rr_grid[ir]), z_pts], axis=-1)
#                 xray  = jnp.array([ray_r[is_], ray_z[is_]])
#                 delta = xrcvr - xray
#                 s = delta @ t_scaled
#                 n = jnp.abs(delta @ n_ray)

#                 # --- Interpolate beam parameters ---
#                 q_i   = ray_q[is_] + s * (ray_q[is_+1] - ray_q[is_])
#                 sigma = jnp.abs(q_i / q0)
#                 sigma_lim = jnp.minimum(0.2 * freq * ray_tau[is_] / lambda_, jnp.pi * lambda_)
#                 sigma = jnp.where(sigma < sigma_lim, sigma_lim, sigma)

#                 mask = n < IBWin * sigma

#                 def no_contrib(Uc):
#                     return Uc

#                 def contrib(Uc):
#                     # Use the boolean mask to select valid data, avoiding potential index errors
#                     irz = jnp.nonzero(mask, size=Nrz, fill_value=-1)[0]
#                     valid_irz = irz[irz != -1]

#                     s_valid = s[valid_irz]
#                     n_valid = n[valid_irz]
#                     sigma_valid = sigma[valid_irz]
#                     q_i_valid = q_i[valid_irz]
                    
#                     # --- Sub-beam caustic check using REAL parts ---
#                     # CRITICAL FIX for TypeError: Convert to real before comparison.
#                     q_i_real = jnp.real(q_i_valid)
#                     ray_q_is_real = jnp.real(ray_q[is_])

#                     flip_sub = ((q_i_real <= 0) & (ray_q_is_real > 0)) | \
#                                ((q_i_real >= 0) & (ray_q_is_real < 0))
                    
#                     caustic_interp = jnp.where(flip_sub, 1j * ca, ca)

#                     # --- Amplitude and Phase Calculation ---
#                     delay = ray_tau[is_] + s_valid * (ray_tau[is_+1] - ray_tau[is_])
#                     const = Rat1 * caustic_interp * jnp.sqrt(C0 / jnp.abs(q_i_valid))

#                     if RunTypeE == 'S':
#                         const = DS * const
                    
#                     A = jnp.abs(q0 / q_i_valid)
#                     amp = const * jnp.exp(-0.5*(n_valid/sigma_valid)**2)/(sigma_valid*A)

#                     if RunTypeE == 'C':  # Coherent TL
#                         contri = amp * jnp.exp(-1j * omega * delay)
#                     else:  # Incoherent / semi-coherent
#                         W = jnp.exp(-0.5*(n_valid/sigma_valid)**2)/(2*sigma_valid*A)
#                         contri = (jnp.abs(amp)*jnp.exp(omega*jnp.imag(delay))/W)**2 * W
                    
#                     return Uc.at[valid_irz, ir].add(contri)
                
#                 # Refined check for whether to run the contribution logic
#                 return jax.lax.cond(mask.any(), contrib, no_contrib, U_inner)

#             Unew = jax.lax.fori_loop(ir1, ir2 + 1, body_ir, Uloc)
#             return (ca, Unew)

#         caustic, U = jax.lax.cond(ir2 >= ir1, work, no_op, (caustic, U))
#         return (caustic, U), None

#     # --- Run the scan over all ray segments ---
#     (_, final_U), _ = jax.lax.scan(body_is, (init_c, init_U), jnp.arange(Nsteps - 1))
#     return final_U
