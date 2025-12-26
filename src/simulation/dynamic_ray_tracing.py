import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import sys
import os
import functools

sys.path.append('.')
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import functions from other modules.
from sound_speed import c, c_r, c_z, c_rr, c_zz, c_rz
from boundary import (
    bathymetry as bty, altimetry as ati, normal_vector_to_altimetry, normal_vector_to_bathymetry as n_bty,
    normal_vector_to_altimetry as n_ati, tangent_vector_to_altimetry as t_ati, tangent_vector_to_bathymetry as t_bty
)



@jax.jit
def dynamic_ray_eqns(Y):
    """
    Defines the ODE system for 'dynamic' ray tracing, which includes
    (q, p) for beam spreading / amplitude calculation.
    
    State: Y = [r, z, rho, zeta, q_r, q_i, p_r, p_i, tau]
    """
    r, z, rho, zeta, q_r, q_i, p_r, p_i, tau = Y

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
    def __init__(self, function, dt, ati=ati, bty=bty, t_ati=t_ati, n_ati=n_ati, t_bty=t_bty, n_bty=n_bty):
        self.function = function  # Expecting extended_ray_eqns.
        self.dt = dt
        self.ati = ati
        self.bty = bty
        self.n_bty = n_bty
        self.t_bty = t_bty
        self.t_ati = t_ati
        self.n_ati = n_ati

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

        return Y_new


    def __call__(self, Y):
        X0 = jnp.array([Y[0], Y[1]])
        Y_pass = jnp.copy(Y)
        K = Y + (self.dt / 2.0) * self.function(Y)
        Y_new = Y + self.dt * self.function(K)
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
        theta = jnp.arctan(Y_next[3] / Y_next[2])
        theta_logic = jnp.logical_or(jnp.abs(theta) >= jnp.pi / 2.0, jnp.isnan(Y_next).any())
        r = Y_next[0]
        z = Y_next[1]
        r_logic = jnp.logical_or(r <= 0.0, r >= 100000.0)
        z_logic = jnp.logical_or(z <= 0.0, z >= 5000.0)
        should_break = jnp.logical_or(r_logic, jnp.logical_or(z_logic, theta_logic))
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

def compute_multiple_ray_paths(freq, r_s, z_s, theta, ray_eqns=dynamic_ray_eqns, ds=10.0, R_max=100000, Z_max=5000, ati=ati, bty=bty):
    """
    Computes trajectories for an array of initial angles using vectorized integration.
    """
    omega = 2 * jnp.pi * freq
    delta_alpha = theta[1] - theta[0]
    n_theta = len(theta)
    Y0_set = jnp.zeros((n_theta, 9), dtype=jnp.float64)
    C0 = c(r_s, z_s)
    rho = jnp.cos(theta) / C0
    zeta = jnp.sin(theta) / C0
    epsilon = (2*C0**2)/(omega * delta_alpha**2)
    q0_r = 0.0
    q0_i = epsilon
    p0_r = 1.0
    p0_i = 0.0
    tau_0 = 0.0
    
    Y0_set = Y0_set.at[:, 0].set(r_s) # r
    Y0_set = Y0_set.at[:, 1].set(z_s) # z
    Y0_set = Y0_set.at[:, 2].set(rho) # rho
    Y0_set = Y0_set.at[:, 3].set(zeta) # zeta
    Y0_set = Y0_set.at[:, 4].set(q0_r) # q_r
    Y0_set = Y0_set.at[:, 5].set(q0_i) # q_i
    Y0_set = Y0_set.at[:, 6].set(p0_r) # p_r
    Y0_set = Y0_set.at[:, 7].set(p0_i) # p_i
    Y0_set = Y0_set.at[:, 8].set(tau_0) # tau


    ds = 1.0
    stepper = RK2_stepper(ray_eqns, ds, ati=ati, bty=bty)
    trj = jax.vmap(rollout(stepper, int(R_max / ds), include_init=True))(Y0_set)
    return trj

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
    return hist[-1]


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

@functools.partial(jax.jit, static_argnames=('RunTypeE',))
def InfluenceGeoGaussian(freq, single_trj, z_s, delta_alpha, rr_grid, rz_grid, RunTypeE='S'):
    
    omega = 2 * jnp.pi * freq
    IBWin = 4
    DS  = jnp.sqrt(2.0)* jnp.sin( omega * z_s * single_trj[0, 3])
    C =  c(single_trj[0, 0], single_trj[0, 1])
    q0  = C/ delta_alpha
    lambda_ = C / freq
    Rat1 = 1 / jnp.sqrt(2 * jnp.pi)
    Nsteps = single_trj.shape[0]
    Nrr = rr_grid.shape[0]
    Nrz = rz_grid.shape[0]
    
    # ray_q = single_trj[:,4]
    q_r = single_trj[:, 4]
    q_i = single_trj[:, 5]
    ray_q = q_r + 1j * q_i                 # complex q along ray
    ray_q_re = jnp.real(ray_q)             # for sign checks only

    ray_x = jnp.real(single_trj[:, :2])
    ray_tau = jnp.real(single_trj[:, 8])
    ray_Rfa = jnp.ones(Nsteps, dtype=jnp.float64)  # Placeholder for ray_Rfa, if needed.
    
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
        
        crossed = ((ray_q_re[is_] <= 0) & (ray_q_re[is_-1] > 0)) | \
                  ((ray_q_re[is_] >= 0) & (ray_q_re[is_-1] < 0))
        caustic = jax.lax.cond(is_>0 & crossed, flip, lambda c: c, caustic)

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
                q_seg = ray_q[is_] + s * (ray_q[is_+1] - ray_q[is_])   # complex
                sigma = jnp.abs(q_seg) / (q0 + 1e-12)                 # q0 from line 331 (scalar)
                sigma_lim = jnp.minimum(0.2 * freq * ray_tau[is_] / lambda_, jnp.pi * lambda_)
                sigma = jnp.where(sigma < sigma_lim, sigma_lim, sigma)

                # select depth‑indices where beam contributes
                mask = n < IBWin * sigma
                irz = jnp.nonzero(mask, size=Nrz)[0]  # padded with zeros if empty

                def no_contrib(Uc):
                    return Uc

                def contrib(Uc):
                    # A = jnp.abs(q0 / q_i[irz])
                    A = jnp.abs(q0 / (q_seg[irz] + 1e-12))
                    delay = ray_tau[is_] + s[irz] * (ray_tau[is_+1] - ray_tau[is_])

                    # caustic phase per sub‐beam
                    caust = caustic_loc * jnp.ones_like(irz, dtype=jnp.complex64)
                    # flip_sub = ((q_i[irz] <= 0) & (ray_q[is_] > 0)) | \
                    #            ((q_i[irz] >= 0) & (ray_q[is_] < 0))
                    
                    q_seg_re = jnp.real(q_seg)
                    flip_sub = ((q_seg_re[irz] <= 0) & (ray_q_re[is_] > 0)) | \
                            ((q_seg_re[irz] >= 0) & (ray_q_re[is_] < 0))
                            
                    caust = jnp.where(flip_sub, 1j * caust, caust)

                    # const = (Rat1 * ray_Rfa[is_] *
                            #  jnp.sqrt(c(ray_x[is_, 0], ray_x[is_, 1]) / jnp.abs(q_i[irz])) *
                            #  caust)
                            
                    const = (Rat1 * ray_Rfa[is_] *jnp.sqrt(c(ray_x[is_, 0], ray_x[is_, 1]) / (jnp.abs(q_seg[irz]) + 1e-12)) *caust)

                    if RunTypeE == 'S':
                        const = DS * const

                    amp = const * jnp.exp(-0.5 * (n[irz] / sigma[irz])**2) / (sigma[irz] * A)

                    if RunTypeE == 'C':  # coherent TL
                        contri = amp * jnp.exp(-1j * 2*jnp.pi*freq * delay)
                        return Uc.at[irz, ir].add(contri)
                    else:               # incoherent / semi‐coherent
                        W = jnp.exp(-0.5 * (n[irz]/sigma[irz])**2) / (2.*sigma[irz]*A)
                        contri = (jnp.abs(amp) * jnp.exp(2*jnp.pi*freq * jnp.imag(delay)) / W)**2 * W
                        return Uc.at[irz, ir].add(contri)

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

