# ray_tracing.py
import jax
jax.config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp

import sys
sys.path.append('.')
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import functions from other modules.
from sound_speed import c, c_r, c_z
from boundary import bathymetry as bty, altimetry as ati, normal_vector_to_bathymetry as n_bty, normal_vector_to_altimetry as n_ati
from plot import plot_environment, plot_ray_paths

@jax.jit
def ray_eqns(Y):
    """
    Defines the ODE system for ray propagation.
    Y is assumed to be [r, z, rho, zeta].
    """
    r = Y[0]
    z = Y[1]
    rho = Y[2]
    zeta = Y[3]
    
    
    C = c(r, z)
    Cr = c_r(r, z)
    Cz = c_z(r, z)
    C2 = C**2
    
    
    
    Y_dot = jnp.zeros(4, dtype=Y.dtype)
    Y_dot = Y_dot.at[0].set(C * rho)
    Y_dot = Y_dot.at[1].set(C * zeta)
    Y_dot = Y_dot.at[2].set(-Cr / C2)
    Y_dot = Y_dot.at[3].set(-Cz / C2)

    return Y_dot



# Modified RK2 stepper with reflection counting.
class RK2_stepper:
    def __init__(self, function, dt, ati=ati, bty=bty, n_ati=n_ati, n_bty=n_bty):
        self.function = function  # Expecting extended_ray_eqns.
        self.dt = dt
        self.ati = ati
        self.bty = bty
        self.n_bty = n_bty
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
        
        # eps = 1e-6
        # Y_new = Y_new.at[1].set(jnp.minimum(Y_new[1], self.bty(Y_new[0]) - eps))
        
        alpha = jnp.arctan(jax.grad(self.bty)(Y_new[0]))
        theta = jnp.arctan(Y_new[3] / Y_new[2])
        theta_new = -theta + 2 * alpha
        C_new = c(Y_new[0], Y_new[1])
        Y_new = Y_new.at[2].set(jnp.cos(theta_new) / C_new)
        Y_new = Y_new.at[3].set(jnp.sin(theta_new) / C_new)
        
        # # --- Ray-based (vector) specular reflection about boundary normal ---
        # C_new = c(Y_new[0], Y_new[1])

        # # Incident unit tangent vector (cosθ, sinθ) = c * (rho, zeta)
        # t_inc = C_new * jnp.array([Y_new[2], Y_new[3]])

        # # Boundary unit normal at reflection point
        # n = self.n_bty(Y_new[0])

        # # Reflect: t_ref = t_inc - 2*(t_inc·n)*n
        # t_ref = t_inc - 2.0 * jnp.dot(t_inc, n) * n
        
        # # Safety: ensure reflected ray points into the water
        # t_ref = jnp.where(jnp.dot(t_ref, n) > 0.0, t_ref, -t_ref)
        
        # # Convert back to slowness components
        # Y_new = Y_new.at[2].set(t_ref[0] / C_new)
        # Y_new = Y_new.at[3].set(t_ref[1] / C_new)

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
        
        alpha = jnp.arctan(jax.grad(self.ati)(Y_new[0]))
        theta = jnp.arctan(Y_new[3] / Y_new[2])
        theta_new = -theta - 2 * alpha
        C_new = c(Y_new[0], Y_new[1])
        Y_new = Y_new.at[2].set(jnp.cos(theta_new) / C_new)
        Y_new = Y_new.at[3].set(jnp.sin(theta_new) / C_new)
        
        # # --- Ray-based (vector) specular reflection about boundary normal ---
        # C_new = c(Y_new[0], Y_new[1])

        # t_inc = C_new * jnp.array([Y_new[2], Y_new[3]])
        # n = self.n_ati(Y_new[0])

        # t_ref = t_inc - 2.0 * jnp.dot(t_inc, n) * n

        # Y_new = Y_new.at[2].set(t_ref[0] / C_new)
        # Y_new = Y_new.at[3].set(t_ref[1] / C_new)
        
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

def compute_single_ray_path(r_s, z_s, theta_0, ray_eqns=ray_eqns, ds=10.0, R_max=100000, Z_max=5000):
    """
    Computes a single ray trajectory using the extended state.
    The state is [r, z, rho, zeta, n_top, n_bottom].
    """
    C0 = c(r_s, z_s)
    rho = jnp.cos(theta_0) / C0
    zeta = jnp.sin(theta_0) / C0
    Y0 = jnp.array([r_s, z_s, rho, zeta])
    num_steps = int(R_max / ds)
    stepper = jax.jit(RK2_stepper(ray_eqns, ds))
    trj = rollout(stepper, num_steps, include_init=True)(Y0)
    return trj

def compute_multiple_ray_paths(r_s, z_s, theta, ray_eqns=ray_eqns, ds=10.0, R_max=100000, Z_max=5000, ati=ati, bty=bty):
    """
    Computes trajectories for an array of initial angles using vectorized integration.
    """
    n_theta = len(theta)
    Y0_set = jnp.zeros((n_theta, 4), dtype=jnp.float64)
    C0 = c(r_s, z_s)
    rho = jnp.cos(theta) / C0
    zeta = jnp.sin(theta) / C0
    Y0_set = Y0_set.at[:, 0].set(r_s) # r
    Y0_set = Y0_set.at[:, 1].set(z_s) # z
    Y0_set = Y0_set.at[:, 2].set(rho) # rho
    Y0_set = Y0_set.at[:, 3].set(zeta) # zeta

    ds = 1.0
    stepper = RK2_stepper(ray_eqns, ds, ati=ati, bty=bty)
    trj = jax.vmap(rollout(stepper, int(R_max / ds), include_init=True))(Y0_set)
    return trj


def write_ray_file(filename, ray_summary):
    """
    Write a ray summary to a .ray file in a Bellhop-like format.
    Optimized using buffered I/O and vectorized string formatting.
    """
    from io import StringIO
    import numpy as np
    
    out = StringIO()
    out.write(f"'{ray_summary['title']}'\n")
    out.write(f"{ray_summary['freq']:.6f}\n")
    out.write(f"{ray_summary['Nsxyz'][0]} {ray_summary['Nsxyz'][1]} {ray_summary['Nsxyz'][2]}\n")
    out.write(f"{int(ray_summary['NbeamAngles'][0])} {int(ray_summary['NbeamAngles'][1])}\n")
    out.write(f"{ray_summary['DepthT']}\n")
    out.write(f"{ray_summary['DepthB']}\n")
    out.write(f"'{str(ray_summary['type'])}'\n")
    
    trj_set_np = np.asarray(ray_summary['trj_set_np'])
    num_rays = trj_set_np.shape[0]
    
    for i in (range(num_rays)):
        ray_trj = np.asarray(trj_set_np[i])
        # Compute the initial ray angle in degrees using numpy.
        angle_rad = np.arctan(ray_trj[0, 3] / ray_trj[0, 2])
        angle_deg = angle_rad * (180.0 / np.pi)
        out.write(f"{angle_deg:.6f}\n")
        out.write(f"{int(ray_trj.shape[0])} {int(ray_trj[-1, 7])} {int(ray_trj[-1, 8])}\n")
        # Convert the (r,z) points to a list of tuples so that the format string applies correctly.
        formatted_points = ["%.6f\t%.6f" % (pt[0], pt[1]) for pt in ray_trj[:, :2]]
        out.write("\n".join(formatted_points))
        out.write("\n")

    # Write the entire output to file in one go.
    with open(filename, "w") as f:
        f.write(out.getvalue())

