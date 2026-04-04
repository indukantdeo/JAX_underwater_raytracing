# sound_speed.py
import jax
import jax.numpy as jnp

DEFAULT_PROFILE_Z = jnp.array([
    0.0, 38.0, 50.0, 70.0, 100.0, 140.0, 160.0, 170.0, 200.0,
    215.0, 250.0, 300.0, 370.0, 450.0, 500.0, 700.0, 900.0,
    1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0,
], dtype=jnp.float64)

DEFAULT_PROFILE_C = jnp.array([
    1476.7, 1476.7, 1472.6, 1468.8, 1467.2, 1471.6, 1473.6,
    1473.6, 1472.7, 1472.2, 1471.6, 1471.6, 1472.0, 1472.7,
    1473.1, 1474.9, 1477.0, 1478.1, 1480.7, 1483.8, 1490.5,
    1498.3, 1506.5,
], dtype=jnp.float64)


def make_2d_ssp_operators(c_fn):
    """
    Build a consistent set of first- and second-derivative operators
    for a 2D sound-speed field c(r, z).
    """
    c_eval = jax.jit(c_fn)
    c_r_eval = jax.jit(jax.grad(c_eval, argnums=0))
    c_z_eval = jax.jit(jax.grad(c_eval, argnums=1))
    c_rr_eval = jax.jit(jax.grad(c_r_eval, argnums=0))
    c_rz_eval = jax.jit(jax.grad(c_z_eval, argnums=0))
    c_zz_eval = jax.jit(jax.grad(c_z_eval, argnums=1))
    return {
        "c": c_eval,
        "c_r": c_r_eval,
        "c_z": c_z_eval,
        "c_rr": c_rr_eval,
        "c_rz": c_rz_eval,
        "c_zz": c_zz_eval,
        "interface_depths_m": jnp.asarray([], dtype=jnp.float64),
    }


@jax.jit
def default_profile(z):
    return jnp.interp(z, DEFAULT_PROFILE_Z, DEFAULT_PROFILE_C)


@jax.jit
def c(r, z):
    """
    Returns the interpolated sound speed at (r, z). In this implementation,
    the sound speed is defined solely as a function of depth z.
    """
    return default_profile(z)

# First derivatives
c_r = jax.jit(jax.grad(c, argnums=0))  # Gradient with respect to r (typically zero)
c_z = jax.jit(jax.grad(c, argnums=1))  # Gradient with respect to z

# Second derivatives
c_rr = jax.jit(jax.grad(c_r, argnums=0))  # Second derivative with respect to r
c_rz = jax.jit(jax.grad(c_z, argnums=0))  # Mixed derivative (first with respect to r then z)
c_zz = jax.jit(jax.grad(c_z, argnums=1))  # Second derivative with respect to z

@jax.jit
def grad_r_c(c_field, r):
    grad_r = jnp.zeros_like(c_field)
    epsilon = 1e-6
    grad_r = grad_r.at[:, :-1].set((c_field[:, 1:] - c_field[:, :-1]) / (r[1:] - r[:-1] + epsilon))
    grad_r = grad_r.at[:, -1].set(grad_r[:, -2])
    return grad_r

@jax.jit
def grad_z_c(c_field, z):
    grad_z = jnp.zeros_like(c_field)
    epsilon = 1e-6
    grad_z = grad_z.at[:-1, :].set(((c_field[1:, :] - c_field[:-1, :]).T / (z[1:] - z[:-1] + epsilon)).T)
    grad_z = grad_z.at[-1, :].set(grad_z[-2, :])
    return grad_z

@jax.jit
def munk_profile(z):
    """
    Returns the Munk sound speed profile at depth z.
    """
    c0 = 1500.0
    eps = 0.00737
    z_axis = 1300.0
    z_scaled = 2 * (z - z_axis) / z_axis
    return c0 * (1 + eps * (z_scaled - 1 + jnp.exp(-z_scaled)))

@jax.jit
def analytical_grad_munk_profile(z):
    c0 = 1500.0
    epsilon = 0.00737
    z_axis = 1300.0
    z_scaled = 2 * (z - z_axis) / z_axis
    A = 2.0 / z_axis
    return c0 * epsilon * (1 - jnp.exp(-z_scaled)) * A

@jax.jit
def analytical_2nd_grad_munk_profile(z):
    c0 = 1500.0
    epsilon = 0.00737
    z_axis = 1300.0
    z_scaled = 2 * (z - z_axis) / z_axis
    A = 2.0 / z_axis
    return c0 * epsilon * jnp.exp(-z_scaled) * A**2


@jax.jit
def c_munk(r, z):
    return munk_profile(z)


MUNK_OPERATORS = make_2d_ssp_operators(c_munk)
DEFAULT_OPERATORS = {
    "c": c,
    "c_r": c_r,
    "c_z": c_z,
    "c_rr": c_rr,
    "c_rz": c_rz,
    "c_zz": c_zz,
    "interface_depths_m": DEFAULT_PROFILE_Z,
}
