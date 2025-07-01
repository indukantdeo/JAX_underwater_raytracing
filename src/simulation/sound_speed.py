# sound_speed.py
import jax
import jax.numpy as jnp

@jax.jit
def c(r, z):
    """
    Returns the interpolated sound speed at (r, z). In this implementation,
    the sound speed is defined solely as a function of depth z.
    """
    c0_vals = jnp.array([1476.7, 1476.7, 1472.6, 1468.8, 1467.2, 1471.6, 1473.6,
                          1473.6, 1472.7, 1472.2, 1471.6, 1471.6, 1472.0, 1472.7,
                          1473.1, 1474.9, 1477.0, 1478.1, 1480.7, 1483.8, 1490.5,
                          1498.3, 1506.5], dtype=jnp.float64)
    z0 = jnp.array([0.0, 38.0, 50.0, 70.0, 100.0, 140.0, 160.0, 170.0, 200.0,
                    215.0, 250.0, 300.0, 370.0, 450.0, 500.0, 700.0, 900.0,
                    1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0], dtype=jnp.float64)
    return jnp.interp(z, z0, c0_vals)

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
