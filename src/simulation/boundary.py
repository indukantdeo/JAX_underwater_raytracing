# boundary.py
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


@jax.jit
def flat_bathymetry(r):
    return 5000.0 + 0.0 * r


@jax.jit
def bathymetry(r):
    """
    Returns the bottom depth (bathymetry) at range r.
    """
    b = 3000.0
    ro = jnp.array([0.0, 10000.0, 20000.0, 30000.0, 100000.0], dtype=jnp.float64)
    z0 = jnp.array([b, b, 500.0, b, b], dtype=jnp.float64)
    z = jnp.interp(r, ro, z0)
    return z

@jax.jit
def tangent_vector_to_bathymetry(r):
    """
    Returns the unit tangent vector to the bathymetry at range r.
    """
    dz_dr = jax.grad(bathymetry)(r)
    # tangent_vector = jnp.array([-1, dz_dr], dtype=jnp.float64)
    tangent_vector = jnp.array([1.0, dz_dr], dtype=jnp.float64)
    tangent_vector /= jnp.linalg.norm(tangent_vector)
    return tangent_vector

@jax.jit
def normal_vector_to_bathymetry(r):
    """
    Returns the unit normal vector to the bathymetry at range r.
    """
    tangent_vector = tangent_vector_to_bathymetry(r)
    normal_vector = jnp.array([tangent_vector[1], -tangent_vector[0]], dtype=jnp.float64)
    normal_vector /= jnp.linalg.norm(normal_vector)
    
    # Enforce bottom normal points into the water: upward => n_z < 0 (since z is down)
    # normal_vector = jnp.where(normal_vector[1] < 0.0, normal_vector, -normal_vector)
    
    return normal_vector
    # t = tangent_vector_to_bathymetry(r)
    # n = jnp.array([-t[1], t[0]])
    # n = n / jnp.linalg.norm(n)

    # # Enforce bottom normal points into the water.
    # # Depth z is positive downward, so "into water" is upward => n_z < 0.
    # n = jnp.where(n[1] < 0.0, n, -n)
    # return n
    
    

@jax.jit
def altimetry(r):
    """
    Returns the surface elevation (altimetry) at range r.
    """
    a0 = 0.0
    return a0


def make_boundary_operators(bathymetry_fn, altimetry_fn):
    bathymetry_eval = jax.jit(bathymetry_fn)
    altimetry_eval = jax.jit(altimetry_fn)

    @jax.jit
    def tangent_vector_to_bathymetry_eval(r):
        dz_dr = jax.grad(bathymetry_eval)(r)
        tangent_vector = jnp.array([1.0, dz_dr], dtype=jnp.float64)
        tangent_vector /= jnp.linalg.norm(tangent_vector)
        return tangent_vector

    @jax.jit
    def normal_vector_to_bathymetry_eval(r):
        tangent_vector = tangent_vector_to_bathymetry_eval(r)
        normal_vector = jnp.array([tangent_vector[1], -tangent_vector[0]], dtype=jnp.float64)
        normal_vector /= jnp.linalg.norm(normal_vector)
        return normal_vector

    @jax.jit
    def tangent_vector_to_altimetry_eval(r):
        dz_dr = jax.grad(altimetry_eval)(r)
        tangent_vector = jnp.array([1.0, dz_dr], dtype=jnp.float64)
        tangent_vector /= jnp.linalg.norm(tangent_vector)
        return tangent_vector

    @jax.jit
    def normal_vector_to_altimetry_eval(r):
        tangent_vector = tangent_vector_to_altimetry_eval(r)
        normal_vector = jnp.array([-tangent_vector[1], tangent_vector[0]], dtype=jnp.float64)
        normal_vector /= jnp.linalg.norm(normal_vector)
        return normal_vector

    return {
        "bathymetry": bathymetry_eval,
        "altimetry": altimetry_eval,
        "tangent_vector_to_bathymetry": tangent_vector_to_bathymetry_eval,
        "normal_vector_to_bathymetry": normal_vector_to_bathymetry_eval,
        "tangent_vector_to_altimetry": tangent_vector_to_altimetry_eval,
        "normal_vector_to_altimetry": normal_vector_to_altimetry_eval,
    }

@jax.jit
def tangent_vector_to_altimetry(r):
    """
    Returns the unit tangent vector to the altimetry at range r.
    """
    dz_dr = jax.grad(altimetry)(r)
    tangent_vector = jnp.array([1, dz_dr], dtype=jnp.float64)
    tangent_vector /= jnp.linalg.norm(tangent_vector)
    return tangent_vector

@jax.jit
def normal_vector_to_altimetry(r):
    """
    Returns the unit normal vector to the altimetry at range r.
    """
    tangent_vector = tangent_vector_to_altimetry(r)
    normal_vector = jnp.array([-tangent_vector[1], tangent_vector[0]], dtype=jnp.float64)
    normal_vector /= jnp.linalg.norm(normal_vector)
    return normal_vector


DEFAULT_BOUNDARY_OPERATORS = make_boundary_operators(bathymetry, altimetry)
FLAT_BOUNDARY_OPERATORS = make_boundary_operators(flat_bathymetry, altimetry)
