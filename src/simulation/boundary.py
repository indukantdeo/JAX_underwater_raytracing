# boundary.py
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
    tangent_vector = jnp.array([-1, dz_dr], dtype=jnp.float64)
    tangent_vector /= jnp.linalg.norm(tangent_vector)
    return tangent_vector

@jax.jit
def normal_vector_to_bathymetry(r):
    """
    Returns the unit normal vector to the bathymetry at range r.
    """
    tangent_vector = tangent_vector_to_bathymetry(r)
    normal_vector = jnp.array([-tangent_vector[1], tangent_vector[0]], dtype=jnp.float64)
    normal_vector /= jnp.linalg.norm(normal_vector)
    
    
    
    return normal_vector

@jax.jit
def altimetry(r):
    """
    Returns the surface elevation (altimetry) at range r.
    """
    a0 = 0.0
    return a0

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
