import matplotlib.pyplot as plt
import sys
import jax.numpy as jnp
import jax
import os
sys.path.append('../src')
sys.path.append('.')
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# from src.plot import plot
# from src.plot import *
from plot import *
from simulation.dynamic_ray_tracing import dynamic_ray_eqns, InfluenceGeoGaussian
from simulation.dynamic_ray_tracing import compute_multiple_ray_paths
from simulation.boundary import bathymetry as bty
from simulation.boundary import altimetry as ati
from simulation.sound_speed import c

# Example usage:

# Source parameters.
freq = 230.0  # Frequency in Hz
alpha2 = 45.0 * jnp.pi / 180.0  # Beam width in radians
alpha1 = -45.0 * jnp.pi / 180.0  # Beam
r_s = 0.0
z_s = 18.0
rr_grid = jnp.linspace(0, 100000, 1001)  # Range grid in meters
rz_grid = jnp.linspace(0, 3000, 601)  # Depth grid in meters
C = c(r_s, z_s)  # Speed of sound at source depth
delta_alpha_opt = jnp.sqrt(C/(6*freq*rr_grid[-1]))  # Optimal beam width
Nbeam_opt = jnp.round(2 + (alpha2 - alpha1)/delta_alpha_opt)  # Optimal number of beams jnp.round
print("Optimal number of beams:", Nbeam_opt)
theta =  jnp.arange(alpha1, alpha2 + delta_alpha_opt, delta_alpha_opt, dtype=jnp.float64)  # Beam angles
trj_set = compute_multiple_ray_paths(freq, r_s, z_s, theta, ds=1.0, ati=ati, bty=bty)
x_lim = (0, 100000.0)
y_lim = (0, 3000.0)
fig, ax = plot_environment(100000, r_s, z_s, x_lim=x_lim, y_lim=y_lim, plot_src=True,
                        plot_bty=True, fill_bty=True, plot_ati=True, fill_ati=True,
                        grid=False, legend=False)
fig, ax = plot_ray_paths(trj_set, fig=fig, ax=ax, grid=False)

# Convert jax array to numpy array.
trj_set_np = trj_set.copy().astype('float64')


# batched_influence = jax.vmap(
#     lambda trj: InfluenceGeoGaussian(
#         freq, trj, z_s, delta_alpha_opt, rr_grid, rz_grid, RunTypeE='S'
#     ),
#     in_axes=0  # only the first arg (trj) varies; everything else is fixed
# )
# U_beams = batched_influence(trj_set)   # (Nbeams, Nz, Nr)
# U_total = jnp.sum(U_beams, axis=0)


# # Convert to Transmission Loss (TL) in dB
# # Add a small epsilon to avoid log(0)
# epsilon = 1e-20
# TL = -20 * jnp.log10(jnp.abs(U_total) + epsilon)

# print("TL shape:", TL.shape)
# print("TL dtype:", TL.dtype)

# print("TL min:", jnp.min(TL), "TL max:", jnp.max(TL))

# # --- Plotting ---
# print("Plotting results...")
# fig, ax = plt.subplots(figsize=(12, 5))
# im = ax.imshow(
#     TL,
#     extent=[rr_grid[0]/1000, rr_grid[-1]/1000, rz_grid[-1], rz_grid[0]],
#     aspect='auto',
#     cmap='jet_r',
#     vmin=40,  # Adjust color limits for better visualization
#     vmax=100
# )
# ax.set_title(f'Transmission Loss (TL) at {freq} Hz')
# ax.set_xlabel('Range (km)')
# ax.set_ylabel('Depth (m)')
# plt.show()
# print("Plotting complete.")
