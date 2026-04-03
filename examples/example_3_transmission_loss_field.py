import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append('../src')
sys.path.append('.')
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from plot import plot_environment
from simulation.boundary import altimetry as ati
from simulation.boundary import bathymetry as bty
from simulation.dynamic_ray_tracing import solve_transmission_loss


freq = 230.0
r_s = 0.0
z_s = 18.0
theta_min = -45.0 * jnp.pi / 180.0
theta_max = 45.0 * jnp.pi / 180.0
n_beams = 181

rr_grid = jnp.linspace(0.0, 100000.0, 1001)
rz_grid = jnp.linspace(0.0, 3000.0, 601)

result = solve_transmission_loss(
    freq,
    r_s,
    z_s,
    theta_min,
    theta_max,
    n_beams,
    rr_grid,
    rz_grid,
    ds=5.0,
    beam_type='geometric',
    coherent=True,
    min_width_wavelengths=0.5,
    ati=ati,
    bty=bty,
)

tl_db = result['tl_db']

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(
    tl_db,
    extent=[rr_grid[0] / 1000.0, rr_grid[-1] / 1000.0, rz_grid[-1], rz_grid[0]],
    aspect='auto',
    cmap='jet_r',
)
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')
ax.set_title(f'Transmission Loss at {freq:.1f} Hz')
plt.colorbar(im, ax=ax, label='TL (dB)')

fig_env, ax_env = plot_environment(
    rr_grid[-1],
    r_s,
    z_s,
    x_lim=(rr_grid[0], rr_grid[-1]),
    y_lim=(rz_grid[-1], rz_grid[0]),
    plot_src=True,
    plot_bty=True,
    fill_bty=False,
    plot_ati=True,
    fill_ati=False,
)

plt.show()
