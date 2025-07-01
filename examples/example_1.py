import matplotlib.pyplot as plt
import sys
import jax.numpy as jnp
import jax

sys.path.append('../src')
# from src.plot import plot
# from src.plot import *
from plot import *
from simulation.ray_tracing import ray_eqns
from simulation.ray_tracing import compute_multiple_ray_paths
from simulation.boundary import bathymetry as bty
from simulation.boundary import altimetry as ati

# Example usage:

# Source parameters.
r_s = 0.0
z_s = 18.0
theta = (jnp.pi / 180.0) * jnp.linspace(-20, 20, 20)
trj_set = compute_multiple_ray_paths(r_s, z_s, theta, ds=1.0, ati=ati, bty=bty)
x_lim = (0, 100000.0)
y_lim = (0, 3000.0)
fig, ax = plot_environment(100000, r_s, z_s, x_lim=x_lim, y_lim=y_lim, plot_src=True,
                        plot_bty=True, fill_bty=True, plot_ati=True, fill_ati=True,
                        grid=False, legend=False)
fig, ax = plot_ray_paths(trj_set, fig=fig, ax=ax, grid=False)

# Convert jax array to numpy array.
trj_set_np = trj_set.copy().astype('float64')
print(trj_set_np[:,-1,7])
print(trj_set_np[:,-1,8])
# Build the summary dictionary.
ray_summary = {
    "title": "BELLHOP- Dickins seamount",
    "freq": 230.0,
    "Nsxyz": [1, 1, 1],
    "NbeamAngles": [trj_set_np.shape[0], 1],
    "DepthT": 0.0,
    "DepthB": 3000.0,
    "type": 'rz',
    "trj_set_np": trj_set_np,
}

# # Write the .ray file.
# write_ray_file("myray.ray", ray_summary)