# plot.py
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'simulation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'simulation')))
from boundary import bathymetry as bty, altimetry as ati
from sound_speed import c
earthbrown = (0.5, 0.3, 0.1)
def plot_environment(R_max, r_src, z_src, fig=None, ax=None, plot_src=True,
                     x_lim=None, y_lim=None, bty=bty, ati=ati,
                     plot_bty=True, fill_bty=True, plot_ati=True,
                     fill_ati=True, grid=False, legend=False):

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    R = jnp.linspace(0, R_max, 1000)
    Z_bty = bty(R) if bty else jnp.zeros_like(R)
    Z_ati = ati(R) if ati else jnp.zeros_like(R)
    if Z_ati.shape == ():
        Z_ati = jnp.ones_like(R) * Z_ati
    if Z_bty.shape == ():
        Z_bty = jnp.ones_like(R) * Z_bty

    if plot_bty:
        ax.plot(R, Z_bty, "k--", label="Bathymetry")
        if fill_bty:
            ax.fill_between(R, Z_bty, jnp.max(Z_bty), color=earthbrown)

    if plot_ati:
        ax.plot(R, Z_ati, "b--", label="Altimetry")
        if fill_ati:
            ax.fill_between(R, Z_ati, jnp.min(Z_ati), color='gray', alpha=0.5)

    if plot_src:
        ax.plot(r_src, z_src, "ro", label="Source")
        ax.text(r_src, z_src, 'Source', fontsize=10, color='red')
        
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Depth/Height (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Environment")
    if legend:
        ax.legend()
    ax.grid(grid)

    return fig, ax

def plot_sound_speed_profile(Z_max, c=c, r0=0.0, legend=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    Z = jnp.linspace(0, Z_max, 1000)
    C = c(r0, Z)
    ax.plot(C, Z, "k--", label="Sound Speed")
    ax.set_xlabel("Sound Speed (m/s)")
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    
    ax.set_title("Sound Speed Profile")
    if legend:
        ax.legend()
    ax.grid(True)
    
    return fig, ax

def plot_sound_speed_field(R_max, Z_max, c=c):
    fig, ax = plt.subplots(figsize=(10, 6))
    R = jnp.linspace(0, R_max, 100)
    Z = jnp.linspace(0, Z_max, 100)
    R, Z = jnp.meshgrid(R, Z)
    C = c(R, Z)
    ax.contourf(R, Z, C, cmap='viridis')
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Depth (m)")
    if not ax.invert_yaxis():
        ax.invert_yaxis()
    ax.set_title("Sound Speed Field")
    ax.grid(True)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return fig, ax

def combine_plots(ax_env, ax_ssp):
    fig_combined = plt.figure(figsize=(15, 6))
    ax_combined_env = fig_combined.add_subplot(1, 2, 1)
    for line in ax_env.get_lines():
        ax_combined_env.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle=line.get_linestyle())
    ax_combined_env.set_xlim(ax_env.get_xlim())
    ax_combined_env.set_ylim(ax_env.get_ylim())
    ax_combined_env.set_xlabel(ax_env.get_xlabel())
    ax_combined_env.set_ylabel(ax_env.get_ylabel())
    ax_combined_env.set_title(ax_env.get_title())
    ax_combined_env.grid(True)
    if ax_env.get_legend():
        ax_combined_env.legend()
    if not ax_combined_env.yaxis_inverted():
        ax_combined_env.invert_yaxis()

    ax_combined_ssp = fig_combined.add_subplot(1, 3, 3)
    for line in ax_ssp.get_lines():
        ax_combined_ssp.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle=line.get_linestyle())
    ax_combined_ssp.set_xlim(ax_ssp.get_xlim())
    ax_combined_ssp.set_ylim(ax_ssp.get_ylim())
    ax_combined_ssp.set_xlabel(ax_ssp.get_xlabel())
    ax_combined_ssp.set_ylabel(ax_ssp.get_ylabel())
    ax_combined_ssp.set_title(ax_ssp.get_title())
    ax_combined_ssp.grid(True)
    if ax_ssp.get_legend():
        ax_combined_ssp.legend()
    if not ax_combined_ssp.yaxis_inverted():
        ax_combined_ssp.invert_yaxis()
    
    fig_combined.tight_layout()
    return fig_combined

def plot_ray_path(trj, p=1, fig=None, ax=None, x_lim=None, y_lim=None, linestyle=None, legend=False, grid=False):
    r = trj[::p, 0]
    z = trj[::p, 1]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    if linestyle is not None:
        ax.plot(r, z, linestyle, label="Ray Path")
    ax.plot(r, z, label=f"theta = {trj[0,2]:.2f} deg")
    if ax.get_xlim() is None and x_lim is not None:
        ax.set_xlim(x_lim)
    if ax.get_ylim() is None and y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Ray Path")
    if legend:
        ax.legend()
    if grid:
        ax.grid(True)
    return fig, ax

def plot_ray_paths(trjs, fig=None, ax=None, x_lim=None, y_lim=None, legend=False, grid=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    for trj in trjs:
        ax.plot(trj[:, 0], trj[:, 1], color = 'black', label=f"theta = {trj[0,2]:.2f} deg")
        
    if ax.get_xlim() is None and x_lim is not None:
        ax.set_xlim(x_lim)
    if ax.get_ylim() is None and y_lim is not None:
        ax.set_ylim(y_lim)
        
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Ray Paths")
    if legend:
        ax.legend()
    if grid:
        ax.grid(True)
    return fig, ax
