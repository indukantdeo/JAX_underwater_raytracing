# plot.py
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'simulation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'simulation')))

from boundary import bathymetry as bty, altimetry as ati
from sound_speed import c

# Specify the color to match Bellhop
earthbrown = (0.5, 0.3, 0.1)


def bellhop_tl_color_limits(tl_db):
    tl = np.asarray(tl_db, dtype=float)
    tl = np.where(np.isfinite(tl), tl, np.nan)
    valid = tl[np.isfinite(tl)]
    if valid.size == 0:
        return 50.0, 100.0

    tlmed = float(np.median(valid))
    tlstd = float(np.std(valid))
    tlmax = 10.0 * round((tlmed + 0.75 * tlstd) / 10.0)
    tlmin = tlmax - 50.0
    if not np.isfinite(tlmin) or not np.isfinite(tlmax) or tlmax <= tlmin:
        return 50.0, 100.0
    return tlmin, tlmax


def _format_plotshd_title(title=None, freq_hz=None, source_depth_m=None):
    title_lines = []
    if title:
        title_lines.append(str(title).replace("_", " "))
    meta = []
    if freq_hz is not None:
        meta.append(f"Freq = {float(freq_hz):g} Hz")
    if source_depth_m is not None:
        meta.append(f"z_src = {float(source_depth_m):g} m")
    if meta:
        title_lines.append("    ".join(meta))
    return "\n".join(title_lines) if title_lines else None


def plot_tl_field(
    rr_grid_m,
    rz_grid_m,
    tl_db,
    *,
    ax=None,
    title=None,
    freq_hz=None,
    source_depth_m=None,
    units="km",
    cmap="jet_r",
    tl_limits=None,
    add_colorbar=True,
    colorbar_label="TL (dB)",
):
    rr_grid_m = np.asarray(rr_grid_m, dtype=float)
    rz_grid_m = np.asarray(rz_grid_m, dtype=float)
    tl_db = np.asarray(tl_db, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    if units == "km":
        rr_plot = rr_grid_m / 1000.0
        xlab = "Range (km)"
    else:
        rr_plot = rr_grid_m
        xlab = "Range (m)"

    plot_title = _format_plotshd_title(title=title, freq_hz=freq_hz, source_depth_m=source_depth_m)

    if tl_limits is None:
        tl_limits = bellhop_tl_color_limits(tl_db)
    vmin, vmax = tl_limits

    if tl_db.ndim != 2:
        raise ValueError(f"Expected a 2D TL field, got shape {tl_db.shape}")

    if tl_db.shape[0] > 1 and tl_db.shape[1] > 1:
        mesh = ax.pcolormesh(
            rr_plot,
            rz_grid_m,
            tl_db,
            shading="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel("Depth (m)")
        ax.set_title(plot_title)
        ax.invert_yaxis()
        ax.tick_params(direction="out")
        if add_colorbar:
            cbar = plt.colorbar(mesh, ax=ax, label=colorbar_label)
            cbar.ax.tick_params(direction="out")
        return ax

    if tl_db.shape[0] == 1:
        ax.plot(rr_plot, tl_db[0], linewidth=2.0)
        ax.set_xlabel(xlab)
        ax.set_ylabel("TL (dB)")
        ax.set_title(plot_title)
        ax.invert_yaxis()
        ax.tick_params(direction="out")
        return ax

    ax.plot(tl_db[:, 0], rz_grid_m, linewidth=2.0)
    ax.set_xlabel("TL (dB)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(plot_title)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.tick_params(direction="out")
    return ax


def plot_tl_comparison(
    rr_grid_m,
    rz_grid_m,
    reference_tl_db,
    solver_tl_db,
    *,
    fig=None,
    axes=None,
    title_prefix=None,
    freq_hz=None,
    source_depth_m=None,
    units="km",
):
    reference_tl_db = np.asarray(reference_tl_db, dtype=float)
    solver_tl_db = np.asarray(solver_tl_db, dtype=float)
    tl_limits = bellhop_tl_color_limits(reference_tl_db)
    diff = solver_tl_db - reference_tl_db
    diff_lim = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    if diff_lim == 0.0:
        diff_lim = 1.0

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    prefix = f"{title_prefix}: " if title_prefix else ""

    plot_tl_field(
        rr_grid_m,
        rz_grid_m,
        reference_tl_db,
        ax=axes[0],
        title=f"{prefix}Bellhop TL",
        freq_hz=freq_hz,
        source_depth_m=source_depth_m,
        units=units,
        tl_limits=tl_limits,
        add_colorbar=True,
    )
    plot_tl_field(
        rr_grid_m,
        rz_grid_m,
        solver_tl_db,
        ax=axes[1],
        title=f"{prefix}JAX TL",
        freq_hz=freq_hz,
        source_depth_m=source_depth_m,
        units=units,
        tl_limits=tl_limits,
        add_colorbar=True,
    )
    plot_tl_field(
        rr_grid_m,
        rz_grid_m,
        diff,
        ax=axes[2],
        title=f"{prefix}Difference (JAX - Bellhop)",
        freq_hz=freq_hz,
        source_depth_m=source_depth_m,
        units=units,
        cmap="coolwarm",
        tl_limits=(-diff_lim, diff_lim),
        add_colorbar=True,
        colorbar_label="TL Difference (dB)",
    )
    return fig, axes

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
