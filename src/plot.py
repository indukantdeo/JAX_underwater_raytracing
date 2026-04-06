# plot.py
import matplotlib.pyplot as plt
from matplotlib import animation
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
skyblue = (0.7, 0.7, 1.0)


def _range_for_units(rr_grid_m, units="km"):
    rr_grid_m = np.asarray(rr_grid_m, dtype=float)
    if units == "km":
        return rr_grid_m / 1000.0, "Range (km)"
    return rr_grid_m, "Range (m)"


def _nearest_index(values, target):
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - float(target))))


def _sample_boundary(fn, r_grid_m):
    values = fn(jnp.asarray(r_grid_m, dtype=jnp.float64))
    values = np.asarray(values, dtype=float)
    if values.shape == ():
        values = np.full_like(np.asarray(r_grid_m, dtype=float), float(values))
    return values


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


def plot_tl_vs_range(
    rr_grid_m,
    rz_grid_m,
    tl_db,
    receiver_depth_m,
    *,
    ax=None,
    title=None,
    freq_hz=None,
    source_depth_m=None,
    units="km",
    linewidth=2.0,
):
    rr_plot, xlab = _range_for_units(rr_grid_m, units=units)
    tl_db = np.asarray(tl_db, dtype=float)
    iz = _nearest_index(rz_grid_m, receiver_depth_m)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rr_plot, tl_db[iz], linewidth=linewidth)
    ax.set_xlabel(xlab)
    ax.set_ylabel("TL (dB)")
    ax.set_ylim(*bellhop_tl_color_limits(tl_db)[::-1])
    ax.set_title(_format_plotshd_title(title=title, freq_hz=freq_hz, source_depth_m=source_depth_m))
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="out")
    return ax


def plot_tl_vs_depth(
    rr_grid_m,
    rz_grid_m,
    tl_db,
    receiver_range_m,
    *,
    ax=None,
    title=None,
    freq_hz=None,
    linewidth=2.0,
):
    tl_db = np.asarray(tl_db, dtype=float)
    ir = _nearest_index(rr_grid_m, receiver_range_m)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))
    ax.plot(tl_db[:, ir], np.asarray(rz_grid_m, dtype=float), linewidth=linewidth)
    ax.set_xlabel("TL (dB)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_title(_format_plotshd_title(title=title, freq_hz=freq_hz, source_depth_m=None))
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="out")
    return ax


def plot_tl_depth_frequency(
    frequencies_hz,
    rz_grid_m,
    tl_depth_frequency_db,
    *,
    ax=None,
    title=None,
    receiver_range_km=None,
    tl_limits=(60.0, 100.0),
):
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    rz_grid_m = np.asarray(rz_grid_m, dtype=float)
    tl_depth_frequency_db = np.asarray(tl_depth_frequency_db, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        frequencies_hz,
        rz_grid_m,
        tl_depth_frequency_db.T,
        shading="nearest",
        cmap="jet_r",
        vmin=tl_limits[0],
        vmax=tl_limits[1],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Depth (m)")
    ttl = title
    if receiver_range_km is not None:
        suffix = f"r_rcvr = {float(receiver_range_km):g} km"
        ttl = f"{title}\n{suffix}" if title else suffix
    ax.set_title(ttl)
    plt.colorbar(mesh, ax=ax, label="TL (dB)")
    return ax


def plot_tl_field_series(
    rr_grid_m,
    rz_grid_m,
    tl_fields,
    *,
    titles=None,
    fig=None,
    axes=None,
    units="km",
    tl_limits=None,
):
    tl_fields = np.asarray(tl_fields, dtype=float)
    n_fields = tl_fields.shape[0]
    if fig is None or axes is None:
        fig, axes = plt.subplots(n_fields, 1, figsize=(10, 3.8 * n_fields), squeeze=False)
        axes = axes[:, 0]
    if titles is None:
        titles = [None] * n_fields
    if tl_limits is None:
        tl_limits = bellhop_tl_color_limits(tl_fields)
    for idx, ax in enumerate(np.atleast_1d(axes)):
        plot_tl_field(rr_grid_m, rz_grid_m, tl_fields[idx], ax=ax, title=titles[idx], units=units, tl_limits=tl_limits)
    return fig, np.atleast_1d(axes)


def make_tl_movie(
    rr_grid_m,
    rz_grid_m,
    field_sequence,
    *,
    frame_values=None,
    db_scale=False,
    interval_ms=200,
    units="m",
    title=None,
):
    field_sequence = np.asarray(field_sequence)
    if field_sequence.ndim != 3:
        raise ValueError(f"Expected field_sequence with shape (n_frames, n_depth, n_range), got {field_sequence.shape}.")
    rr_plot, xlab = _range_for_units(rr_grid_m, units=units)
    rz_grid_m = np.asarray(rz_grid_m, dtype=float)

    if db_scale:
        tl_frames = -20.0 * np.log10(np.maximum(np.abs(field_sequence), 1e-16))
        cmap = "jet_r"
        vmin, vmax = bellhop_tl_color_limits(tl_frames)
    else:
        tl_frames = 1e6 * np.real(field_sequence)
        cmap = "jet_r"
        max_abs = float(np.nanmax(np.abs(tl_frames))) if np.isfinite(tl_frames).any() else 1.0
        max_abs = max(max_abs * 0.2, 1e-6)
        vmin, vmax = -max_abs, max_abs

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(rr_plot, rz_grid_m, tl_frames[0], shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel(xlab)
    ax.set_ylabel("Depth (m)")
    plt.colorbar(mesh, ax=ax)

    def _frame_title(frame_idx):
        if frame_values is None:
            return f"{title or 'Field Movie'}\nFrame = {frame_idx}"
        return f"{title or 'Field Movie'}\nTime = {frame_values[frame_idx]}"

    ax.set_title(_frame_title(0))

    def update(frame_idx):
        mesh.set_array(tl_frames[frame_idx].ravel())
        ax.set_title(_frame_title(frame_idx))
        return (mesh,)

    anim = animation.FuncAnimation(fig, update, frames=tl_frames.shape[0], interval=interval_ms, blit=False)
    return fig, anim


def plot_bathymetry(
    rr_grid_m,
    *,
    bathymetry_fn=bty,
    ax=None,
    units="km",
    fill=True,
    color=earthbrown,
):
    rr_plot, xlab = _range_for_units(rr_grid_m, units=units)
    rr_grid_m = np.asarray(rr_grid_m, dtype=float)
    z = _sample_boundary(bathymetry_fn, rr_grid_m)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel(xlab)
    ax.set_ylabel("Depth (m)")
    if fill:
        zmax = float(np.max(z))
        thickness = max(float(np.max(z) - np.min(z)), 1.0)
        rr_fill = np.concatenate([rr_plot, [rr_plot[-1], rr_plot[0]]])
        z_fill = np.concatenate([z, [zmax + thickness, zmax + thickness]])
        ax.fill(rr_fill, z_fill, color=color)
    else:
        ax.plot(rr_plot, z, color="k", linewidth=2.0)
    ax.invert_yaxis()
    ax.tick_params(direction="out")
    return ax


def plot_altimetry(
    rr_grid_m,
    *,
    altimetry_fn=ati,
    ax=None,
    units="km",
    fill=True,
    color=skyblue,
):
    rr_plot, xlab = _range_for_units(rr_grid_m, units=units)
    rr_grid_m = np.asarray(rr_grid_m, dtype=float)
    z = _sample_boundary(altimetry_fn, rr_grid_m)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel(xlab)
    ax.set_ylabel("Depth (m)")
    if fill:
        zmin = float(np.min(z))
        thickness = max(float(np.max(z) - np.min(z)), 1.0)
        rr_fill = np.concatenate([rr_plot, [rr_plot[-1], rr_plot[0]]])
        z_fill = np.concatenate([z, [zmin - thickness, zmin - thickness]])
        ax.fill(rr_fill, z_fill, color=color)
    else:
        ax.plot(rr_plot, z, color="b", linewidth=2.0)
    ax.invert_yaxis()
    ax.tick_params(direction="out")
    return ax

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

def plot_sound_speed_profile(
    Z_max=None,
    c=c,
    r0=0.0,
    *,
    z_grid_m=None,
    c_values_mps=None,
    sample_knots_z_m=None,
    sample_knots_c_mps=None,
    ax=None,
    legend=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    if z_grid_m is None:
        if Z_max is None:
            raise ValueError("Provide either Z_max or z_grid_m.")
        z_grid_m = jnp.linspace(0, Z_max, 1000)
    if c_values_mps is None:
        c_values_mps = c(r0, z_grid_m)
    ax.plot(c_values_mps, z_grid_m, "b-", label="Sound Speed")
    if sample_knots_z_m is not None and sample_knots_c_mps is not None:
        ax.plot(sample_knots_c_mps, sample_knots_z_m, "ko", label="Samples")
    ax.set_xlabel("Sound Speed (m/s)")
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Sound Speed Profile")
    if legend:
        ax.legend()
    ax.grid(True)
    return ax.figure, ax

def plot_sound_speed_field(R_max, Z_max, c=c, *, units="km", ax=None, n_range=100, n_depth=100):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    R = jnp.linspace(0, R_max, n_range)
    Z = jnp.linspace(0, Z_max, n_depth)
    Rm, Zm = jnp.meshgrid(R, Z)
    C = c(Rm, Zm)
    rr_plot, xlab = _range_for_units(np.asarray(R), units=units)
    mesh = ax.pcolormesh(rr_plot, np.asarray(Z), np.asarray(C), shading="nearest", cmap='jet')
    plt.colorbar(mesh, ax=ax, label="Sound Speed (m/s)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Sound Speed Field")
    ax.grid(True)
    return ax.figure, ax

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

def _ray_plot_color(trj):
    trj = np.asarray(trj)
    if trj.shape[1] >= 11:
        top_hits = int(np.rint(np.max(trj[:, 9])))
        bot_hits = int(np.rint(np.max(trj[:, 10])))
        if top_hits >= 1 and bot_hits >= 1:
            return "k"
        if bot_hits >= 1:
            return "b"
        if top_hits >= 1:
            return "g"
    return "r"


def plot_ray_paths(
    trjs,
    fig=None,
    ax=None,
    x_lim=None,
    y_lim=None,
    legend=False,
    grid=False,
    units="m",
    color_by_bounces=True,
    linewidth=1.0,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    scale = 1.0 / 1000.0 if units == "km" else 1.0
    for trj in trjs:
        trj_np = np.asarray(trj)
        color = _ray_plot_color(trj_np) if color_by_bounces else "k"
        ax.plot(trj_np[:, 0] * scale, trj_np[:, 1], color=color, linewidth=linewidth)
        
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlabel("Range (km)" if units == "km" else "Range (m)")
    ax.set_ylabel("Depth (m)")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title("Ray Paths")
    if legend:
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color="r", label="Direct path"),
            Line2D([0], [0], color="g", label="Surface only"),
            Line2D([0], [0], color="b", label="Bottom only"),
            Line2D([0], [0], color="k", label="Surface and bottom"),
        ]
        ax.legend(handles=legend_items)
    if grid:
        ax.grid(True)
    return fig, ax


def plotshd(rr_grid_m, rz_grid_m, tl_db, **kwargs):
    return plot_tl_field(rr_grid_m, rz_grid_m, tl_db, **kwargs)


def plotshd2(rr_grid_m, rz_grid_m, tl_fields, **kwargs):
    return plot_tl_field_series(rr_grid_m, rz_grid_m, tl_fields, **kwargs)


def plotssp(*, Z_max=None, c_fn=c, r0=0.0, z_grid_m=None, c_values_mps=None, sample_knots_z_m=None, sample_knots_c_mps=None, **kwargs):
    return plot_sound_speed_profile(
        Z_max=Z_max,
        c=c_fn,
        r0=r0,
        z_grid_m=z_grid_m,
        c_values_mps=c_values_mps,
        sample_knots_z_m=sample_knots_z_m,
        sample_knots_c_mps=sample_knots_c_mps,
        **kwargs,
    )


def plotssp2d(R_max, Z_max, *, c_fn=c, **kwargs):
    return plot_sound_speed_field(R_max, Z_max, c=c_fn, **kwargs)


def plotray(trjs, **kwargs):
    return plot_ray_paths(trjs, **kwargs)


def plottld(rr_grid_m, rz_grid_m, tl_db, receiver_range_km, **kwargs):
    return plot_tl_vs_depth(rr_grid_m, rz_grid_m, tl_db, receiver_range_m=1000.0 * receiver_range_km, **kwargs)


def plottlr(rr_grid_m, rz_grid_m, tl_db, receiver_depth_m, **kwargs):
    return plot_tl_vs_range(rr_grid_m, rz_grid_m, tl_db, receiver_depth_m=receiver_depth_m, **kwargs)


def plotmovie(rr_grid_m, rz_grid_m, field_sequence, **kwargs):
    return make_tl_movie(rr_grid_m, rz_grid_m, field_sequence, **kwargs)


def plottl_zf(frequencies_hz, rz_grid_m, tl_depth_frequency_db, **kwargs):
    return plot_tl_depth_frequency(frequencies_hz, rz_grid_m, tl_depth_frequency_db, **kwargs)


def plotbty(rr_grid_m, **kwargs):
    return plot_bathymetry(rr_grid_m, **kwargs)


def plotati(rr_grid_m, **kwargs):
    return plot_altimetry(rr_grid_m, **kwargs)
