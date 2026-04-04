from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(ROOT / "src" / "simulation") not in sys.path:
    sys.path.append(str(ROOT / "src" / "simulation"))

try:
    from cases import get_bathymetry_sampler, get_case
    from run_benchmarks import _configure_case, _load_runtime_modules
except ModuleNotFoundError:
    from validation.cases import get_bathymetry_sampler, get_case
    from validation.run_benchmarks import _configure_case, _load_runtime_modules


def read_bellhop_ray(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fid:
        title = fid.readline().strip()
        freq_hz = float(fid.readline().strip().replace(",", " "))
        n_sources = tuple(int(val) for val in fid.readline().split())
        n_beam_angles = tuple(int(val) for val in fid.readline().split())
        depth_top_m = float(fid.readline().strip().replace(",", " "))
        depth_bottom_m = float(fid.readline().strip().replace(",", " "))
        ray_type = fid.readline().strip().strip("'").strip()

        n_alpha = n_beam_angles[0]
        n_source_depths = n_sources[2]
        rays = []

        for _ in range(n_source_depths):
            for _ in range(n_alpha):
                angle_line = fid.readline()
                if angle_line == "":
                    raise ValueError(f"Unexpected end of Bellhop ray file: {path}")
                angle_deg = float(angle_line.strip().replace(",", " "))

                meta = fid.readline().split()
                if len(meta) != 3:
                    raise ValueError(f"Invalid Bellhop ray metadata line in {path}: {meta}")
                n_steps, n_top_bnc, n_bot_bnc = (int(val) for val in meta)

                points = np.loadtxt(fid, max_rows=n_steps)
                points = np.atleast_2d(points)
                if points.shape[1] != 2:
                    raise ValueError(f"Expected 2 columns in Bellhop ray points, got shape {points.shape} in {path}")

                rays.append(
                    {
                        "angle_deg": angle_deg,
                        "n_top_bnc": n_top_bnc,
                        "n_bot_bnc": n_bot_bnc,
                        "points_m": points,
                    }
                )

    return {
        "title": title,
        "freq_hz": freq_hz,
        "ray_type": ray_type,
        "depth_top_m": depth_top_m,
        "depth_bottom_m": depth_bottom_m,
        "rays": rays,
    }


def _plot_ray_set(ax, rays: list[dict], bathy_r_m: np.ndarray, bathy_z_m: np.ndarray, title: str) -> None:
    bathy_r_km = bathy_r_m / 1000.0
    ax.fill_between(bathy_r_km, bathy_z_m, np.max(bathy_z_m) + 200.0, color="0.8", alpha=0.9)

    for ray in rays:
        points = ray["points_m"]
        ax.plot(points[:, 0] / 1000.0, points[:, 1], color="black", linewidth=0.7, alpha=0.6)

    source = rays[0]["points_m"][0]
    ax.scatter([source[0] / 1000.0], [source[1]], color="crimson", s=20, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(float(bathy_r_km[0]), float(bathy_r_km[-1]))
    ax.set_ylim(float(np.max(bathy_z_m)), 0.0)
    ax.grid(True, alpha=0.25)


def _nearest_ray(rays: list[dict], target_angle_deg: float) -> dict:
    return min(rays, key=lambda ray: abs(ray["angle_deg"] - target_angle_deg))


def _plot_single_ray_comparison(
    ax,
    bellhop_ray: dict,
    jax_ray: dict,
    bathy_r_m: np.ndarray,
    bathy_z_m: np.ndarray,
) -> None:
    bathy_r_km = bathy_r_m / 1000.0
    ax.fill_between(bathy_r_km, bathy_z_m, np.max(bathy_z_m) + 200.0, color="0.85", alpha=0.9)
    ax.plot(
        bellhop_ray["points_m"][:, 0] / 1000.0,
        bellhop_ray["points_m"][:, 1],
        color="black",
        linewidth=1.2,
        label=f"Bellhop {bellhop_ray['angle_deg']:.2f}°",
    )
    ax.plot(
        jax_ray["points_m"][:, 0] / 1000.0,
        jax_ray["points_m"][:, 1],
        color="crimson",
        linewidth=1.0,
        linestyle="--",
        label=f"JAX {jax_ray['angle_deg']:.2f}°",
    )
    source = bellhop_ray["points_m"][0]
    ax.scatter([source[0] / 1000.0], [source[1]], color="royalblue", s=18, zorder=5)
    ax.set_xlim(float(bathy_r_km[0]), float(bathy_r_km[-1]))
    ax.set_ylim(float(np.max(bathy_z_m)), 0.0)
    ax.set_ylabel("Depth (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Bellhop and JAX Dickins ray fans side by side.")
    parser.add_argument(
        "--bellhop-ray",
        default="/home/indu/MELO/OALIB_AcousticToolBox/at/tests/Dickins/DickinsBray.ray",
    )
    parser.add_argument("--out", default="validation/results/dickins/dickins_ray_comparison.png")
    parser.add_argument("--stacked-out", default="validation/results/dickins/dickins_ray_rows.png")
    parser.add_argument("--beam-stride", type=int, default=10)
    parser.add_argument("--ds", type=float, default=5.0)
    parser.add_argument("--stacked-angles-deg", nargs="+", type=float, default=[-6.0, -3.0, 0.0, 3.0, 6.0])
    args = parser.parse_args()

    if args.beam_stride < 1:
        raise ValueError("--beam-stride must be at least 1.")

    bellhop = read_bellhop_ray(args.bellhop_ray)
    case = get_case("dickins")

    jnp, boundary_mod, dyn_mod, ssp_mod = _load_runtime_modules()
    _configure_case(case, boundary_mod, dyn_mod, ssp_mod)

    bellhop_rays = bellhop["rays"][:: args.beam_stride]
    theta_deg = np.array([ray["angle_deg"] for ray in bellhop_rays], dtype=float)
    theta_rad = np.deg2rad(theta_deg)
    jax_trj = dyn_mod.compute_multiple_ray_paths(
        case.frequency_hz,
        case.source_range_m,
        case.source_depth_m,
        jnp.asarray(theta_rad),
        ds=args.ds,
        R_max=case.max_range_m,
        Z_max=case.max_depth_m,
        ati=dyn_mod.ati,
        bty=dyn_mod.bty,
        beam_type="geometric",
    )
    jax_trj = np.asarray(jax_trj)

    jax_rays = [{"angle_deg": angle, "points_m": trj[:, :2]} for angle, trj in zip(theta_deg, jax_trj)]

    bathy_r_m = case.rr_grid
    bathy_z_m = get_bathymetry_sampler(case.bathymetry_profile)(bathy_r_m)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    _plot_ray_set(
        axes[0],
        bellhop_rays,
        bathy_r_m,
        bathy_z_m,
        f"Bellhop Dickins Rays ({len(bellhop_rays)} shown)",
    )
    _plot_ray_set(
        axes[1],
        jax_rays,
        bathy_r_m,
        bathy_z_m,
        f"JAX Dickins Rays ({len(jax_rays)} shown)",
    )
    fig.suptitle("Dickins Bellhop vs JAX Ray Fan")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    stacked_out_path = Path(args.stacked_out)
    stacked_out_path.parent.mkdir(parents=True, exist_ok=True)
    target_angles_deg = list(args.stacked_angles_deg)
    jax_dense_angles_deg = np.array(target_angles_deg, dtype=float)
    jax_dense_angles_rad = np.deg2rad(jax_dense_angles_deg)
    jax_dense_trj = dyn_mod.compute_multiple_ray_paths(
        case.frequency_hz,
        case.source_range_m,
        case.source_depth_m,
        jnp.asarray(jax_dense_angles_rad),
        ds=args.ds,
        R_max=case.max_range_m,
        Z_max=case.max_depth_m,
        ati=dyn_mod.ati,
        bty=dyn_mod.bty,
        beam_type="geometric",
    )
    jax_dense_trj = np.asarray(jax_dense_trj)
    jax_dense_rays = [{"angle_deg": angle, "points_m": trj[:, :2]} for angle, trj in zip(jax_dense_angles_deg, jax_dense_trj)]

    fig, axes = plt.subplots(len(target_angles_deg), 1, figsize=(12, 3.2 * len(target_angles_deg)), sharex=True, sharey=True)
    if len(target_angles_deg) == 1:
        axes = [axes]
    for ax, target_angle_deg, jax_ray in zip(axes, target_angles_deg, jax_dense_rays):
        bellhop_ray = _nearest_ray(bellhop["rays"], target_angle_deg)
        _plot_single_ray_comparison(ax, bellhop_ray, jax_ray, bathy_r_m, bathy_z_m)
        ax.set_title(f"Dickins Ray Comparison Near {target_angle_deg:.1f}°")
    axes[-1].set_xlabel("Range (km)")
    fig.suptitle("Dickins Bellhop vs JAX Rays in Five Rows")
    fig.tight_layout()
    fig.savefig(stacked_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
