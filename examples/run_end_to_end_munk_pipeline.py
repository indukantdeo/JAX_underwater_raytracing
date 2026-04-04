import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(SRC / "simulation") not in sys.path:
    sys.path.append(str(SRC / "simulation"))

from simulation import boundary as boundary_mod
from simulation import dynamic_ray_tracing as dyn_mod
from simulation import sound_speed as ssp_mod
from plot import bellhop_tl_color_limits, plot_tl_field


def configure_munk_environment():
    dyn_mod.configure_acoustic_operators(
        sound_speed_operators=ssp_mod.MUNK_OPERATORS,
        boundary_operators=boundary_mod.FLAT_BOUNDARY_OPERATORS,
        reflection_model={
            "source_type": "point",
            "top_boundary_condition": "vacuum",
            "bottom_boundary_condition": "rigid",
            "kill_backward_rays": False,
        },
    )


def build_solver(rr_grid, rz_grid, theta_min, theta_max, n_beams, ds, run_mode, max_range_m, max_depth_m):
    def solve_single(freq_hz, source_depth_m):
        return dyn_mod.solve_transmission_loss_autodiff(
            freq_hz,
            0.0,
            source_depth_m,
            theta_min,
            theta_max,
            n_beams,
            rr_grid,
            rz_grid,
            ds=ds,
            beam_type="geometric",
            run_mode=run_mode,
            auto_beam_count=False,
            min_width_wavelengths=0.75,
            range_window_softness_m=2.0 * ds,
            R_max=max_range_m,
            Z_max=max_depth_m,
        )

    solve_over_sources = jax.vmap(solve_single, in_axes=(None, 0))
    solve_over_frequencies = jax.vmap(solve_over_sources, in_axes=(0, None))
    return jax.jit(solve_over_frequencies)


def build_loss(rr_grid, rz_grid, theta_min, theta_max, n_beams, ds, run_mode, max_range_m, max_depth_m):
    def loss_fn(params):
        freq_hz, source_depth_m = params
        result = dyn_mod.solve_transmission_loss_autodiff(
            freq_hz,
            0.0,
            source_depth_m,
            theta_min,
            theta_max,
            n_beams,
            rr_grid,
            rz_grid,
            ds=ds,
            beam_type="geometric",
            run_mode=run_mode,
            auto_beam_count=False,
            min_width_wavelengths=0.75,
            range_window_softness_m=2.0 * ds,
            R_max=max_range_m,
            Z_max=max_depth_m,
        )
        field = result["field_total"]
        return jnp.mean(jnp.real(field * jnp.conj(field)))

    return jax.jit(loss_fn), jax.jit(jax.grad(loss_fn))


def maybe_run_pmap(solver_fn, frequencies_hz, source_depths_m):
    device_count = jax.local_device_count()
    if device_count <= 1 or frequencies_hz.shape[0] % device_count != 0:
        return solver_fn(frequencies_hz, source_depths_m), "jit+vmap"

    shard = frequencies_hz.reshape(device_count, frequencies_hz.shape[0] // device_count)
    pmap_solver = jax.pmap(lambda freq_chunk: solver_fn(freq_chunk, source_depths_m), in_axes=0)
    result = pmap_solver(shard)
    result = jax.tree_util.tree_map(
        lambda x: x.reshape((frequencies_hz.shape[0],) + x.shape[2:]),
        result,
    )
    return result, f"pmap({device_count})+jit+vmap"


def save_outputs(out_dir, rr_grid, rz_grid, frequencies_hz, source_depths_m, result_tree, execution_mode, gradient):
    out_dir.mkdir(parents=True, exist_ok=True)

    tl_db = np.asarray(result_tree["tl_db"])
    field_total = np.asarray(result_tree["field_total"])
    np.savez_compressed(
        out_dir / "munk_pipeline_outputs.npz",
        rr_grid_m=np.asarray(rr_grid),
        rz_grid_m=np.asarray(rz_grid),
        frequencies_hz=np.asarray(frequencies_hz),
        source_depths_m=np.asarray(source_depths_m),
        tl_db=tl_db,
        field_total_real=np.real(field_total),
        field_total_imag=np.imag(field_total),
    )

    summary = {
        "execution_mode": execution_mode,
        "jax_devices": [str(device) for device in jax.devices()],
        "frequencies_hz": [float(x) for x in np.asarray(frequencies_hz)],
        "source_depths_m": [float(x) for x in np.asarray(source_depths_m)],
        "tl_db_shape": list(tl_db.shape),
        "sample_gradient": {
            "d_loss_d_frequency_hz": float(gradient[0]),
            "d_loss_d_source_depth_m": float(gradient[1]),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    n_freq = tl_db.shape[0]
    n_src = tl_db.shape[1]
    fig, axes = plt.subplots(n_freq, n_src, figsize=(5 * n_src, 3.8 * n_freq), squeeze=False)
    rz_m = np.asarray(rz_grid)
    tl_limits = bellhop_tl_color_limits(tl_db)
    for i, freq_hz in enumerate(np.asarray(frequencies_hz)):
        for j, source_depth_m in enumerate(np.asarray(source_depths_m)):
            ax = axes[i, j]
            plot_tl_field(
                rr_grid,
                rz_grid,
                tl_db[i, j],
                ax=ax,
                title="Munk TL Field",
                freq_hz=freq_hz,
                source_depth_m=source_depth_m,
                tl_limits=tl_limits,
            )
    fig.tight_layout()
    fig.savefig(out_dir / "munk_tl_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    center_freq = n_freq // 2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rr_km = np.asarray(rr_grid) / 1000.0
    for j, source_depth_m in enumerate(np.asarray(source_depths_m)):
        depth_idx = int(np.argmin(np.abs(np.asarray(rz_grid) - source_depth_m)))
        ax.plot(
            rr_km,
            tl_db[center_freq, j, depth_idx],
            linewidth=1.5,
            label=f"zs={source_depth_m:.0f} m",
        )
    ax.set_title(f"TL vs Range at f={float(np.asarray(frequencies_hz)[center_freq]):.1f} Hz")
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("TL (dB)")
    ax.set_ylim(tl_limits[1], tl_limits[0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "munk_tl_slices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="End-to-end JAX Munk-profile TL pipeline.")
    parser.add_argument("--out-dir", default="validation/results/munk_end_to_end")
    parser.add_argument("--freqs-hz", nargs="+", type=float, default=[40.0, 50.0, 60.0])
    parser.add_argument("--source-depths-m", nargs="+", type=float, default=[900.0, 1100.0])
    parser.add_argument("--n-beams", type=int, default=64)
    parser.add_argument("--ds-m", type=float, default=50.0)
    parser.add_argument("--n-range", type=int, default=301)
    parser.add_argument("--n-depth", type=int, default=201)
    parser.add_argument("--max-range-m", type=float, default=100000.0)
    parser.add_argument("--max-depth-m", type=float, default=5000.0)
    parser.add_argument("--theta-min-deg", type=float, default=-18.0)
    parser.add_argument("--theta-max-deg", type=float, default=18.0)
    parser.add_argument("--run-mode", choices=["coherent", "incoherent", "semicoherent"], default="coherent")
    args = parser.parse_args()

    configure_munk_environment()

    rr_grid = jnp.linspace(0.0, args.max_range_m, args.n_range, dtype=jnp.float64)
    rz_grid = jnp.linspace(0.0, args.max_depth_m, args.n_depth, dtype=jnp.float64)
    frequencies_hz = jnp.asarray(args.freqs_hz, dtype=jnp.float64)
    source_depths_m = jnp.asarray(args.source_depths_m, dtype=jnp.float64)
    theta_min = jnp.deg2rad(args.theta_min_deg)
    theta_max = jnp.deg2rad(args.theta_max_deg)

    solver_fn = build_solver(
        rr_grid,
        rz_grid,
        theta_min,
        theta_max,
        args.n_beams,
        args.ds_m,
        args.run_mode,
        args.max_range_m,
        args.max_depth_m,
    )
    result_tree, execution_mode = maybe_run_pmap(solver_fn, frequencies_hz, source_depths_m)

    loss_fn, grad_fn = build_loss(
        rr_grid,
        rz_grid,
        theta_min,
        theta_max,
        args.n_beams,
        args.ds_m,
        args.run_mode,
        args.max_range_m,
        args.max_depth_m,
    )
    sample_params = jnp.array([frequencies_hz[0], source_depths_m[0]], dtype=jnp.float64)
    sample_loss = loss_fn(sample_params)
    gradient = grad_fn(sample_params)

    out_dir = ROOT / args.out_dir
    save_outputs(out_dir, rr_grid, rz_grid, frequencies_hz, source_depths_m, result_tree, execution_mode, gradient)

    print(f"Execution mode: {execution_mode}")
    print(f"Output directory: {out_dir}")
    print(f"Sample loss: {float(sample_loss):.6e}")
    print(f"Sample gradient: dL/df={float(gradient[0]):.6e}, dL/dzs={float(gradient[1]):.6e}")


if __name__ == "__main__":
    main()
