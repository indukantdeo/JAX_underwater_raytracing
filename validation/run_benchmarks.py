from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(ROOT / "src" / "simulation") not in sys.path:
    sys.path.append(str(ROOT / "src" / "simulation"))

from plot import bellhop_tl_color_limits, plot_tl_comparison, plot_tl_field

try:
    from cases import BENCHMARK_CASES, get_bathymetry_sampler, get_ssp_sampler
    from metrics import safe_correlation, summarize_difference
except ModuleNotFoundError:
    from validation.cases import BENCHMARK_CASES, get_bathymetry_sampler, get_ssp_sampler
    from validation.metrics import safe_correlation, summarize_difference


def _load_runtime_modules():
    import jax.numpy as jnp
    from simulation import boundary as boundary_mod
    from simulation import dynamic_ray_tracing as dyn_mod
    from simulation import sound_speed as ssp_mod

    return jnp, boundary_mod, dyn_mod, ssp_mod


def _configure_case(case, boundary_mod, dyn_mod, ssp_mod):
    if case.sound_speed_profile == "munk":
        sound_speed_operators = ssp_mod.MUNK_OPERATORS
    else:
        sound_speed_operators = ssp_mod.DEFAULT_OPERATORS

    if case.bathymetry_profile == "flat":
        boundary_operators = boundary_mod.FLAT_BOUNDARY_OPERATORS
    else:
        boundary_operators = boundary_mod.DEFAULT_BOUNDARY_OPERATORS

    reflection_model = {
        "source_type": "point",
        "top_boundary_condition": case.top_boundary_condition,
        "bottom_boundary_condition": case.bottom_boundary_condition,
        "kill_backward_rays": case.kill_backward_rays,
        "bottom_alpha_r_mps": case.bottom_alpha_r_mps,
        "bottom_alpha_i_user": case.bottom_alpha_i_user,
        "bottom_beta_r_mps": case.bottom_beta_r_mps,
        "bottom_beta_i_user": case.bottom_beta_i_user,
        "bottom_density_gcc": case.bottom_density_gcc,
        "attenuation_units": case.attenuation_units,
    }

    dyn_mod.configure_acoustic_operators(sound_speed_operators, boundary_operators, reflection_model)


def _run_solver(
    case,
    *,
    rr_grid=None,
    rz_grid=None,
    n_beams=None,
    auto_beam_count: bool | None = None,
    beam_chunk_size: int | None = None,
    accumulation_backend: str = "windowed",
    precision: str = "float64",
):
    """Run one benchmark case and return a NumPy-native result bundle for reporting."""
    jnp, boundary_mod, dyn_mod, ssp_mod = _load_runtime_modules()
    _configure_case(case, boundary_mod, dyn_mod, ssp_mod)

    rr_grid = jnp.asarray(case.rr_grid if rr_grid is None else rr_grid)
    rz_grid = jnp.asarray(case.rz_grid if rz_grid is None else rz_grid)
    n_beams = case.n_beams if n_beams is None else n_beams
    auto_beam_count = case.auto_beam_count if auto_beam_count is None else auto_beam_count

    t0 = time.perf_counter()
    result = dyn_mod.solve_transmission_loss(
        case.frequency_hz,
        case.source_range_m,
        case.source_depth_m,
        np.deg2rad(case.theta_min_deg),
        np.deg2rad(case.theta_max_deg),
        n_beams,
        rr_grid,
        rz_grid,
        ds=case.ds_m,
        beam_type="geometric",
        run_mode="coherent",
        accumulation_model=case.beam_influence_model,
        auto_beam_count=auto_beam_count,
        source_beam_pattern_angles_deg=case.source_beam_pattern_angles_deg,
        source_beam_pattern_db=case.source_beam_pattern_db,
        store_field_per_beam=False,
        store_trajectories=False,
        beam_chunk_size=beam_chunk_size,
        accumulation_backend=accumulation_backend,
        precision=precision,
    )
    runtime_s = time.perf_counter() - t0

    return {
        "rr_grid_m": np.asarray(rr_grid),
        "rz_grid_m": np.asarray(rz_grid),
        "tl_db": np.asarray(result["tl_db"]),
        "runtime_s": runtime_s,
        "requested_n_beams": int(n_beams),
        "actual_n_beams": int(np.asarray(result["theta"]).shape[0]),
        "recommended_n_beams": int(np.asarray(result["n_beams_recommended"]).item()),
        "auto_beam_count": bool(auto_beam_count),
        "source_amplitude_min": float(np.min(np.asarray(result["source_amplitudes"]))),
        "source_amplitude_max": float(np.max(np.asarray(result["source_amplitudes"]))),
        "timings_s": result["timings_s"],
        "storage": result["storage"],
    }


def _load_reference(case_name: str, reference_root: Path) -> dict:
    case_dir = reference_root / case_name
    required = [
        case_dir / "rr_grid_m.csv",
        case_dir / "rz_grid_m.csv",
        case_dir / "tl_field_db.csv",
    ]
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError("Missing Bellhop reference files: " + ", ".join(missing))

    rr_grid = np.loadtxt(case_dir / "rr_grid_m.csv", delimiter=",")
    rz_grid = np.loadtxt(case_dir / "rz_grid_m.csv", delimiter=",")
    tl_field_db = np.loadtxt(case_dir / "tl_field_db.csv", delimiter=",")

    return {
        "rr_grid_m": rr_grid,
        "rz_grid_m": rz_grid,
        "tl_db": tl_field_db,
    }


def _nearest_depth_index(rz_grid_m: np.ndarray, depth_m: float) -> int:
    return int(np.argmin(np.abs(rz_grid_m - depth_m)))


def _resample_reference_to_solver_grid(reference: dict, solver_result: dict) -> dict:
    rr_idx = np.array([int(np.argmin(np.abs(reference["rr_grid_m"] - rr))) for rr in solver_result["rr_grid_m"]], dtype=int)
    rz_idx = np.array([int(np.argmin(np.abs(reference["rz_grid_m"] - rz))) for rz in solver_result["rz_grid_m"]], dtype=int)
    tl_db = reference["tl_db"][np.ix_(rz_idx, rr_idx)]
    return {
        "rr_grid_m": reference["rr_grid_m"][rr_idx],
        "rz_grid_m": reference["rz_grid_m"][rz_idx],
        "tl_db": tl_db,
    }


def _save_plots(case, solver_result: dict, reference: dict | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tl_field(
        solver_result["rr_grid_m"],
        solver_result["rz_grid_m"],
        solver_result["tl_db"],
        ax=ax,
        title=f"JAX TL field: {case.name}",
        freq_hz=case.frequency_hz,
        source_depth_m=case.source_depth_m,
    )
    fig.savefig(out_dir / f"{case.name}_jax_tl_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if reference is None:
        return

    fig, axes = plot_tl_comparison(
        reference["rr_grid_m"],
        reference["rz_grid_m"],
        reference["tl_db"],
        solver_result["tl_db"],
        title_prefix=case.name,
        freq_hz=case.frequency_hz,
        source_depth_m=case.source_depth_m,
    )
    fig.savefig(out_dir / f"{case.name}_comparison_tl_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    tl_limits = bellhop_tl_color_limits(reference["tl_db"])
    rr_ref_km = reference["rr_grid_m"] / 1000.0
    for depth_m in case.tl_slice_depths_m:
        iz = _nearest_depth_index(reference["rz_grid_m"], depth_m)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rr_ref_km, reference["tl_db"][iz], label="Bellhop", linewidth=2.0)
        ax.plot(rr_ref_km, solver_result["tl_db"][iz], label="JAX", linewidth=1.5)
        ax.set_title(f"{case.name}: TL vs range at z={reference['rz_grid_m'][iz]:.1f} m")
        ax.set_xlabel("Range (km)")
        ax.set_ylabel("TL (dB)")
        ax.set_ylim(tl_limits[1], tl_limits[0])
        ax.tick_params(direction="out")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / f"{case.name}_slice_{int(round(depth_m))}m.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_report(case, solver_result: dict, reference: dict | None, out_dir: Path) -> dict:
    report = {
        "case": case.name,
        "runtime_s": solver_result["runtime_s"],
        "frequency_hz": case.frequency_hz,
        "n_beams_requested": solver_result["requested_n_beams"],
        "n_beams_actual": solver_result["actual_n_beams"],
        "n_beams_recommended": solver_result["recommended_n_beams"],
        "auto_beam_count": solver_result["auto_beam_count"],
        "source_amplitude_min": solver_result["source_amplitude_min"],
        "source_amplitude_max": solver_result["source_amplitude_max"],
        "ds_m": case.ds_m,
        "timings_s": solver_result["timings_s"],
        "storage": solver_result["storage"],
    }

    if reference is None:
        report["status"] = "solver_only"
        return report

    field_metrics = summarize_difference(reference["tl_db"], solver_result["tl_db"])
    field_metrics["correlation"] = safe_correlation(reference["tl_db"], solver_result["tl_db"])
    report["status"] = "validated_against_reference"
    report["field_metrics"] = field_metrics
    report["slice_metrics"] = {}

    for depth_m in case.tl_slice_depths_m:
        iz = _nearest_depth_index(reference["rz_grid_m"], depth_m)
        depth_key = f"{reference['rz_grid_m'][iz]:.1f}m"
        metrics = summarize_difference(reference["tl_db"][iz], solver_result["tl_db"][iz])
        metrics["correlation"] = safe_correlation(reference["tl_db"][iz], solver_result["tl_db"][iz])
        report["slice_metrics"][depth_key] = metrics

    return report


def _write_markdown_report(case, report: dict, out_dir: Path) -> None:
    lines = [
        f"# Validation Report: {case.name}",
        "",
        f"- Frequency: `{case.frequency_hz:.2f} Hz`",
        f"- Source depth: `{case.source_depth_m:.2f} m`",
        f"- Requested beams: `{report['n_beams_requested']}`",
        f"- Actual beams used: `{report['n_beams_actual']}`",
        f"- Bellhop-recommended beams: `{report['n_beams_recommended']}`",
        f"- Auto beam count: `{report['auto_beam_count']}`",
        f"- Source amplitude range: `[{report['source_amplitude_min']:.6f}, {report['source_amplitude_max']:.6f}]`",
        f"- Step size: `{case.ds_m:.2f} m`",
        f"- Solver runtime: `{report['runtime_s']:.6f} s`",
        f"- Launch fan time: `{report['timings_s']['launch_fan']:.6f} s`",
        f"- Ray rollout time: `{report['timings_s']['ray_rollout']:.6f} s`",
        f"- Accumulation time: `{report['timings_s']['accumulation']:.6f} s`",
        f"- Accumulation backend: `{report['storage']['accumulation_backend']}`",
        f"- Precision: `{report['storage']['precision']}`",
        f"- Beam chunk size requested: `{report['storage']['beam_chunk_size_requested']}`",
        f"- Beam chunk size used: `{report['storage']['beam_chunk_size_used']}`",
        f"- Status: `{report['status']}`",
        "",
    ]

    if "field_metrics" in report:
        lines.extend([
            "## Field Metrics",
            "",
            *(f"- {key}: `{value:.6f}`" for key, value in report["field_metrics"].items()),
            "",
            "## Slice Metrics",
            "",
        ])
        for depth_key, metrics in report["slice_metrics"].items():
            lines.append(f"### {depth_key}")
            lines.append("")
            lines.extend(f"- {key}: `{value:.6f}`" for key, value in metrics.items())
            lines.append("")
    else:
        lines.append("Reference Bellhop data were not available, so only the JAX-side solve and plots were generated.")
        lines.append("")

    (out_dir / f"{case.name}_report.md").write_text("\n".join(lines))


def run_case(
    case_name: str,
    reference_root: Path,
    out_root: Path,
    *,
    n_beams_override: int | None = None,
    auto_beam_count_override: bool | None = None,
    beam_chunk_size: int | None = None,
    accumulation_backend: str = "windowed",
    precision: str = "float64",
    range_stride: int = 1,
    depth_stride: int = 1,
) -> dict:
    case = BENCHMARK_CASES[case_name]
    out_dir = out_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rr_grid = case.rr_grid[::range_stride]
    rz_grid = case.rz_grid[::depth_stride]
    requested_n_beams = case.n_beams if n_beams_override is None else n_beams_override
    auto_beam_count = case.auto_beam_count if auto_beam_count_override is None else auto_beam_count_override
    case_for_report = replace(case, n_beams=requested_n_beams, auto_beam_count=auto_beam_count)
    solver_result = _run_solver(
        case,
        rr_grid=rr_grid,
        rz_grid=rz_grid,
        n_beams=n_beams_override,
        auto_beam_count=auto_beam_count,
        beam_chunk_size=beam_chunk_size,
        accumulation_backend=accumulation_backend,
        precision=precision,
    )

    reference = None
    try:
        reference = _load_reference(case_name, reference_root)
        if (
            reference["tl_db"].shape != solver_result["tl_db"].shape
            or not np.allclose(reference["rr_grid_m"], solver_result["rr_grid_m"])
            or not np.allclose(reference["rz_grid_m"], solver_result["rz_grid_m"])
        ):
            reference = _resample_reference_to_solver_grid(reference, solver_result)
    except FileNotFoundError:
        pass

    _save_plots(case_for_report, solver_result, reference, out_dir)
    report = _build_report(case_for_report, solver_result, reference, out_dir)
    report["range_stride"] = int(range_stride)
    report["depth_stride"] = int(depth_stride)
    _write_markdown_report(case_for_report, report, out_dir)
    (out_dir / f"{case.name}_report.json").write_text(json.dumps(report, indent=2))
    np.savetxt(out_dir / "rr_grid_m.csv", solver_result["rr_grid_m"], delimiter=",")
    np.savetxt(out_dir / "rz_grid_m.csv", solver_result["rz_grid_m"], delimiter=",")
    np.savetxt(out_dir / "tl_field_db.csv", solver_result["tl_db"], delimiter=",")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bellhop-vs-JAX validation and benchmarking cases.")
    parser.add_argument("--case", choices=["all", *BENCHMARK_CASES.keys()], default="all")
    parser.add_argument("--reference-root", default="validation/reference_data")
    parser.add_argument("--out-root", default="validation/results")
    parser.add_argument("--n-beams", type=int, default=None)
    parser.add_argument("--auto-beam-count", action="store_true")
    parser.add_argument("--beam-chunk-size", type=int, default=None)
    parser.add_argument("--accumulation-backend", choices=["windowed", "dense"], default="windowed")
    parser.add_argument("--precision", choices=["float64", "float32"], default="float64")
    parser.add_argument("--range-stride", type=int, default=1)
    parser.add_argument("--depth-stride", type=int, default=1)
    args = parser.parse_args()

    case_names = list(BENCHMARK_CASES) if args.case == "all" else [args.case]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = [
        run_case(
            case_name,
            Path(args.reference_root),
            out_root,
            n_beams_override=args.n_beams,
            auto_beam_count_override=args.auto_beam_count,
            beam_chunk_size=args.beam_chunk_size,
            accumulation_backend=args.accumulation_backend,
            precision=args.precision,
            range_stride=args.range_stride,
            depth_stride=args.depth_stride,
        )
        for case_name in case_names
    ]
    (out_root / "summary.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
