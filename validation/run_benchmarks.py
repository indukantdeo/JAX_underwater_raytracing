from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(ROOT / "src" / "simulation") not in sys.path:
    sys.path.append(str(ROOT / "src" / "simulation"))

from cases import BENCHMARK_CASES, get_bathymetry_sampler, get_ssp_sampler
from metrics import safe_correlation, summarize_difference


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

    dyn_mod.configure_acoustic_operators(sound_speed_operators, boundary_operators)


def _run_solver(case):
    jnp, boundary_mod, dyn_mod, ssp_mod = _load_runtime_modules()
    _configure_case(case, boundary_mod, dyn_mod, ssp_mod)

    rr_grid = jnp.asarray(case.rr_grid)
    rz_grid = jnp.asarray(case.rz_grid)

    t0 = time.perf_counter()
    result = dyn_mod.solve_transmission_loss(
        case.frequency_hz,
        case.source_range_m,
        case.source_depth_m,
        np.deg2rad(case.theta_min_deg),
        np.deg2rad(case.theta_max_deg),
        case.n_beams,
        rr_grid,
        rz_grid,
        ds=case.ds_m,
        beam_type="geometric",
        coherent=True,
    )
    runtime_s = time.perf_counter() - t0

    return {
        "rr_grid_m": np.asarray(rr_grid),
        "rz_grid_m": np.asarray(rz_grid),
        "tl_db": np.asarray(result["tl_db"]),
        "runtime_s": runtime_s,
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


def _save_plots(case, solver_result: dict, reference: dict | None, out_dir: Path) -> None:
    rr_km = solver_result["rr_grid_m"] / 1000.0
    rz_m = solver_result["rz_grid_m"]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        solver_result["tl_db"],
        extent=[rr_km[0], rr_km[-1], rz_m[-1], rz_m[0]],
        aspect="auto",
        cmap="jet_r",
    )
    ax.set_title(f"JAX TL field: {case.name}")
    ax.set_xlabel("Range (km)")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, ax=ax, label="TL (dB)")
    fig.savefig(out_dir / f"{case.name}_jax_tl_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if reference is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rr_ref_km = reference["rr_grid_m"] / 1000.0

    images = [
        (axes[0], reference["tl_db"], "Bellhop TL"),
        (axes[1], solver_result["tl_db"], "JAX TL"),
        (axes[2], solver_result["tl_db"] - reference["tl_db"], "Difference (JAX - Bellhop)"),
    ]
    for axis, field, title in images:
        im = axis.imshow(
            field,
            extent=[rr_ref_km[0], rr_ref_km[-1], reference["rz_grid_m"][-1], reference["rz_grid_m"][0]],
            aspect="auto",
            cmap="jet_r" if "Difference" not in title else "coolwarm",
        )
        axis.set_title(title)
        axis.set_xlabel("Range (km)")
        axis.set_ylabel("Depth (m)")
        plt.colorbar(im, ax=axis)
    fig.savefig(out_dir / f"{case.name}_comparison_tl_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for depth_m in case.tl_slice_depths_m:
        iz = _nearest_depth_index(reference["rz_grid_m"], depth_m)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rr_ref_km, reference["tl_db"][iz], label="Bellhop", linewidth=2.0)
        ax.plot(rr_ref_km, solver_result["tl_db"][iz], label="JAX", linewidth=1.5)
        ax.set_title(f"{case.name}: TL vs range at z={reference['rz_grid_m'][iz]:.1f} m")
        ax.set_xlabel("Range (km)")
        ax.set_ylabel("TL (dB)")
        ax.grid(True)
        ax.legend()
        fig.savefig(out_dir / f"{case.name}_slice_{int(round(depth_m))}m.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_report(case, solver_result: dict, reference: dict | None, out_dir: Path) -> dict:
    report = {
        "case": case.name,
        "runtime_s": solver_result["runtime_s"],
        "frequency_hz": case.frequency_hz,
        "n_beams": case.n_beams,
        "ds_m": case.ds_m,
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
        f"- Beams: `{case.n_beams}`",
        f"- Step size: `{case.ds_m:.2f} m`",
        f"- Solver runtime: `{report['runtime_s']:.6f} s`",
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


def run_case(case_name: str, reference_root: Path, out_root: Path) -> dict:
    case = BENCHMARK_CASES[case_name]
    out_dir = out_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_result = _run_solver(case)

    reference = None
    try:
        reference = _load_reference(case_name, reference_root)
    except FileNotFoundError:
        pass

    _save_plots(case, solver_result, reference, out_dir)
    report = _build_report(case, solver_result, reference, out_dir)
    _write_markdown_report(case, report, out_dir)
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
    args = parser.parse_args()

    case_names = list(BENCHMARK_CASES) if args.case == "all" else [args.case]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = [run_case(case_name, Path(args.reference_root), out_root) for case_name in case_names]
    (out_root / "summary.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
