from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cases import BENCHMARK_CASES, get_bathymetry_sampler, get_ssp_sampler


def _write_env_template(case, out_dir: Path) -> None:
    rr_grid = case.rr_grid
    rz_grid = case.rz_grid
    bathy = get_bathymetry_sampler(case.bathymetry_profile)(rr_grid)
    ssp = get_ssp_sampler(case.sound_speed_profile)(rz_grid)

    env_text = f"""'{case.title}'
{case.frequency_hz:.3f}
1
'SVW'
{rz_grid.size} 0.0 {case.max_depth_m:.1f}
"""
    for z, c in zip(rz_grid, ssp):
        env_text += f"{z:.3f} {c:.6f} /\n"

    env_text += f"""'A' 0.0
{case.max_depth_m:.1f} 1600.0 0.0 1.0 0.0 0.0 /
1
{case.source_depth_m:.3f} /
{len(case.tl_slice_depths_m)}
{" ".join(f"{d:.3f}" for d in case.tl_slice_depths_m)} /
{case.n_range}
0.0 {case.max_range_m/1000.0:.3f} /
'C'
{case.theta_min_deg:.3f} {case.theta_max_deg:.3f} /
"""
    (out_dir / f"{case.name}.env").write_text(env_text)
    np.savetxt(out_dir / f"{case.name}_ssp.csv", np.column_stack([rz_grid, ssp]), delimiter=",", header="z_m,c_mps", comments="")
    np.savetxt(out_dir / f"{case.name}_bathy.csv", np.column_stack([rr_grid, bathy]), delimiter=",", header="r_m,bathy_m", comments="")


def _write_case_manifest(case, out_dir: Path) -> None:
    manifest = {
        "name": case.name,
        "title": case.title,
        "frequency_hz": case.frequency_hz,
        "source_range_m": case.source_range_m,
        "source_depth_m": case.source_depth_m,
        "theta_min_deg": case.theta_min_deg,
        "theta_max_deg": case.theta_max_deg,
        "n_beams": case.n_beams,
        "ds_m": case.ds_m,
        "max_range_m": case.max_range_m,
        "max_depth_m": case.max_depth_m,
        "n_range": case.n_range,
        "n_depth": case.n_depth,
        "tl_slice_depths_m": list(case.tl_slice_depths_m),
        "sound_speed_profile": case.sound_speed_profile,
        "bathymetry_profile": case.bathymetry_profile,
    }
    (out_dir / f"{case.name}.json").write_text(json.dumps(manifest, indent=2))


def export_case(case_name: str, root: Path) -> None:
    case = BENCHMARK_CASES[case_name]
    out_dir = root / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_env_template(case, out_dir)
    _write_case_manifest(case, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Bellhop validation inputs for repository benchmark cases.")
    parser.add_argument("--case", choices=["all", *BENCHMARK_CASES.keys()], default="all")
    parser.add_argument("--out", default="validation/bellhop_inputs")
    args = parser.parse_args()

    out_root = Path(args.out)
    names = list(BENCHMARK_CASES) if args.case == "all" else [args.case]
    for case_name in names:
        export_case(case_name, out_root)


if __name__ == "__main__":
    main()
