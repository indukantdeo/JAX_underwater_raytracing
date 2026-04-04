from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from cases import BENCHMARK_CASES
    from read_shd_bin import pressure_to_tl_db, read_shd_bin
except ModuleNotFoundError:
    from validation.cases import BENCHMARK_CASES
    from validation.read_shd_bin import pressure_to_tl_db, read_shd_bin


DEFAULT_CASE_PATHS = {
    "munk": Path("/home/indu/MELO/OALIB_AcousticToolBox/at/tests/Munk/MunkB_Coh.shd"),
    "dickins": Path("/home/indu/MELO/OALIB_AcousticToolBox/at/tests/Dickins/DickinsB.shd"),
}


def import_case(case_name: str, shd_path: Path, reference_root: Path) -> dict:
    case = BENCHMARK_CASES[case_name]
    header, pressure = read_shd_bin(shd_path, freq_hz=case.frequency_hz)

    tl_field_db = pressure_to_tl_db(pressure[0, 0, :, :]).astype(float)
    rr_raw = np.asarray(header.receiver_r_km, dtype=float)
    rr_grid_m = rr_raw * 1000.0 if rr_raw[-1] <= 1.0e3 else rr_raw
    rz_grid_m = np.asarray(header.receiver_z_m, dtype=float)

    case_dir = reference_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(case_dir / "rr_grid_m.csv", rr_grid_m, delimiter=",")
    np.savetxt(case_dir / "rz_grid_m.csv", rz_grid_m, delimiter=",")
    np.savetxt(case_dir / "tl_field_db.csv", tl_field_db, delimiter=",")

    summary = {
        "case": case_name,
        "source_shd": str(shd_path),
        "title": header.title,
        "plot_type": header.plot_type,
        "rr_count": int(rr_grid_m.size),
        "rz_count": int(rz_grid_m.size),
        "tl_shape": list(tl_field_db.shape),
        "frequency_hz": float(case.frequency_hz),
    }
    (case_dir / "import_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Bellhop .shd outputs into validation/reference_data.")
    parser.add_argument("--case", choices=["all", *BENCHMARK_CASES.keys()], default="all")
    parser.add_argument("--reference-root", default="validation/reference_data")
    parser.add_argument("--munk-shd", default=str(DEFAULT_CASE_PATHS["munk"]))
    parser.add_argument("--dickins-shd", default=str(DEFAULT_CASE_PATHS["dickins"]))
    args = parser.parse_args()

    reference_root = Path(args.reference_root)
    shd_paths = {
        "munk": Path(args.munk_shd),
        "dickins": Path(args.dickins_shd),
    }
    case_names = list(BENCHMARK_CASES) if args.case == "all" else [args.case]
    summaries = [import_case(case_name, shd_paths[case_name], reference_root) for case_name in case_names]
    (reference_root / "import_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
