import argparse
import json
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

from plot import plot_tl_field
from simulation import dynamic_ray_tracing as dyn_mod
from simulation.boundary import make_boundary_operators
from simulation.sound_speed import make_2d_ssp_operators


def _strip_comment(line: str) -> str:
    return line.split("!")[0].strip()


def _unquote(token: str) -> str:
    return token.strip().strip("'").strip('"')


def _parse_value_line(line: str) -> list[str]:
    cleaned = _strip_comment(line).replace("/", " ")
    return cleaned.split()


def _parse_bellhop_env(env_path: Path) -> dict:
    lines = [_strip_comment(line) for line in env_path.read_text().splitlines()]
    lines = [line for line in lines if line]

    title = _unquote(lines[0])
    freq_hz = float(_parse_value_line(lines[1])[0])
    _nmedia = int(_parse_value_line(lines[2])[0])
    sspopt = _unquote(_parse_value_line(lines[3])[0])
    bottom_descriptor = _parse_value_line(lines[4])

    ssp_depths_m: list[float] = []
    ssp_speeds_mps: list[float] = []
    idx = 5
    while idx < len(lines):
        candidate = lines[idx]
        if candidate.startswith("'") or candidate.startswith('"'):
            break
        tokens = _parse_value_line(candidate)
        if len(tokens) < 2:
            raise ValueError(f"Malformed SSP line in {env_path}: {candidate!r}")
        ssp_depths_m.append(float(tokens[0]))
        ssp_speeds_mps.append(float(tokens[1]))
        idx += 1

    boundary_option_tokens = _parse_value_line(lines[idx])
    idx += 1
    bottom_tokens = _parse_value_line(lines[idx])
    idx += 1
    nsd = int(_parse_value_line(lines[idx])[0])
    idx += 1
    source_depth_tokens = _parse_value_line(lines[idx])
    idx += 1
    nrd = int(_parse_value_line(lines[idx])[0])
    idx += 1
    receiver_depth_tokens = _parse_value_line(lines[idx])
    idx += 1
    nr = int(_parse_value_line(lines[idx])[0])
    idx += 1
    receiver_range_tokens = _parse_value_line(lines[idx])
    idx += 1
    run_type_tokens = _parse_value_line(lines[idx])
    idx += 1
    nbeams = int(float(_parse_value_line(lines[idx])[0]))
    idx += 1
    alpha_tokens = _parse_value_line(lines[idx])
    idx += 1
    step_tokens = _parse_value_line(lines[idx])

    if len(source_depth_tokens) < nsd:
        raise ValueError(f"Expected {nsd} source depths in {env_path}, got {source_depth_tokens}.")
    if len(receiver_depth_tokens) < 2:
        raise ValueError(f"Expected receiver depth bounds in {env_path}, got {receiver_depth_tokens}.")
    if len(receiver_range_tokens) < 2:
        raise ValueError(f"Expected receiver range bounds in {env_path}, got {receiver_range_tokens}.")
    if len(alpha_tokens) < 2:
        raise ValueError(f"Expected launch angle bounds in {env_path}, got {alpha_tokens}.")
    if len(step_tokens) < 3:
        raise ValueError(f"Expected STEP/ZBOX/RBOX values in {env_path}, got {step_tokens}.")

    source_depths_m = np.array([float(value) for value in source_depth_tokens[:nsd]], dtype=float)
    rz_grid_m = np.linspace(float(receiver_depth_tokens[0]), float(receiver_depth_tokens[1]), nrd, dtype=float)
    rr_grid_m = np.linspace(float(receiver_range_tokens[0]), float(receiver_range_tokens[1]) * 1000.0, nr, dtype=float)

    boundary_option = _unquote(boundary_option_tokens[0]) if boundary_option_tokens else ""
    run_type = _unquote(run_type_tokens[0]) if run_type_tokens else "CB"

    return {
        "title": title,
        "frequency_hz": freq_hz,
        "ssp_option": sspopt,
        "ssp_depths_m": np.asarray(ssp_depths_m, dtype=float),
        "ssp_speeds_mps": np.asarray(ssp_speeds_mps, dtype=float),
        "bottom_descriptor": bottom_descriptor,
        "boundary_option": boundary_option,
        "bottom_properties": {
            "depth_m": float(bottom_tokens[0]),
            "alpha_r_mps": float(bottom_tokens[1]),
            "beta_r_mps": float(bottom_tokens[2]),
            "density_gcc": float(bottom_tokens[3]),
            "alpha_i_user": float(bottom_tokens[4]),
        },
        "source_depth_m": float(source_depths_m[0]),
        "rz_grid_m": rz_grid_m,
        "rr_grid_m": rr_grid_m,
        "run_type": run_type,
        "n_beams": nbeams,
        "theta_min_deg": float(alpha_tokens[0]),
        "theta_max_deg": float(alpha_tokens[1]),
        "step_m": float(step_tokens[0]),
        "zbox_m": float(step_tokens[1]),
        "rbox_m": float(step_tokens[2]) * 1000.0,
    }


def _parse_bellhop_bty(bty_path: Path) -> dict:
    lines = [_strip_comment(line) for line in bty_path.read_text().splitlines()]
    lines = [line for line in lines if line]
    interpolation = _unquote(lines[0])
    n_points = int(lines[1])
    knots = np.array([[float(value) for value in _parse_value_line(line)[:2]] for line in lines[2 : 2 + n_points]], dtype=float)
    if knots.shape[0] != n_points:
        raise ValueError(f"Expected {n_points} bathymetry points in {bty_path}, got {knots.shape[0]}.")
    return {
        "interpolation": interpolation,
        "range_knots_m": knots[:, 0] * 1000.0,
        "depth_knots_m": knots[:, 1],
    }


def _build_dickins_operators(env_data: dict, bty_data: dict):
    z_knots_m = jnp.asarray(env_data["ssp_depths_m"], dtype=jnp.float64)
    c_knots_mps = jnp.asarray(env_data["ssp_speeds_mps"], dtype=jnp.float64)
    r_bathy_m = jnp.asarray(bty_data["range_knots_m"], dtype=jnp.float64)
    z_bathy_m = jnp.asarray(bty_data["depth_knots_m"], dtype=jnp.float64)

    @jax.jit
    def ssp_from_table(r, z):
        del r
        return jnp.interp(z, z_knots_m, c_knots_mps)

    @jax.jit
    def bathymetry_from_table(r):
        return jnp.interp(r, r_bathy_m, z_bathy_m)

    @jax.jit
    def altimetry_flat(r):
        return 0.0 * r

    ssp_operators = make_2d_ssp_operators(ssp_from_table)
    ssp_operators["interface_depths_m"] = z_knots_m

    boundary_operators = make_boundary_operators(
        bathymetry_fn=bathymetry_from_table,
        altimetry_fn=altimetry_flat,
    )
    boundary_operators["bathymetry_range_breaks_m"] = r_bathy_m
    boundary_operators["altimetry_range_breaks_m"] = jnp.asarray([], dtype=jnp.float64)
    return ssp_operators, boundary_operators


def _run_mode_from_bellhop(run_type: str) -> str:
    run_type = run_type.upper()
    if "C" in run_type:
        return "coherent"
    if "S" in run_type:
        return "semicoherent"
    if "I" in run_type:
        return "incoherent"
    return "coherent"


def main():
    parser = argparse.ArgumentParser(description="Run the Dickins Bellhop case in the JAX solver using .env and .bty inputs.")
    parser.add_argument(
        "--env",
        default="/home/indu/MELO/OALIB_AcousticToolBox/at/tests/Dickins/DickinsB.env",
        help="Path to Bellhop DickinsB.env",
    )
    parser.add_argument(
        "--bty",
        default="/home/indu/MELO/OALIB_AcousticToolBox/at/tests/Dickins/DickinsB.bty",
        help="Path to Bellhop DickinsB.bty",
    )
    parser.add_argument(
        "--out-dir",
        default="validation/results/dickins_env_jax",
        help="Directory to store TL field outputs and plots.",
    )
    parser.add_argument(
        "--accumulation-model",
        choices=["bellhop", "gaussian", "hat"],
        default="gaussian",
        help="Receiver-grid accumulation model for the Bellhop-style solver path.",
    )
    parser.add_argument(
        "--auto-beam-count",
        action="store_true",
        help="Replace the requested beam count with the Bellhop-style recommended beam count heuristic.",
    )
    parser.add_argument(
        "--n-beams-override",
        type=int,
        default=None,
        help="Override the beam count from the .env file. If omitted, uses the .env value unless it is zero.",
    )
    args = parser.parse_args()

    env_path = Path(args.env)
    bty_path = Path(args.bty)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_data = _parse_bellhop_env(env_path)
    bty_data = _parse_bellhop_bty(bty_path)
    ssp_operators, boundary_operators = _build_dickins_operators(env_data, bty_data)

    reflection_model = {
        "source_type": "point",
        "top_boundary_condition": "vacuum",
        "bottom_boundary_condition": "acoustic_halfspace",
        "kill_backward_rays": True,
        "bottom_alpha_r_mps": env_data["bottom_properties"]["alpha_r_mps"],
        "bottom_alpha_i_user": env_data["bottom_properties"]["alpha_i_user"],
        "bottom_beta_r_mps": env_data["bottom_properties"]["beta_r_mps"],
        "bottom_beta_i_user": 0.0,
        "bottom_density_gcc": env_data["bottom_properties"]["density_gcc"],
        "attenuation_units": "W",
    }

    dyn_mod.configure_acoustic_operators(
        sound_speed_operators=ssp_operators,
        boundary_operators=boundary_operators,
        reflection_model=reflection_model,
    )

    env_n_beams = int(env_data["n_beams"])
    requested_n_beams = args.n_beams_override if args.n_beams_override is not None else (181 if env_n_beams == 0 else env_n_beams)
    rr_grid = jnp.asarray(env_data["rr_grid_m"], dtype=jnp.float64)
    rz_grid = jnp.asarray(env_data["rz_grid_m"], dtype=jnp.float64)

    result = dyn_mod.solve_transmission_loss(
        freq=env_data["frequency_hz"],
        r_s=0.0,
        z_s=env_data["source_depth_m"],
        theta_min=jnp.deg2rad(env_data["theta_min_deg"]),
        theta_max=jnp.deg2rad(env_data["theta_max_deg"]),
        n_beams=requested_n_beams,
        rr_grid=rr_grid,
        rz_grid=rz_grid,
        ds=env_data["step_m"] if env_data["step_m"] > 0.0 else 5.0,
        beam_type="geometric",
        run_mode=_run_mode_from_bellhop(env_data["run_type"]),
        accumulation_model=args.accumulation_model,
        auto_beam_count=args.auto_beam_count,
        R_max=env_data["rbox_m"],
        Z_max=env_data["zbox_m"],
    )

    tl_db = np.asarray(result["tl_db"])
    rr_grid_m = np.asarray(rr_grid)
    rz_grid_m = np.asarray(rz_grid)

    np.savetxt(out_dir / "tl_field_db.csv", tl_db, delimiter=",")
    np.savetxt(out_dir / "rr_grid_m.csv", rr_grid_m, delimiter=",")
    np.savetxt(out_dir / "rz_grid_m.csv", rz_grid_m, delimiter=",")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tl_field(
        rr_grid_m=rr_grid_m,
        rz_grid_m=rz_grid_m,
        tl_db=tl_db,
        ax=ax,
        title=env_data["title"],
        freq_hz=env_data["frequency_hz"],
        source_depth_m=env_data["source_depth_m"],
    )
    fig.savefig(out_dir / "dickins_env_jax_tl_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "env_path": str(env_path),
        "bty_path": str(bty_path),
        "title": env_data["title"],
        "frequency_hz": env_data["frequency_hz"],
        "source_depth_m": env_data["source_depth_m"],
        "theta_min_deg": env_data["theta_min_deg"],
        "theta_max_deg": env_data["theta_max_deg"],
        "requested_n_beams": requested_n_beams,
        "actual_n_beams": int(np.asarray(result["theta"]).shape[0]),
        "auto_beam_count": bool(args.auto_beam_count),
        "run_mode": _run_mode_from_bellhop(env_data["run_type"]),
        "accumulation_model": args.accumulation_model,
        "rr_shape": int(rr_grid_m.shape[0]),
        "rz_shape": int(rz_grid_m.shape[0]),
        "plot_path": str(out_dir / "dickins_env_jax_tl_field.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved TL field plot to: {out_dir / 'dickins_env_jax_tl_field.png'}")
    print(f"Saved TL field CSV to: {out_dir / 'tl_field_db.csv'}")
    print(f"Requested beams: {requested_n_beams}")
    print(f"Actual beams used: {int(np.asarray(result['theta']).shape[0])}")


if __name__ == "__main__":
    main()
