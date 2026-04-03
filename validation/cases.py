from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    title: str
    frequency_hz: float
    source_range_m: float
    source_depth_m: float
    theta_min_deg: float
    theta_max_deg: float
    n_beams: int
    ds_m: float
    max_range_m: float
    max_depth_m: float
    n_range: int
    n_depth: int
    tl_slice_depths_m: tuple[float, ...]
    sound_speed_profile: str
    bathymetry_profile: str

    @property
    def rr_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.max_range_m, self.n_range)

    @property
    def rz_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.max_depth_m, self.n_depth)


def dickins_bathymetry_profile(rr_grid: np.ndarray) -> np.ndarray:
    range_knots = np.array([0.0, 10000.0, 20000.0, 30000.0, 100000.0], dtype=float)
    depth_knots = np.array([3000.0, 3000.0, 500.0, 3000.0, 3000.0], dtype=float)
    return np.interp(rr_grid, range_knots, depth_knots)


def flat_bathymetry_profile(rr_grid: np.ndarray, depth_m: float = 5000.0) -> np.ndarray:
    return depth_m + np.zeros_like(rr_grid)


def default_ssp_profile(z_grid: np.ndarray) -> np.ndarray:
    c0_vals = np.array([
        1476.7, 1476.7, 1472.6, 1468.8, 1467.2, 1471.6, 1473.6, 1473.6,
        1472.7, 1472.2, 1471.6, 1471.6, 1472.0, 1472.7, 1473.1, 1474.9,
        1477.0, 1478.1, 1480.7, 1483.8, 1490.5, 1498.3, 1506.5,
    ], dtype=float)
    z0 = np.array([
        0.0, 38.0, 50.0, 70.0, 100.0, 140.0, 160.0, 170.0, 200.0,
        215.0, 250.0, 300.0, 370.0, 450.0, 500.0, 700.0, 900.0,
        1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0,
    ], dtype=float)
    return np.interp(z_grid, z0, c0_vals)


def munk_ssp_profile(z_grid: np.ndarray) -> np.ndarray:
    c0 = 1500.0
    eps = 0.00737
    z_axis = 1300.0
    z_scaled = 2.0 * (z_grid - z_axis) / z_axis
    return c0 * (1.0 + eps * (z_scaled - 1.0 + np.exp(-z_scaled)))


MUNK_CASE = BenchmarkCase(
    name="munk",
    title="Bellhop/JAX validation: canonical Munk profile",
    frequency_hz=50.0,
    source_range_m=0.0,
    source_depth_m=1000.0,
    theta_min_deg=-18.0,
    theta_max_deg=18.0,
    n_beams=241,
    ds_m=10.0,
    max_range_m=100000.0,
    max_depth_m=5000.0,
    n_range=801,
    n_depth=501,
    tl_slice_depths_m=(1000.0, 1500.0),
    sound_speed_profile="munk",
    bathymetry_profile="flat",
)


DICKINS_CASE = BenchmarkCase(
    name="dickins",
    title="Bellhop/JAX validation: Dickins seamount",
    frequency_hz=230.0,
    source_range_m=0.0,
    source_depth_m=18.0,
    theta_min_deg=-45.0,
    theta_max_deg=45.0,
    n_beams=181,
    ds_m=5.0,
    max_range_m=100000.0,
    max_depth_m=3000.0,
    n_range=1001,
    n_depth=601,
    tl_slice_depths_m=(18.0, 250.0, 1000.0),
    sound_speed_profile="default",
    bathymetry_profile="dickins",
)


BENCHMARK_CASES = {
    MUNK_CASE.name: MUNK_CASE,
    DICKINS_CASE.name: DICKINS_CASE,
}


def get_case(case_name: str) -> BenchmarkCase:
    return BENCHMARK_CASES[case_name]


def get_ssp_sampler(profile_name: str) -> Callable[[np.ndarray], np.ndarray]:
    samplers = {
        "default": default_ssp_profile,
        "munk": munk_ssp_profile,
    }
    return samplers[profile_name]


def get_bathymetry_sampler(profile_name: str) -> Callable[[np.ndarray], np.ndarray]:
    samplers = {
        "flat": flat_bathymetry_profile,
        "dickins": dickins_bathymetry_profile,
    }
    return samplers[profile_name]


def reference_case_dir(root: str | Path, case_name: str) -> Path:
    return Path(root) / case_name
