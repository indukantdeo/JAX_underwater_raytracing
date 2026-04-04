from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShadeHeader:
    title: str
    plot_type: str
    nfreq: int
    ntheta: int
    nsx: int
    nsy: int
    nsz: int
    nrz: int
    nrr: int
    freq0_hz: float
    atten: float
    freq_vec_hz: np.ndarray
    theta_deg: np.ndarray
    source_x_m: np.ndarray
    source_y_m: np.ndarray
    source_z_m: np.ndarray
    receiver_z_m: np.ndarray
    receiver_r_km: np.ndarray


def _read_exact(fid, count: int, dtype) -> np.ndarray:
    data = np.fromfile(fid, dtype=dtype, count=count)
    if data.size != count:
        raise EOFError(f"Expected {count} values of type {dtype}, found {data.size}")
    return data


def read_shd_bin(path: str | Path, *, freq_hz: float | None = None) -> tuple[ShadeHeader, np.ndarray]:
    path = Path(path)
    with path.open("rb") as fid:
        recl = int(_read_exact(fid, 1, np.int32)[0])
        record_bytes = 4 * recl

        title = _read_exact(fid, 80, "S1").tobytes().decode("ascii", errors="ignore").rstrip("\x00 ").strip()

        fid.seek(record_bytes, 0)
        plot_type = _read_exact(fid, 10, "S1").tobytes().decode("ascii", errors="ignore")

        fid.seek(2 * record_bytes, 0)
        nfreq = int(_read_exact(fid, 1, np.int32)[0])
        ntheta = int(_read_exact(fid, 1, np.int32)[0])
        nsx = int(_read_exact(fid, 1, np.int32)[0])
        nsy = int(_read_exact(fid, 1, np.int32)[0])
        nsz = int(_read_exact(fid, 1, np.int32)[0])
        nrz = int(_read_exact(fid, 1, np.int32)[0])
        nrr = int(_read_exact(fid, 1, np.int32)[0])
        freq0_hz = float(_read_exact(fid, 1, np.float64)[0])
        atten = float(_read_exact(fid, 1, np.float64)[0])

        fid.seek(3 * record_bytes, 0)
        freq_vec_hz = _read_exact(fid, nfreq, np.float64)

        fid.seek(4 * record_bytes, 0)
        theta_deg = _read_exact(fid, ntheta, np.float64)

        fid.seek(5 * record_bytes, 0)
        if plot_type.startswith("TL"):
            sx_bounds = _read_exact(fid, 2, np.float64)
            source_x_m = np.linspace(sx_bounds[0], sx_bounds[-1], nsx)
        else:
            source_x_m = _read_exact(fid, nsx, np.float64)

        fid.seek(6 * record_bytes, 0)
        if plot_type.startswith("TL"):
            sy_bounds = _read_exact(fid, 2, np.float64)
            source_y_m = np.linspace(sy_bounds[0], sy_bounds[-1], nsy)
        else:
            source_y_m = _read_exact(fid, nsy, np.float64)

        fid.seek(7 * record_bytes, 0)
        source_z_m = _read_exact(fid, nsz, np.float32)

        fid.seek(8 * record_bytes, 0)
        receiver_z_m = _read_exact(fid, nrz, np.float32)

        fid.seek(9 * record_bytes, 0)
        receiver_r_km = _read_exact(fid, nrr, np.float64)

        ifreq = 0
        if freq_hz is not None:
            ifreq = int(np.argmin(np.abs(freq_vec_hz - freq_hz)))

        if plot_type.strip() == "irregular":
            n_receivers_per_range = 1
            pressure = np.zeros((ntheta, nsz, 1, nrr), dtype=np.complex64)
        else:
            n_receivers_per_range = nrz
            pressure = np.zeros((ntheta, nsz, nrz, nrr), dtype=np.complex64)

        for itheta in range(ntheta):
            for isz in range(nsz):
                for irz in range(n_receivers_per_range):
                    recnum = (
                        10
                        + ifreq * ntheta * nsz * n_receivers_per_range
                        + itheta * nsz * n_receivers_per_range
                        + isz * n_receivers_per_range
                        + irz
                    )
                    fid.seek(recnum * record_bytes, 0)
                    raw = _read_exact(fid, 2 * nrr, np.float32)
                    pressure[itheta, isz, irz, :] = raw[0::2] + 1j * raw[1::2]

    header = ShadeHeader(
        title=title,
        plot_type=plot_type.strip(),
        nfreq=nfreq,
        ntheta=ntheta,
        nsx=nsx,
        nsy=nsy,
        nsz=nsz,
        nrz=nrz,
        nrr=nrr,
        freq0_hz=freq0_hz,
        atten=atten,
        freq_vec_hz=freq_vec_hz,
        theta_deg=theta_deg,
        source_x_m=source_x_m,
        source_y_m=source_y_m,
        source_z_m=source_z_m,
        receiver_z_m=receiver_z_m,
        receiver_r_km=receiver_r_km,
    )
    return header, pressure


def pressure_to_tl_db(pressure: np.ndarray, *, floor_db: float = 160.0) -> np.ndarray:
    magnitude = np.abs(pressure)
    tl_db = -20.0 * np.log10(np.maximum(magnitude, 10.0 ** (-floor_db / 20.0)))
    return tl_db
