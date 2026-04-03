from __future__ import annotations

import numpy as np


def summarize_difference(reference: np.ndarray, candidate: np.ndarray) -> dict:
    residual = candidate - reference
    abs_residual = np.abs(residual)

    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(abs_residual)),
        "max_abs_error": float(np.max(abs_residual)),
        "p95_abs_error": float(np.percentile(abs_residual, 95.0)),
        "bias": float(np.mean(residual)),
        "reference_mean": float(np.mean(reference)),
        "candidate_mean": float(np.mean(candidate)),
    }


def safe_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.ravel()
    cand = candidate.ravel()
    if ref.size < 2:
        return float("nan")
    if np.allclose(ref.std(), 0.0) or np.allclose(cand.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(ref, cand)[0, 1])
