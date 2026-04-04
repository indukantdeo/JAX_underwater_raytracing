# Running The JAX Underwater Ray Tracing Solver

This document explains how to run the current solver code in this repository.

For architecture and API-level documentation, also see:

- `docs/solver_architecture.md`
- `docs/api_usage_guide.md`

There are now two main solver paths:

- `solve_transmission_loss(...)`
  Bellhop-style validation path. Use this when comparing against Bellhop reference cases.
- `solve_transmission_loss_autodiff(...)`
  Autodiff-safe SciML path. Use this for gradient-based optimization, inversion, or training workflows.

## Environment

Use the local virtual environment that already has JAX installed:

```bash
source JAX_underwater_raytracing/bin/activate
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/matplotlib-jax-underwater
mkdir -p "$MPLCONFIGDIR"
```

If you have a working GPU-enabled JAX install, you can change:

```bash
export JAX_PLATFORMS=gpu
```

## Core Solver Files

- `src/simulation/dynamic_ray_tracing.py`
  Main ray tracing and TL solvers
- `src/simulation/sound_speed.py`
  Sound-speed profile operators
- `src/simulation/boundary.py`
  Bathymetry and altimetry operators
- `validation/run_benchmarks.py`
  Bellhop-vs-JAX benchmark runner

## 1. Run The Bellhop-Style TL Solver

This path is intended for Bellhop-style field assembly and validation.

Example:

```bash
JAX_underwater_raytracing/bin/python - <<'PY'
import jax.numpy as jnp
from src.simulation.dynamic_ray_tracing import solve_transmission_loss

rr_grid = jnp.linspace(0.0, 100000.0, 1001)
rz_grid = jnp.linspace(0.0, 3000.0, 601)

result = solve_transmission_loss(
    freq=230.0,
    r_s=0.0,
    z_s=18.0,
    theta_min=jnp.deg2rad(-45.0),
    theta_max=jnp.deg2rad(45.0),
    n_beams=181,
    rr_grid=rr_grid,
    rz_grid=rz_grid,
    ds=5.0,
    beam_type="geometric",
    run_mode="coherent",
    accumulation_model="gaussian",
    auto_beam_count=False,
)

print(result["tl_db"].shape)
print(result["field_total"].shape)
PY
```

Important arguments:

- `run_mode`
  One of `coherent`, `incoherent`, `semicoherent`
- `accumulation_model`
  Usually `gaussian` or `hat` for Bellhop-style influence
- `auto_beam_count`
  Uses the Bellhop-style recommended beam-count heuristic

## 2. Run The Autodiff-Safe SciML Solver

This path avoids hard receiver bracketing and uses smooth receiver accumulation.

Example:

```bash
JAX_underwater_raytracing/bin/python - <<'PY'
import jax
import jax.numpy as jnp
from src.simulation.dynamic_ray_tracing import solve_transmission_loss_autodiff

rr_grid = jnp.linspace(500.0, 4000.0, 64)
rz_grid = jnp.linspace(900.0, 1700.0, 48)

result = solve_transmission_loss_autodiff(
    freq=50.0,
    r_s=0.0,
    z_s=1200.0,
    theta_min=-0.18,
    theta_max=0.18,
    n_beams=11,
    rr_grid=rr_grid,
    rz_grid=rz_grid,
    ds=25.0,
    beam_type="geometric",
    run_mode="coherent",
    min_width_wavelengths=0.75,
    range_window_softness_m=40.0,
)

loss = jnp.mean(result["tl_db"])
grad_fn = jax.grad(
    lambda source_depth: jnp.mean(
        solve_transmission_loss_autodiff(
            freq=50.0,
            r_s=0.0,
            z_s=source_depth,
            theta_min=-0.18,
            theta_max=0.18,
            n_beams=11,
            rr_grid=rr_grid,
            rz_grid=rz_grid,
            ds=25.0,
            beam_type="geometric",
            run_mode="coherent",
            min_width_wavelengths=0.75,
            range_window_softness_m=40.0,
        )["field_total"].real
    )
)

print(loss)
print(grad_fn(1200.0))
PY
```

Use this path when you need gradients with respect to:

- source depth
- frequency
- launch-angle limits
- beam pattern parameters
- any other differentiable scalar inputs passed into the forward solve

## 3. Run Bellhop-vs-JAX Validation

Run all configured validation logic for one case:

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins
```

Run with Bellhop-style automatic beam count:

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins --auto-beam-count
```

Outputs are written under:

- `validation/results/<case>/`

Key outputs:

- `*_comparison_tl_field.png`
- `*_jax_tl_field.png`
- `*_slice_<depth>m.png`
- `*_report.json`

## 4. Run The Gradient Verification Test

This verifies that the autodiff-safe solver path has a valid computational graph.

```bash
JAX_underwater_raytracing/bin/python -m unittest tests.test_differentiable_solver
```

This test uses:

- `jax.test_util.check_grads`

and checks reverse-mode and forward-mode gradients against finite differences.

## 5. Plot Dickins Ray Comparisons

Full fan comparison:

```bash
JAX_underwater_raytracing/bin/python validation/plot_dickins_ray_comparison.py
```

Custom stacked ray rows:

```bash
JAX_underwater_raytracing/bin/python validation/plot_dickins_ray_comparison.py \
  --stacked-out validation/results/dickins/dickins_ray_rows_custom.png \
  --stacked-angles-deg -18 -14 -10 -6 -2 2 6 10 14 18
```

## Notes

- The Bellhop-style path is the right path for fidelity studies against Bellhop.
- The autodiff-safe path is the right path for SciML optimization and inverse problems.
- The current validation runner uses `solve_transmission_loss(...)`, not the autodiff-safe path.
- New code should prefer the explicit `run_mode` argument over the legacy `coherent` boolean.
