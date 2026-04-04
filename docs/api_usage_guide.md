# API Usage Guide

This guide shows how to use the main solver APIs and supporting scripts.

## Choose The Right Solver Path

### Bellhop-style validation path

Use:

- `solve_transmission_loss(...)`

Use this when:

- comparing against Bellhop
- validating TL fields
- generating benchmark figures and reports

### Autodiff-safe SciML path

Use:

- `solve_transmission_loss_autodiff(...)`

Use this when:

- computing gradients with respect to source or environment parameters
- embedding the solver inside optimization or training loops
- running JAX `jit`, `grad`, `vmap`, or `pmap` workflows

## Basic Bellhop-Style Solve

```python
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
)

tl_db = result["tl_db"]
field = result["field_total"]
```

Returned keys:

- `theta`
- `launch_weights`
- `source_amplitudes`
- `trajectories`
- `field_per_beam`
- `field_total_raw`
- `field_total`
- `tl_db`

## Basic Autodiff Solve

```python
import jax
import jax.numpy as jnp
from src.simulation.dynamic_ray_tracing import solve_transmission_loss_autodiff

rr_grid = jnp.linspace(500.0, 4000.0, 64)
rz_grid = jnp.linspace(900.0, 1700.0, 48)

def objective(source_depth_m):
    result = solve_transmission_loss_autodiff(
        freq=50.0,
        r_s=0.0,
        z_s=source_depth_m,
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
    return jnp.mean(jnp.abs(result["field_total"]) ** 2)

grad_objective = jax.grad(objective)
print(grad_objective(1200.0))
```

## Configure A Custom Environment

```python
from src.simulation import boundary as boundary_mod
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation import sound_speed as ssp_mod

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
```

## Run A Validation Benchmark

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins
```

Auto beam count:

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins --auto-beam-count
```

Outputs:

- TL field CSV
- JAX TL field plot
- Bellhop vs JAX comparison plot
- TL-vs-range slice plots
- JSON and Markdown reports

## Run The End-To-End Synthetic Pipeline

```bash
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/matplotlib-jax-underwater
mkdir -p "$MPLCONFIGDIR"

JAX_underwater_raytracing/bin/python examples/run_end_to_end_munk_pipeline.py \
  --out-dir validation/results/munk_end_to_end \
  --freqs-hz 40 50 60 \
  --source-depths-m 900 1100
```

This script:

- configures a synthetic Munk environment
- runs the vectorized JAX forward solve
- computes a sample gradient
- writes field arrays and TL plots

## Plotting TL Fields

Use the Bellhop-style plotting helper:

```python
import matplotlib.pyplot as plt
from src.plot import plot_tl_field

fig, ax = plt.subplots(figsize=(12, 5))
plot_tl_field(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_db=tl_db,
    ax=ax,
    title="Transmission Loss",
    freq_hz=230.0,
    source_depth_m=18.0,
)
plt.show()
```

## Recommended JAX Patterns

- use `run_mode`, not the legacy `coherent` boolean, in new code
- use `jax.jit` around parameter sweeps or repeated solves
- use `jax.vmap` for sweeps over source depth, frequency, or source location
- use `solve_transmission_loss_autodiff(...)` if gradients matter

## Common Pitfalls

- `rr_grid` and `rz_grid` must be strictly increasing
- `theta_max` must be greater than `theta_min`
- `freq` and `ds` must be positive
- the Bellhop-style path is not guaranteed to be smoothly differentiable in all scenarios
- the autodiff-safe path is not guaranteed to match Bellhop exactly
