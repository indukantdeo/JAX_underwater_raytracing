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

## Specify SSP, Bathymetry, And Altimetry In Python

The solver is configured through:

- `make_2d_ssp_operators(...)` in `src/simulation/sound_speed.py`
- `make_boundary_operators(...)` in `src/simulation/boundary.py`
- `configure_acoustic_operators(...)` in `src/simulation/dynamic_ray_tracing.py`

In practice, you can specify the environment in two common ways:

1. analytic Python/JAX functions
2. sampled tables, interpolated with `jnp.interp`

### Option 1: Specify SSP As A Python Function

If you have an analytic SSP `c(r, z)`, wrap it with `make_2d_ssp_operators(...)`. The factory automatically builds the derivative operators needed by the tracer.

```python
import jax.numpy as jnp
from src.simulation.sound_speed import make_2d_ssp_operators

def c_custom(r, z):
    c0 = 1480.0
    gradient = 0.017
    range_perturbation = 3.0 * jnp.sin(r / 20000.0)
    return c0 + gradient * z + range_perturbation

CUSTOM_SSP_OPERATORS = make_2d_ssp_operators(c_custom)
```

Use this pattern when:

- your sound-speed model is naturally analytic
- you want JAX to differentiate the SSP and its derivatives automatically

### Option 2: Specify SSP As A Depth Table `z` vs `c(z)`

If your SSP comes from measured or tabulated data, define depth and sound-speed arrays and interpolate them.

```python
import jax
import jax.numpy as jnp
from src.simulation.sound_speed import make_2d_ssp_operators

z_knots_m = jnp.array([0.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0], dtype=jnp.float64)
c_knots_mps = jnp.array([1505.0, 1498.0, 1492.0, 1488.0, 1486.0, 1490.0, 1502.0], dtype=jnp.float64)

@jax.jit
def c_from_table(r, z):
    del r
    return jnp.interp(z, z_knots_m, c_knots_mps)

TABLE_SSP_OPERATORS = make_2d_ssp_operators(c_from_table)
TABLE_SSP_OPERATORS["interface_depths_m"] = z_knots_m
```

Notes:

- `interface_depths_m` is optional but recommended when using layered or piecewise SSP data.
- the tracer uses those depths for step reduction near SSP interfaces.

### Option 3: Specify Bathymetry And Altimetry As Python Functions

Bathymetry and altimetry are scalar functions of range `r`.

```python
import jax.numpy as jnp
from src.simulation.boundary import make_boundary_operators

def bathymetry_fn(r):
    return 3000.0 - 2200.0 * jnp.exp(-((r - 25000.0) / 7000.0) ** 2)

def altimetry_fn(r):
    return 5.0 * jnp.sin(r / 8000.0)

CUSTOM_BOUNDARY_OPERATORS = make_boundary_operators(
    bathymetry_fn=bathymetry_fn,
    altimetry_fn=altimetry_fn,
)
```

Use this when:

- bottom or surface shape is known analytically
- you want JAX to differentiate slope and normals automatically

Important requirement:

- these functions must be JAX-compatible because the boundary operator factory differentiates them with `jax.grad`

### Option 4: Specify Bathymetry And Altimetry As Range Tables

If the geometry comes from sampled survey data, interpolate from range knots.

```python
import jax
import jax.numpy as jnp
from src.simulation.boundary import make_boundary_operators

r_bathy_m = jnp.array([0.0, 10000.0, 20000.0, 30000.0, 100000.0], dtype=jnp.float64)
z_bathy_m = jnp.array([3000.0, 3000.0, 500.0, 3000.0, 3000.0], dtype=jnp.float64)

r_surface_m = jnp.array([0.0, 50000.0, 100000.0], dtype=jnp.float64)
z_surface_m = jnp.array([0.0, 2.0, 0.0], dtype=jnp.float64)

@jax.jit
def bathymetry_from_table(r):
    return jnp.interp(r, r_bathy_m, z_bathy_m)

@jax.jit
def altimetry_from_table(r):
    return jnp.interp(r, r_surface_m, z_surface_m)

TABLE_BOUNDARY_OPERATORS = make_boundary_operators(
    bathymetry_fn=bathymetry_from_table,
    altimetry_fn=altimetry_from_table,
)
TABLE_BOUNDARY_OPERATORS["bathymetry_range_breaks_m"] = r_bathy_m
TABLE_BOUNDARY_OPERATORS["altimetry_range_breaks_m"] = r_surface_m
```

Notes:

- `bathymetry_range_breaks_m` and `altimetry_range_breaks_m` are optional but recommended for piecewise or tabulated geometry.
- the tracer uses those breakpoints to reduce step size near slope changes and range-segment boundaries.

### Full Example: Configure The Solver With Table-Based SSP And Bathymetry

```python
import jax
import jax.numpy as jnp
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation.boundary import make_boundary_operators
from src.simulation.sound_speed import make_2d_ssp_operators

z_knots_m = jnp.array([0.0, 100.0, 300.0, 1000.0, 3000.0], dtype=jnp.float64)
c_knots_mps = jnp.array([1502.0, 1494.0, 1488.0, 1490.0, 1504.0], dtype=jnp.float64)

@jax.jit
def c_from_table(r, z):
    del r
    return jnp.interp(z, z_knots_m, c_knots_mps)

ssp_operators = make_2d_ssp_operators(c_from_table)
ssp_operators["interface_depths_m"] = z_knots_m

r_bathy_m = jnp.array([0.0, 20000.0, 50000.0, 100000.0], dtype=jnp.float64)
z_bathy_m = jnp.array([3500.0, 3200.0, 2800.0, 3000.0], dtype=jnp.float64)

@jax.jit
def bathymetry_from_table(r):
    return jnp.interp(r, r_bathy_m, z_bathy_m)

@jax.jit
def flat_surface(r):
    return 0.0 * r

boundary_operators = make_boundary_operators(
    bathymetry_fn=bathymetry_from_table,
    altimetry_fn=flat_surface,
)
boundary_operators["bathymetry_range_breaks_m"] = r_bathy_m

dyn_mod.configure_acoustic_operators(
    sound_speed_operators=ssp_operators,
    boundary_operators=boundary_operators,
    reflection_model={
        "source_type": "point",
        "top_boundary_condition": "vacuum",
        "bottom_boundary_condition": "acoustic_halfspace",
        "kill_backward_rays": False,
    },
)
```

After this configuration, call `solve_transmission_loss(...)` or `solve_transmission_loss_autodiff(...)` as usual.

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

Performance-oriented options:

- `--beam-chunk-size <int>`
  Process the launch fan in fixed-size beam chunks to reduce memory pressure.
- `--accumulation-backend windowed|dense`
  `windowed` is the optimized chunked Bellhop path; `dense` keeps the reference all-beam accumulation path for parity checks.
- `--precision float64|float32`
  `float64` is the default validation mode. `float32` is available for speed experiments.

## Example: Run The DickinsB Case In The JAX Solver

`DickinsB.env` is the Bellhop reference case. In this repository, the matching JAX-side benchmark case is `dickins` in `validation/cases.py`.

The simplest way to run the JAX version of the DickinsB case is through the validation runner:

```bash
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/matplotlib-jax-underwater
mkdir -p "$MPLCONFIGDIR"

JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins
```

This will:

- configure the Dickins seamount bathymetry
- configure the default repository SSP used for the Dickins benchmark
- run the coherent JAX TL solve
- save TL fields, slice plots, and comparison artifacts under `validation/results/dickins/`

If you want Bellhop-style beam-density selection instead of the fixed benchmark beam count:

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py --case dickins --auto-beam-count
```

Example with chunked accumulation:

```bash
JAX_underwater_raytracing/bin/python validation/run_benchmarks.py \
  --case dickins \
  --beam-chunk-size 64 \
  --accumulation-backend windowed \
  --precision float64
```

If you want to run the Dickins case directly from Python without the benchmark harness:

```python
import jax.numpy as jnp
from validation.cases import DICKINS_CASE
from src.simulation import boundary as boundary_mod
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation import sound_speed as ssp_mod

dyn_mod.configure_acoustic_operators(
    sound_speed_operators=ssp_mod.DEFAULT_OPERATORS,
    boundary_operators=boundary_mod.DEFAULT_BOUNDARY_OPERATORS,
    reflection_model={
        "source_type": "point",
        "top_boundary_condition": DICKINS_CASE.top_boundary_condition,
        "bottom_boundary_condition": DICKINS_CASE.bottom_boundary_condition,
        "kill_backward_rays": DICKINS_CASE.kill_backward_rays,
        "bottom_alpha_r_mps": DICKINS_CASE.bottom_alpha_r_mps,
        "bottom_alpha_i_user": DICKINS_CASE.bottom_alpha_i_user,
        "bottom_beta_r_mps": DICKINS_CASE.bottom_beta_r_mps,
        "bottom_beta_i_user": DICKINS_CASE.bottom_beta_i_user,
        "bottom_density_gcc": DICKINS_CASE.bottom_density_gcc,
        "attenuation_units": DICKINS_CASE.attenuation_units,
    },
)

rr_grid = jnp.asarray(DICKINS_CASE.rr_grid)
rz_grid = jnp.asarray(DICKINS_CASE.rz_grid)

result = dyn_mod.solve_transmission_loss(
    freq=DICKINS_CASE.frequency_hz,
    r_s=DICKINS_CASE.source_range_m,
    z_s=DICKINS_CASE.source_depth_m,
    theta_min=jnp.deg2rad(DICKINS_CASE.theta_min_deg),
    theta_max=jnp.deg2rad(DICKINS_CASE.theta_max_deg),
    n_beams=DICKINS_CASE.n_beams,
    rr_grid=rr_grid,
    rz_grid=rz_grid,
    ds=DICKINS_CASE.ds_m,
    beam_type="geometric",
    run_mode="coherent",
    accumulation_model=DICKINS_CASE.beam_influence_model,
)

print(result["tl_db"].shape)
```

## Example: Run A Munk Profile With Gaussian Beam Accumulation

This example uses the Bellhop-style solver path, configures a flat-bottom Munk environment, and explicitly selects Gaussian beam accumulation:

```python
import jax.numpy as jnp
from src.simulation import boundary as boundary_mod
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation import sound_speed as ssp_mod

dyn_mod.configure_acoustic_operators(
    sound_speed_operators=ssp_mod.MUNK_OPERATORS,
    boundary_operators=boundary_mod.FLAT_BOUNDARY_OPERATORS,
    reflection_model={
        "source_type": "point",
        "top_boundary_condition": "vacuum",
        "bottom_boundary_condition": "acoustic_halfspace",
        "bottom_alpha_r_mps": 1600.0,
        "bottom_alpha_i_user": 0.8,
        "bottom_density_gcc": 1.8,
        "attenuation_units": "W",
        "kill_backward_rays": False,
    },
)

rr_grid = jnp.linspace(0.0, 100000.0, 1001)
rz_grid = jnp.linspace(0.0, 5000.0, 501)

result = dyn_mod.solve_transmission_loss(
    freq=50.0,
    r_s=0.0,
    z_s=1000.0,
    theta_min=jnp.deg2rad(-20.3),
    theta_max=jnp.deg2rad(20.3),
    n_beams=241,
    rr_grid=rr_grid,
    rz_grid=rz_grid,
    ds=50.0,
    beam_type="geometric",
    run_mode="coherent",
    accumulation_model="gaussian",
    auto_beam_count=False,
)

print(result["tl_db"].shape)
print(result["field_total"].shape)
```

To plot the resulting field with the Bellhop-style TL display rules:

```python
import matplotlib.pyplot as plt
from src.plot import plot_tl_field

fig, ax = plt.subplots(figsize=(12, 5))
plot_tl_field(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_db=result["tl_db"],
    ax=ax,
    title="Munk Profile Gaussian-Beam TL",
    freq_hz=50.0,
    source_depth_m=1000.0,
)
plt.show()
```

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

## Plotting Utilities Equivalent To Bellhop MATLAB Tools

The plotting helpers in `src/plot.py` now include Bellhop-style wrapper names so you can work with solver outputs using function names similar to:

- `plotshd`, `plotshd2`
- `plotssp`, `plotssp2d`
- `plotray`
- `plottld`, `plottlr`
- `plotbty`, `plotati`
- `plotmovie`
- `plottl_zf`

### Full-Field TL Plot: `plotshd`

```python
import matplotlib.pyplot as plt
from src.plot import plotshd

fig, ax = plt.subplots(figsize=(12, 5))
plotshd(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_db=result["tl_db"],
    ax=ax,
    title="Dickins TL",
    freq_hz=230.0,
    source_depth_m=18.0,
)
plt.show()
```

### Multi-Panel TL Plot: `plotshd2`

```python
import numpy as np
from src.plot import plotshd2

tl_fields = np.stack([result_a["tl_db"], result_b["tl_db"]], axis=0)
fig, axes = plotshd2(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_fields=tl_fields,
    titles=["Source depth = 10 m", "Source depth = 20 m"],
)
```

### SSP Profile: `plotssp`

```python
import matplotlib.pyplot as plt
from src.plot import plotssp

fig, ax = plt.subplots(figsize=(6, 8))
plotssp(
    Z_max=5000.0,
    c_fn=ssp_mod.MUNK_OPERATORS["c"],
    r0=0.0,
    ax=ax,
)
plt.show()
```

For a tabulated SSP:

```python
plotssp(
    z_grid_m=z_knots_m,
    c_values_mps=c_knots_mps,
    sample_knots_z_m=z_knots_m,
    sample_knots_c_mps=c_knots_mps,
)
```

### 2D SSP Field: `plotssp2d`

```python
from src.plot import plotssp2d

fig, ax = plotssp2d(
    R_max=100000.0,
    Z_max=5000.0,
    c_fn=ssp_mod.MUNK_OPERATORS["c"],
    units="km",
)
```

### Ray Trajectories: `plotray`

```python
from src.plot import plotray

fig, ax = plotray(
    result["trajectories"],
    units="km",
    color_by_bounces=True,
    grid=True,
)
```

The ray plot colors are Bellhop-style:

- red: direct path
- green: surface only
- blue: bottom only
- black: both surface and bottom

### TL vs Depth: `plottld`

```python
from src.plot import plottld

fig, ax = plt.subplots(figsize=(6, 8))
plottld(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_db=result["tl_db"],
    receiver_range_km=20.0,
    ax=ax,
    title="TL vs Depth",
    freq_hz=230.0,
)
plt.show()
```

### TL vs Range: `plottlr`

```python
from src.plot import plottlr

fig, ax = plt.subplots(figsize=(10, 4))
plottlr(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    tl_db=result["tl_db"],
    receiver_depth_m=250.0,
    ax=ax,
    title="TL vs Range",
    freq_hz=230.0,
    source_depth_m=18.0,
)
plt.show()
```

### Bathymetry And Altimetry: `plotbty`, `plotati`

```python
import numpy as np
import matplotlib.pyplot as plt
from src.plot import plotati, plotbty

rr_plot = np.linspace(0.0, 100000.0, 400)
fig, ax = plt.subplots(figsize=(12, 5))
plotbty(rr_plot, ax=ax, units="km", fill=True)
plotati(rr_plot, ax=ax, units="km", fill=True)
plt.show()
```

### TL Movie: `plotmovie`

```python
from src.plot import plotmovie

field_sequence = np.stack(
    [result_t0["field_total"], result_t1["field_total"], result_t2["field_total"]],
    axis=0,
)
fig, anim = plotmovie(
    rr_grid_m=rr_grid,
    rz_grid_m=rz_grid,
    field_sequence=field_sequence,
    frame_values=[0.0, 1.0, 2.0],
    db_scale=False,
    units="m",
    title="Field Evolution",
)

anim.save("tl_movie.gif", writer="pillow")
```

### TL vs Depth And Frequency: `plottl_zf`

```python
from src.plot import plottl_zf

fig, ax = plt.subplots(figsize=(8, 5))
plottl_zf(
    frequencies_hz=freq_vec,
    rz_grid_m=rz_grid,
    tl_depth_frequency_db=tl_depth_frequency_db,
    ax=ax,
    title="TL(z, f)",
    receiver_range_km=20.0,
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
