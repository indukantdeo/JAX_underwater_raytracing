# Bellhop-Style Gaussian Beam Solver Audit

This repository now contains the minimum pieces needed for a differentiable 2D Gaussian-beam transmission-loss workflow in JAX, but it is not yet at full Bellhop parity.

## Mathematical reference

Primary reference used for the audit and implementation:

- Michael B. Porter, "Beam tracing for two- and three-dimensional problems in ocean acoustics," JASA 146(3), 2019.
- URL: https://www.hlsresearch.com/personnel/porter/papers/JASA/3D%20SI%20Beam%20tracing.pdf

Key 2D formulas used here:

- Ray equations: Eqs. (2)-(5)
- Beam field ansatz: Eqs. (6)-(7)
- Paraxial dynamic ray equations: Eqs. (10)-(14)
- Geometric hat / Gaussian beam widths and amplitudes: Eqs. (15)-(23)

## What is already implemented

### Core ray tracing

- 2D ray tracing in range-depth coordinates using JAX RK2:
  - `src/simulation/ray_tracing.py`
  - `src/simulation/dynamic_ray_tracing.py`
- Differentiable sound-speed and derivative access:
  - `src/simulation/sound_speed.py`
- Differentiable bathymetry / altimetry geometry and normals:
  - `src/simulation/boundary.py`
- Boundary reflections handled in the ray integrator through JAX control flow.

### Dynamic beam state

- Dynamic state includes:
  - position `(r, z)`
  - slowness components `(rho, zeta)`
  - dynamic variables `(q_r, q_i, p_r, p_i)`
  - travel time `tau`
- The implemented ODE in `dynamic_ray_tracing.py` matches the standard 2D dynamic-ray form
  - `dq/ds = c p`
  - `dp/ds = -(c_nn / c^2) q`

### Transmission-loss solver layer added in this update

- Bellhop-style launch fan API:
  - `trace_beam_fan(...)`
- High-level TL solve API:
  - `solve_transmission_loss(...)`
- Differentiable geometric Gaussian beam accumulation:
  - `accumulate_geometric_gaussian_field(...)`
- Fixed issues in existing tracing utilities:
  - `compute_multiple_ray_paths(...)` no longer overwrites `ds`
  - `compute_caustic(...)` now returns the full caustic history instead of only the last value

## What is still missing for Bellhop parity

### 1. Boundary physics is incomplete

Current status:

- Rays reflect specularly from surface and bottom.
- Dynamic `p <- p + N q` jump terms are included.

Missing:

- Surface pressure-release phase handling in the field accumulation.
- Bottom reflection coefficients, phase, and loss models.
- Elastic / complex boundary impedance handling.
- Interface transmission and multi-layer media.

Impact:

- TL fields near repeated boundary interactions will not yet match Bellhop quantitatively.

### 2. Current receiver accumulation is only a baseline Bellhop-style field solve

Current status:

- The new geometric Gaussian field solver uses Porter Sec. II.C width/amplitude scaling.
- A differentiable beam-width floor ("stent") is applied to regularize caustics.

Missing:

- Full Bellhop arrival bookkeeping.
- Bellhop-equivalent semi-coherent and incoherent bundling logic.
- Bellhop's exact beam-window and launch-weight policies.
- Validation against Bellhop `.shd` or arrival outputs.

Impact:

- The current solver is suitable as a differentiable baseline, not yet a drop-in Bellhop replacement.

### 3. Environment model is too limited

Current status:

- Sound speed is effectively depth-only in the default implementation.
- Bathymetry is a simple interpolated profile.

Missing:

- Robust range-dependent SSP support on structured grids.
- Interpolation of oceanographic datasets.
- Piecewise interfaces and sediment layers.
- Attenuation and volume absorption models.

Impact:

- Real-world range-dependent TL studies are not yet supported cleanly.

### 4. Differentiability is present, but not yet packaged for inversion workflows

Current status:

- JAX-compatible tracing and beam accumulation are in place.
- The new TL solve path is differentiable with respect to the inputs passed through the environment functions and source parameters.

Missing:

- Explicit gradient-based examples.
- Loss functions for inversion or design optimization.
- Stable batching for multiple frequencies / sources / environmental parameters.
- Regression tests that check gradients are finite and physically sensible.

Impact:

- The code can be used in differentiable workflows, but that workflow is not yet productized.

### 5. GPU / CPU support is inherited from JAX, but runtime validation is still needed

Current status:

- The implementation is JAX-native and therefore targets CPU or GPU depending on the installed `jaxlib`.

Missing:

- Runtime validation in an environment with `jax` installed.
- Performance profiling on CPU vs GPU.
- Memory/performance optimization for large receiver grids and large beam fans.

Impact:

- The code path is portable, but performance claims are not yet benchmarked in this repository.

## Recommended next implementation steps

1. Validate `solve_transmission_loss(...)` against a small Bellhop benchmark case.
2. Add boundary reflection phase / loss coefficients into the propagated beam amplitude.
3. Add a range-dependent SSP representation with differentiable interpolation.
4. Add regression tests for:
   - ray trajectories
   - caustic handling
   - TL field sanity
   - gradient finiteness
5. Add a plotting example that produces TL fields for varying frequency, source depth, and bathymetry.

## Practical reading of repository status

Implemented now:

- differentiable 2D ray tracer
- differentiable dynamic-ray propagator
- baseline differentiable geometric Gaussian TL solver
- Bellhop-style launch fan and TL API

Not implemented yet:

- full Bellhop boundary physics
- full Bellhop arrival machinery
- robust range-dependent environmental modeling
- validation and benchmark suite
