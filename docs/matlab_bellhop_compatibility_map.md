# MATLAB Bellhop Compatibility Map

This note maps the local Acoustics Toolbox MATLAB Bellhop implementation onto the current JAX solver and identifies the steps needed for physics-compatible behavior.

Reference files examined:

- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/bellhopM.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/trace.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/step.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/reflect.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/InfluenceGeoHat.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/InfluenceGeoGaussian.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/ssp.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/reducestep.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/scalep.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/topbot.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/crci.m`
- `/home/indu/MELO/OALIB_AcousticToolBox/at/Matlab/Bellhop/makeshdarr.m`

## Bellhop MATLAB execution structure

`bellhopM.m` organizes the computation in this order:

1. Read environment, boundary options, reflection coefficient files, and source beam pattern.
2. Build the SSP interpolation model:
   - `N`: `n^2` linear
   - `C`: `c` linear
   - `P`: monotone PCHIP
   - `S`: cubic spline
3. Read top/bottom geometry and precompute segment normals, tangents, and curvature.
4. Trace each beam with `trace.m`.
5. Accumulate beam influence with either:
   - `InfluenceGeoHat.m`
   - `InfluenceGeoGaussian.m`
6. Apply final pressure scaling with `scalep.m`.

That separation matters. Bellhop does not treat tracing, boundary physics, beam accumulation, and output normalization as one undifferentiated routine.

## What Bellhop does that the current JAX solver still needs

### 1. SSP interpolation parity

`ssp.m` shows that Bellhop’s ray equations depend on the selected interpolation model, not just on a generic `c(z)` evaluation.

Key details:

- `C` linear gives piecewise-constant `c_z` and zero `c_zz`.
- `P` and `S` use piecewise polynomials and evaluate both first and second derivatives from the polynomial object.
- `N` linear operates in slowness-squared space, which changes both `c` and derivative formulas.

Implication for JAX:

- A single autodiff wrapper around `jnp.interp` is not Bellhop-compatible.
- The JAX solver needs explicit interpolation backends that reproduce Bellhop’s derivative structure.

Minimum implementation sequence:

1. Add Bellhop-style `c_linear` SSP operators.
2. Add Bellhop-style cubic spline / PCHIP operators.
3. Use the selected SSP operator set in both tracing and dynamic-ray equations.

### 2. Adaptive step control

`step.m` and `reducestep.m` are central.

Bellhop does not march rays using a fixed `ds` blindly. It reduces the step to land on:

- SSP interfaces
- top/bottom crossings
- top/bottom segment-range changes

Implication for JAX:

- Fixed-step RK2 with post hoc reflection is not Bellhop-compatible.
- The current JAX integrator should be upgraded to Bellhop-style step reduction before each advance.

Minimum implementation sequence:

1. Add a JAX `reduce_step(...)` that mirrors `reducestep.m`.
2. Update the RK2 stepper to use the reduced step in both the midpoint and final stage.
3. Carry segment IDs for top and bottom geometry if range-dependent boundaries are active.

### 3. Boundary geometry handling

`trace.m` updates the active top and bottom segments as the ray moves, and reflections use segment-local:

- tangent
- normal
- curvature

For curvilinear boundaries, Bellhop interpolates node-based tangent and normal data within the segment.

Implication for JAX:

- The current JAX boundary model is adequate for simple piecewise-linear cases, but it does not yet mirror Bellhop’s segment bookkeeping and curvature handling.

Minimum implementation sequence:

1. Add explicit segment indexing for bathymetry/altimetry.
2. Compute piecewise-linear segment normals/tangents/curvature Bellhop-style.
3. Interpolate boundary frame data within a segment for curvilinear cases.

### 4. Reflection physics

`reflect.m` applies three different things at a boundary:

1. specular change of ray tangent
2. curvature jump to `p`
3. amplitude/phase update through `Rfa`

Boundary conditions in MATLAB Bellhop:

- `V`: vacuum, multiply field by `-1`
- `R`: rigid, no phase flip
- `A`: acoustic half-space reflection coefficient
- `F`: tabulated reflection coefficient

Important detail:

- Bellhop stores reflection amplitude/phase in `ray(is).Rfa` during tracing, not later during field accumulation.

Implication for JAX:

- Boundary reflection factors should be applied during the trace, and the accumulated field should consume that traced beam factor directly.
- The current JAX changes move in that direction, but they still need cleanup and validation.

Minimum implementation sequence:

1. Keep cumulative beam factor in the ray state as a complex multiplier.
2. Use Bellhop’s half-space reflection coefficient model consistently.
3. Add tabulated reflection coefficient support later if needed.

### 5. Beam representation parity

This is one of the largest remaining mismatches.

`bellhopM.m` selects:

- `InfluenceGeoHat.m` for geometric hat beams
- `InfluenceGeoGaussian.m` for geometric Gaussian beams

The Munk case you ran is a hat-beam case.
The Dickins case you ran is a Gaussian-beam case.

The formulas differ materially:

- Hat beam:
  - support radius `RadMax = |q / q0|`
  - shape `RadMax - n`
  - weighting `A = 1 / RadMax`
- Gaussian beam:
  - support width `sigma = |q / q0|`
  - shape `exp( -0.5 * (n / sigma)^2 ) / (sigma * A)`
  - stenting rule for minimum width

Implication for JAX:

- A Gaussian-only accumulation path cannot match the Bellhop hat-beam Munk case.

Minimum implementation sequence:

1. Add a JAX `InfluenceGeoHat(...)` path.
2. Make validation choose the beam representation that matches the Bellhop test case.
3. Only compare Munk after the hat path is implemented.

### 6. Final field scaling

`scalep.m` applies the output normalization after all beams have been accumulated:

- coherent / ray-centered: `const = -Dalpha * sqrt(freq) / c`
- cylindrical spreading factor: `1 / sqrt(r)`
- line-source option has a different factor
- incoherent fields are square-rooted before the final scaling

Implication for JAX:

- Bellhop’s final field is not just the raw sum of beam contributions.
- The current JAX solver still needs Bellhop-style post-accumulation scaling parity.

Minimum implementation sequence:

1. Move source/range scaling into a Bellhop-style `scale_pressure(...)`.
2. Use source depth reference sound speed `c(src)` consistently.
3. Apply the coherent/incoherent distinction exactly as in `scalep.m`.

### 7. Auto beam count and launch fan policy

`bellhopM.m` computes:

- `DalphaOpt = sqrt( c / ( 6 * freq * Rmax ) )`
- `NbeamsOpt = round( 2 + (alpha_max - alpha_min) / DalphaOpt )`

That is one reason Bellhop used many more beams than the reduced JAX validation run.

Implication for JAX:

- Matching Bellhop fidelity requires either using Bellhop’s beam-count heuristic or explicitly matching the beam list from the reference case.

### 8. Arrival/shade assembly separation

`makeshdarr.m` shows that Bellhop can reconstruct coherent pressure from arrivals:

- amplitude
- delay
- launch/arrival angle bookkeeping

Implication for JAX:

- A future Bellhop-compatible JAX solver should expose both:
  - TL field accumulation
  - arrival-level outputs

This is not the first priority for matching `.shd`, but it is part of full parity.

## Direct mapping to the current JAX repository

Current JAX strengths:

- differentiable 2D ray equations
- dynamic-ray propagation
- basic boundary handling
- Bellhop-inspired field accumulation

Current Bellhop-compatibility gaps:

1. no Bellhop-style adaptive step reduction
2. no hat-beam influence path
3. no exact `scalep.m`-style final normalization
4. SSP interpolation is not Bellhop-faithful for `C`, `P`, `S`, and `N` modes
5. segment-aware boundary geometry is incomplete
6. beam-count / launch-spacing policy is not Bellhop-matched

## Recommended implementation order

This is the lowest-risk order for upgrading the JAX solver toward Bellhop fidelity:

1. Implement Bellhop-style final pressure scaling from `scalep.m`.
2. Add JAX `InfluenceGeoHat(...)` and use it for the Munk validation case.
3. Add Bellhop-style adaptive step reduction from `reducestep.m`.
4. Refactor SSP evaluation into Bellhop interpolation modes:
   - `C`
   - `S`
   - `P`
   - later `N`
5. Refine boundary geometry bookkeeping to mirror `trace.m`.
6. Increase beam count using Bellhop’s launch-spacing heuristic.

## Practical conclusion

The current JAX solver is no longer just a generic differentiable ray tracer, but it is still not Bellhop-compatible because Bellhop fidelity depends on a coordinated stack:

- interpolation model
- step reduction
- boundary reflection factor traced through the beam
- correct beam representation
- final field scaling

The next highest-value implementation is:

1. `InfluenceGeoHat(...)`
2. `scalep.m` parity
3. `reducestep.m` parity

Without those three, the JAX solver will continue to miss Bellhop even if the reflection coefficient formulas are improved further.
