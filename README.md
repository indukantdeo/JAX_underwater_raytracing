# JAX Underwater Ray Tracing

This repository provides an implementation of underwater ray tracing using second-order Runge–Kutta (RK2) integration for solving the **Eikonal equation**, written in **JAX** to leverage just-in-time (JIT) compilation and automatic differentiation (AutoDiff).

The code is modular and optimized for simulating underwater acoustic ray paths with varying sound speed profiles and ocean boundaries.

---

## Installation

To set up the virtual environment and install dependencies:

```bash
# Load your preferred Python module (if using HPC environment)
load python

# Create a virtual environment
python -m venv JAX_underwater_raytracing
source JAX_underwater_raytracing/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Project Structure

### `src/`

#### `plot/`
- Contains plotting utilities for visualizing:
  - Ray trajectories
  - Environmental features (e.g., bathymetry and altimetry)
- Color schemes follow the Bellhop standard for consistency.

#### `simulation/`

- Core module for ray tracing based on RK2 integration of ray equations in JAX.

##### `boundary.py`
- Models ocean surface (altimetry) and seabed (bathymetry).
- Provides tangent and normal vectors at boundaries for handling reflections.

##### `sound_speed.py`
- Defines the sound speed profile \( c(r, z) \) and its spatial gradients:
  - $\( \frac{\partial c}{\partial r} \), \( \frac{\partial c}{\partial z} \)$, and higher-order derivatives.

##### `ray_tracing.py`
- Implements the ray equations and their numerical solution using RK2.
- Includes a JAX-compatible `rollout` function for efficiently computing ray paths.
- Handles reflection from top/bottom boundaries using `jax.lax.cond` to preserve AutoDiff compatibility.

---

## Examples

### Example 1: Dickens Seamount Ray Trace
Replicates a classic ray tracing scenario around the Dickens Seamount to validate the ray tracing algorithm.

```bash
# Navigate to the examples directory and run the script
python examples/example_1.py
```

---

## Future Features (Planned)
- Gaussian beam tracing extension  
- 3D ray tracing    
- Interfacing with realistic oceanographic datasets (e.g., World Ocean Atlas)

---

## Citing This Work
If you use this codebase in your research, please consider citing the repository.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.