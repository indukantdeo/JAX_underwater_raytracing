# JAX_underwater_raytracing
This repo contains the code for underwater ray tracing using RK-2 integration for Eikonal equaiton  written in JAX for speed and Auto Diff. functionality . 

## Create a virtual environment
```bash # 
load python 
pip install venv 
python -m venv JAX_underwater_raytracing 
source JAX_underwater_raytracing/bin/activate 
pip install -r requirements.txt 
```

## src
### plot
Plotting module handles all the ray plotting and environment plotting and make sure that the standard plotting color same as Bellhop are used.

### Simulation
Main package for running the numerical integration of the ray eqns implemented in JAX to ensure maximum speed.
#### boundary.py
Handles the altimetry and bathymetry with their tangent and normal
#### sound_speed.py
Defines the sound speed profile and their gradient
#### ray_tracing.py
Define the ray_eqns and solves it using RK-2 Stepper. The function is implementes in JAC with rollout function to ensure maximum speed in calculating ray trajectories. Reflection conditions are handles using lax.cond to ensure compatibility with auto diff.



## Examples

### Example 1
```
Replicates the Dickins seamount ray trace.
```