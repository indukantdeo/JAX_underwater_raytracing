import unittest

import jax.numpy as jnp
from jax.test_util import check_grads

from src.simulation import boundary as boundary_mod
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation import sound_speed as ssp_mod


class DifferentiableSolverTest(unittest.TestCase):
    def setUp(self):
        dyn_mod.configure_acoustic_operators(
            sound_speed_operators=ssp_mod.MUNK_OPERATORS,
            boundary_operators=boundary_mod.FLAT_BOUNDARY_OPERATORS,
            reflection_model={
                "top_boundary_condition": "vacuum",
                "bottom_boundary_condition": "rigid",
                "kill_backward_rays": False,
            },
        )

    def test_autodiff_gradient_matches_finite_difference(self):
        rr_grid = jnp.linspace(500.0, 4000.0, 12, dtype=jnp.float64)
        rz_grid = jnp.linspace(900.0, 1700.0, 10, dtype=jnp.float64)

        def loss_fn(params):
            freq_hz, source_depth_m = params
            result = dyn_mod.solve_transmission_loss_autodiff(
                freq_hz,
                0.0,
                source_depth_m,
                -0.18,
                0.18,
                7,
                rr_grid,
                rz_grid,
                ds=25.0,
                beam_type="geometric",
                run_mode="coherent",
                auto_beam_count=False,
                min_width_wavelengths=0.75,
                range_window_softness_m=40.0,
            )
            field = result["field_total"]
            return jnp.mean(jnp.real(field * jnp.conj(field)))

        x0 = jnp.array([50.0, 1200.0], dtype=jnp.float64)
        check_grads(loss_fn, (x0,), order=1, modes=("fwd", "rev"), atol=5e-3, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
