import unittest

import jax.numpy as jnp

from src.simulation import boundary as boundary_mod
from src.simulation import dynamic_ray_tracing as dyn_mod
from src.simulation import sound_speed as ssp_mod


class BellhopSpeedupPathTest(unittest.TestCase):
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

    def test_dense_and_windowed_backends_match_on_reduced_case(self):
        rr_grid = jnp.linspace(0.0, 8000.0, 21, dtype=jnp.float64)
        rz_grid = jnp.linspace(800.0, 1800.0, 17, dtype=jnp.float64)
        common_kwargs = dict(
            freq=50.0,
            r_s=0.0,
            z_s=1200.0,
            theta_min=-0.18,
            theta_max=0.18,
            n_beams=7,
            rr_grid=rr_grid,
            rz_grid=rz_grid,
            ds=50.0,
            beam_type="geometric",
            run_mode="coherent",
            accumulation_model="gaussian",
            store_field_per_beam=False,
            store_trajectories=False,
        )

        dense = dyn_mod.solve_transmission_loss(accumulation_backend="dense", **common_kwargs)
        windowed = dyn_mod.solve_transmission_loss(accumulation_backend="windowed", beam_chunk_size=3, **common_kwargs)

        self.assertEqual(dense["tl_db"].shape, windowed["tl_db"].shape)
        self.assertTrue(jnp.allclose(dense["field_total"], windowed["field_total"], atol=1e-8, rtol=1e-6))
        self.assertTrue(jnp.allclose(dense["tl_db"], windowed["tl_db"], atol=1e-8, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
