# import jax.numpy as jnp
# import pytest
#
# from src.fdm import (
#     SecondOrderECirreFDMSolverBackwardImplicit,
#     SecondOrderECirreFDMSolverExplicit,
#     SecondOrderECirreFDMSolverNewton,
# )
# from src.params import SecondOrderECirreMechanismFDMParams
# from src.voltammetry import CyclicDC
#
#
# @pytest.fixture(scope="module")
# def ec_irre_reaction():
#     h = 1e-3
#     dtheta = 5e-2
#     voltammetry = CyclicDC()
#
#     newton_solver = SecondOrderECirreFDMSolverNewton(voltammetry, h=h, dtheta=dtheta)
#
#     backward_solver = SecondOrderECirreFDMSolverBackwardImplicit(
#         voltammetry, h=h, dtheta=dtheta
#     )
#
#     explicit_solver = SecondOrderECirreFDMSolverExplicit(
#         voltammetry, h=h, dtheta=dtheta
#     )
#     return dict(
#         newton_solver=newton_solver,
#         backward_solver=backward_solver,
#         explicit_solver=explicit_solver,
#     )
#
#
# def test_agreement_large_Kplus(ec_irre_reaction):
#     backward_solver = ec_irre_reaction["backward_solver"]
#     newton_solver = ec_irre_reaction["newton_solver"]
#
#     params = SecondOrderECirreMechanismFDMParams(
#         alpha=jnp.array(1.0),
#         K0=jnp.array(10000.0),
#         Kplus=jnp.array(1000000.0),
#         Kminus=jnp.array(10.0),
#         dB=jnp.array(1.0),
#         dY=jnp.array(1.0),
#         dZ=jnp.array(1.0),
#         E0=jnp.array(0.0),
#     )
#
#     current_backward = backward_solver.solve(params)
#     current_newton = newton_solver.solve(params)
#
#     mse = jnp.mean(jnp.square(current_backward - current_newton))
#
#     assert mse < 1e-4
#
#
# def test_agreement_small_Kplus(ec_irre_reaction):
#     newton_solver = ec_irre_reaction["newton_solver"]
#     explicit_solver = ec_irre_reaction["explicit_solver"]
#
#     params = SecondOrderECirreMechanismFDMParams(
#         alpha=jnp.array(1.0),
#         K0=jnp.array(100.0),
#         Kplus=jnp.array(100.0),
#         Kminus=jnp.array(10.0),
#         dB=jnp.array(1.0),
#         dY=jnp.array(1.0),
#         dZ=jnp.array(1.0),
#         E0=jnp.array(0.0),
#     )
#
#     current_newton = newton_solver.solve(params)
#     current_explicit = explicit_solver.solve(params)
#
#     mse = jnp.mean(jnp.square(current_explicit - current_newton))
#
#     assert mse < 1e-4
