"""JAX implementation of the FCI 3D Helmholtz solver."""

from .fci import FCIParameters, FCIResult, fci_apply, fci_setup
from .fast_spectral import FastFCIResult, fci_apply_spectral_jit
from .gmres import GMRESResult, gmres
from .operators import flatten_grid, helmop, jit_helmop, jit_stiffop, stiffop, unflatten_grid
from .polynomial import cheby_poly, exp_poly, exp_rate, ric_poly, ric_rate
from .random_fields import gaussian_random_field
from .setup import HelmholtzOperator, mat_setup, mat_setup_from_wavespeed

__all__ = [
    "FCIParameters",
    "FCIResult",
    "FastFCIResult",
    "GMRESResult",
    "HelmholtzOperator",
    "cheby_poly",
    "exp_poly",
    "exp_rate",
    "fci_apply",
    "fci_apply_spectral_jit",
    "fci_setup",
    "flatten_grid",
    "gaussian_random_field",
    "gmres",
    "helmop",
    "jit_helmop",
    "jit_stiffop",
    "mat_setup",
    "mat_setup_from_wavespeed",
    "ric_poly",
    "ric_rate",
    "stiffop",
    "unflatten_grid",
]
