"""JAX implementation of the FCI 3D Helmholtz solver."""

from .fci import FCIParameters, FCIResult, fci_apply, fci_setup
from .gmres import GMRESResult, gmres
from .operators import flatten_grid, helmop, stiffop, unflatten_grid
from .polynomial import cheby_poly, exp_poly, exp_rate, ric_poly, ric_rate
from .random_fields import gaussian_random_field
from .setup import HelmholtzOperator, mat_setup, mat_setup_from_wavespeed

__all__ = [
    "FCIParameters",
    "FCIResult",
    "GMRESResult",
    "HelmholtzOperator",
    "cheby_poly",
    "exp_poly",
    "exp_rate",
    "fci_apply",
    "fci_setup",
    "flatten_grid",
    "gaussian_random_field",
    "gmres",
    "helmop",
    "mat_setup",
    "mat_setup_from_wavespeed",
    "ric_poly",
    "ric_rate",
    "stiffop",
    "unflatten_grid",
]
