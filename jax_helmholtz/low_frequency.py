"""GMRES-based spectral solver for lower-frequency Helmholtz tests."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

from .fast_spectral import helmop_spectral_grid
from .operators import flatten_grid, unflatten_grid
from .setup import HelmholtzOperator


Array = jnp.ndarray


@dataclass(frozen=True)
class GMRESSolveResult:
    u: Array
    residual_history: Array
    cycles: int
    matvecs_estimate: int
    info: int | Array


def solve_gmres_spectral(
    rhs: Array,
    op: HelmholtzOperator,
    *,
    tol: float = 1e-3,
    restart: int = 40,
    max_cycles: int = 50,
    verbose: bool = False,
) -> GMRESSolveResult:
    """Solve ``A u = rhs`` by restarted GMRES on the spectral operator.

    This is intended as a low-frequency baseline. It runs one restart cycle at a
    time so we can record a residual history comparable to the FCI refinement logs.
    """

    if op.mode != "spectral":
        raise ValueError("solve_gmres_spectral currently requires spectral mode.")
    if op.stiffness_eigs is None:
        raise ValueError("spectral mode requires stiffness_eigs.")

    rhs = rhs.astype(jnp.result_type(rhs.dtype, jnp.complex64))
    rhs_grid = unflatten_grid(rhs, op.n)
    rhs_norm = jnp.linalg.norm(rhs_grid)
    u_grid = jnp.zeros_like(rhs_grid)
    residual_history = []
    info = 0

    matvec = lambda x: _helmop_grid_jit(x, op.mass, op.damping, op.stiffness_eigs)

    for cycle in range(1, max_cycles + 1):
        u_grid, info = jax_gmres(
            matvec,
            rhs_grid,
            x0=u_grid,
            tol=tol,
            restart=restart,
            maxiter=1,
            solve_method="batched",
        )
        residual = rhs_grid - matvec(u_grid)
        relres = jnp.linalg.norm(residual) / rhs_norm
        residual_history.append(relres)
        if verbose:
            print(
                "gmres "
                f"cycle={cycle} relres={float(relres):.6e} "
                f"estimated_matvecs={cycle * restart}",
                flush=True,
            )
        if float(relres) < tol:
            break

    history = jnp.asarray(residual_history)
    return GMRESSolveResult(
        u=flatten_grid(u_grid),
        residual_history=history,
        cycles=len(residual_history),
        matvecs_estimate=len(residual_history) * restart,
        info=info,
    )


_helmop_grid_jit = jax.jit(helmop_spectral_grid)
