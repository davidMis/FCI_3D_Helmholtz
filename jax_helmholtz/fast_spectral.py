"""GPU-oriented spectral FCI kernels.

These routines target the spectral, one-block, exponential-polynomial path used by
the current random-field examples. They keep vectors as 3D grids and compile the
polynomial shifted solve with XLA, reducing Python dispatch overhead on GPUs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres as jax_gmres

from .fci import FCIParameters, FCIProfile, _block_until_ready
from .operators import flatten_grid, unflatten_grid
from .setup import HelmholtzOperator


Array = jnp.ndarray


@dataclass(frozen=True)
class FastFCIResult:
    u: Array
    residual: Array
    relres: Array
    matvecs_estimate: int
    inner_info: int | Array
    profile: FCIProfile | None = None


@dataclass(frozen=True)
class ShiftedSolveSample:
    """One shifted-solve training pair from a spectral FCI pole."""

    pole_index: int
    rhs: Array
    solution: Array
    shifted_residual: Array
    shift: Array
    weight: Array
    z0: Array
    d: Array
    q: int
    niter: int


@dataclass(frozen=True)
class ShiftedSolveRequest:
    """Inputs for a custom shifted-solve backend."""

    pole_index: int
    rhs: Array
    shift: Array
    weight: Array
    z0: Array
    d: Array
    q: int
    niter: int


def fci_apply_spectral_jit(
    rhs: Array,
    op: HelmholtzOperator,
    params: FCIParameters,
    *,
    inner_solver: str = "gmres",
    inner_steps: int = 5,
    inner_alpha: float | None = None,
    profile: bool = False,
    shifted_sample_callback: Callable[[ShiftedSolveSample], None] | None = None,
    shifted_solver_callback: Callable[[ShiftedSolveRequest], Array] | None = None,
    shifted_solver_cleanup_steps: int = 0,
) -> FastFCIResult:
    """Apply one FCI step using JIT-compiled spectral kernels.

    Supported configuration:
    - ``op.mode == "spectral"``
    - ``params.nblock == 1``
    - ``params.method == 1``
    """

    if op.mode != "spectral":
        raise ValueError("fci_apply_spectral_jit currently requires spectral mode.")
    if params.nblock != 1 or params.method != 1:
        raise ValueError("fci_apply_spectral_jit supports only nblock=1, method=1.")
    if op.stiffness_eigs is None:
        raise ValueError("spectral mode requires stiffness_eigs.")
    if inner_solver not in {"gmres", "none", "richardson", "chebyshev"}:
        raise ValueError(
            "inner_solver must be 'gmres', 'none', 'richardson', or 'chebyshev'."
        )

    if shifted_solver_cleanup_steps < 0:
        raise ValueError("shifted_solver_cleanup_steps must be non-negative.")

    total_start = time.perf_counter() if profile else 0.0
    shifted_solve_seconds: list[float] = []
    contour_combine_seconds = 0.0
    postprocess_seconds = 0.0
    inner_seconds = 0.0
    sample_seconds = 0.0

    rhs = rhs.astype(jnp.result_type(rhs.dtype, jnp.complex64))
    rhs_grid = unflatten_grid(rhs, op.n)
    u_grid = jnp.zeros_like(rhs_grid)
    matvecs = 0

    for p in range(params.npoles):
        z = params.shifts[p].astype(rhs_grid.dtype)
        z0 = jnp.asarray(complex(params.bet[0], complex(params.shifts[p]).imag), dtype=rhs_grid.dtype)
        pole_start = time.perf_counter() if profile else 0.0
        q = int(params.q[p])
        niter = int(params.num[p])
        d = params.d[p].astype(op.mass.dtype)
        if shifted_solver_callback is None:
            v_grid = _exp_poly_grid_jit(
                rhs_grid,
                z,
                op.mass,
                op.damping,
                op.stiffness_eigs,
                z0,
                d,
                q,
                niter,
            )
            matvecs += q * niter
        else:
            v_grid = shifted_solver_callback(
                ShiftedSolveRequest(
                    pole_index=p,
                    rhs=rhs_grid,
                    shift=z,
                    weight=params.weights[p].astype(rhs_grid.dtype),
                    z0=z0,
                    d=d,
                    q=q,
                    niter=niter,
                )
            )
            if shifted_solver_cleanup_steps > 0:
                shifted_residual_rhs = rhs_grid - (
                    _helmop_grid_jit(v_grid, op.mass, op.damping, op.stiffness_eigs)
                    - z * v_grid
                )
                correction_grid = _exp_poly_grid_jit(
                    shifted_residual_rhs,
                    z,
                    op.mass,
                    op.damping,
                    op.stiffness_eigs,
                    z0,
                    d,
                    q,
                    shifted_solver_cleanup_steps,
                )
                v_grid = v_grid + correction_grid
                matvecs += q * shifted_solver_cleanup_steps
        if profile:
            _block_until_ready(v_grid)
            shifted_solve_seconds.append(time.perf_counter() - pole_start)

        if shifted_sample_callback is not None:
            sample_start = time.perf_counter() if profile else 0.0
            shifted_residual_grid = (
                _helmop_grid_jit(v_grid, op.mass, op.damping, op.stiffness_eigs)
                - z * v_grid
                - rhs_grid
            )
            _block_until_ready(shifted_residual_grid)
            shifted_sample_callback(
                ShiftedSolveSample(
                    pole_index=p,
                    rhs=rhs_grid,
                    solution=v_grid,
                    shifted_residual=shifted_residual_grid,
                    shift=z,
                    weight=params.weights[p].astype(rhs_grid.dtype),
                    z0=z0,
                    d=d,
                    q=q,
                    niter=niter,
                )
            )
            if profile:
                sample_seconds += time.perf_counter() - sample_start

        combine_start = time.perf_counter() if profile else 0.0
        u_grid = u_grid + params.weights[p].astype(rhs_grid.dtype) * v_grid
        if profile:
            _block_until_ready(u_grid)
            contour_combine_seconds += time.perf_counter() - combine_start

    postprocess_start = time.perf_counter() if profile else 0.0
    au_grid = _helmop_grid_jit(u_grid, op.mass, op.damping, op.stiffness_eigs)
    c = jnp.vdot(au_grid, rhs_grid) / jnp.vdot(au_grid, au_grid)
    u_grid = c * u_grid
    residual_grid = rhs_grid - c * au_grid
    if profile:
        _block_until_ready((u_grid, residual_grid))
        postprocess_seconds = time.perf_counter() - postprocess_start

    inner_start = time.perf_counter() if profile else 0.0
    if inner_solver == "gmres":
        restart_cycles = max(1, math.ceil(max(matvecs, 1) * 2 / params.krylov_dim))
        correction_grid, info = jax_gmres(
            lambda x: _helmop_grid_jit(x, op.mass, op.damping, op.stiffness_eigs),
            residual_grid,
            tol=params.tol_inner,
            restart=params.krylov_dim,
            maxiter=restart_cycles,
            solve_method="batched",
        )
        if profile:
            _block_until_ready(correction_grid)
        u_grid = u_grid + correction_grid
        matvecs_estimate = matvecs + restart_cycles * params.krylov_dim + 1
    elif inner_solver == "richardson":
        alpha = (
            _auto_richardson_alpha(residual_grid, op.mass, op.damping, op.stiffness_eigs)
            if inner_alpha is None
            else jnp.asarray(inner_alpha, dtype=op.mass.dtype)
        )
        u_grid, residual_grid = _richardson_inner_jit(
            u_grid,
            residual_grid,
            op.mass,
            op.damping,
            op.stiffness_eigs,
            alpha,
            inner_steps,
        )
        if profile:
            _block_until_ready((u_grid, residual_grid))
        info = 0
        matvecs_estimate = matvecs + inner_steps + 1
    elif inner_solver == "chebyshev":
        u_grid, residual_grid = _chebyshev_inner_jit(
            u_grid,
            residual_grid,
            op.mass,
            op.damping,
            op.stiffness_eigs,
            jnp.asarray(op.rho[0], dtype=op.mass.dtype),
            inner_steps,
        )
        if profile:
            _block_until_ready((u_grid, residual_grid))
        info = 0
        matvecs_estimate = matvecs + inner_steps + 1
    else:
        info = 0
        matvecs_estimate = matvecs + 1
    if profile:
        _block_until_ready(u_grid)
        inner_seconds = time.perf_counter() - inner_start

    final_residual = flatten_grid(residual_grid)
    relres = jnp.linalg.norm(residual_grid) / jnp.linalg.norm(rhs_grid)
    if profile:
        _block_until_ready((final_residual, relres))
    timing_profile = None
    if profile:
        timing_profile = FCIProfile(
            total_seconds=time.perf_counter() - total_start,
            shifted_solve_seconds=tuple(shifted_solve_seconds),
            contour_combine_seconds=contour_combine_seconds,
            postprocess_seconds=postprocess_seconds,
            inner_seconds=inner_seconds,
            sample_seconds=sample_seconds,
        )
    return FastFCIResult(
        u=flatten_grid(u_grid),
        residual=final_residual,
        relres=relres,
        matvecs_estimate=matvecs_estimate,
        inner_info=info,
        profile=timing_profile,
    )


def helmop_spectral_grid(
    x: Array,
    mass: Array,
    damping: Array,
    stiffness_eigs: Array,
) -> Array:
    return jnp.fft.ifftn(stiffness_eigs * jnp.fft.fftn(x)) - (mass + 1j * damping) * x


def exp_poly_grid(
    rhs: Array,
    z: Array,
    mass: Array,
    damping: Array,
    stiffness_eigs: Array,
    z0: Array,
    d: Array,
    q: int,
    niter: int,
) -> Array:
    wd = -1j * d
    wdz = wd * (z - z0)

    weights = [jnp.asarray(1 + 0j, dtype=rhs.dtype)]
    for j in range(1, q + 1):
        weights.append(weights[-1] * (wdz / j))
    wv = jnp.stack(weights)
    wsum = jnp.sum(wv)

    def outer_body(_, sol):
        k0 = sol

        def inner_body(j, state):
            sol_inner, k = state
            jf = jnp.asarray(j, dtype=rhs.real.dtype)
            k = (wd / jf) * (
                helmop_spectral_grid(k, mass, damping, stiffness_eigs)
                - z0 * k
                - wv[j - 1] * rhs
            )
            return sol_inner + k, k

        sol_next, _ = jax.lax.fori_loop(1, q + 1, inner_body, (sol, k0))
        return sol_next / wsum

    return jax.lax.fori_loop(0, niter, outer_body, jnp.zeros_like(rhs))


def richardson_inner(
    u: Array,
    residual: Array,
    mass: Array,
    damping: Array,
    stiffness_eigs: Array,
    alpha: Array,
    steps: int,
) -> tuple[Array, Array]:
    """Fixed-memory Richardson correction on ``A u = rhs`` residual."""

    def body(_, state):
        u_curr, r_curr = state
        u_next = u_curr + alpha * r_curr
        r_next = r_curr - alpha * helmop_spectral_grid(
            r_curr,
            mass,
            damping,
            stiffness_eigs,
        )
        return u_next, r_next

    return jax.lax.fori_loop(0, steps, body, (u, residual))


def auto_richardson_alpha(
    residual: Array,
    mass: Array,
    damping: Array,
    stiffness_eigs: Array,
) -> Array:
    """Residual-line minimum step for one Richardson correction."""

    ar = helmop_spectral_grid(residual, mass, damping, stiffness_eigs)
    return jnp.vdot(ar, residual) / jnp.vdot(ar, ar)


def chebyshev_inner(
    u: Array,
    residual: Array,
    mass: Array,
    damping: Array,
    stiffness_eigs: Array,
    rho_hermitian: Array,
    steps: int,
) -> tuple[Array, Array]:
    """Fixed-memory Chebyshev correction using the symmetric operator model.

    This mirrors the one-block recurrence in ``cheby_poly`` with ``z=0`` but uses
    the full Helmholtz residual update for the corrected solution.
    """

    c = rho_hermitian / 2
    a = c - 1
    beta0 = jnp.asarray(0 + 0j, dtype=u.dtype)
    first_beta = beta0 + (-(c * c) / (2 * a))
    beta_scale = beta0 + ((c / 2) ** 2)
    dsol0 = jnp.zeros_like(u)

    def body(k, state):
        u_curr, r_curr, dsol, beta = state
        gamma = -(a + beta)
        dsol_next = (-r_curr + beta * dsol) / gamma
        u_next = u_curr + dsol_next
        r_next = r_curr - helmop_spectral_grid(
            dsol_next,
            mass,
            damping,
            stiffness_eigs,
        )
        beta_next = jax.lax.cond(
            k < 1,
            lambda _: first_beta,
            lambda _: beta_scale / gamma,
            operand=None,
        )
        return u_next, r_next, dsol_next, beta_next

    u_out, r_out, _, _ = jax.lax.fori_loop(
        0,
        steps,
        body,
        (u, residual, dsol0, beta0),
    )
    return u_out, r_out


_helmop_grid_jit = jax.jit(helmop_spectral_grid)
_exp_poly_grid_jit = jax.jit(exp_poly_grid, static_argnames=("q", "niter"))
_richardson_inner_jit = jax.jit(richardson_inner, static_argnames=("steps",))
_auto_richardson_alpha = jax.jit(auto_richardson_alpha)
_chebyshev_inner_jit = jax.jit(chebyshev_inner, static_argnames=("steps",))
