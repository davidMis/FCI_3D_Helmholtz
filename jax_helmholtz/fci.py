"""Fast contour integration setup."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .gmres import gmres
from .operators import jit_helmop
from .polynomial import cheby_poly, exp_poly
from .polynomial import exp_rate
from .setup import HelmholtzOperator


Array = jnp.ndarray


@dataclass(frozen=True)
class FCIParameters:
    npoles: int
    shifts: Array
    weights: Array
    krylov_dim: int
    tol_outer: float
    tol_inner: float
    nblock: int
    method: int
    bet: tuple[float, float, float]
    num: Array
    q: Array
    d: Array


@dataclass(frozen=True)
class FCIResult:
    u: Array
    matvecs: int
    outer_relres: float
    inner_relres: float


def fci_setup(
    npoles: int,
    sep: float,
    asp: float,
    nblock: int,
    method: int,
    op: HelmholtzOperator,
    *,
    krylov_dim: int = 20,
    tol_outer: float = 2e-1,
    tol_inner: float = 2e-1,
) -> FCIParameters:
    """Set up contour shifts/weights and shifted-solver parameters."""

    dmax = op.rho[1]
    real_dtype = op.mass.dtype
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)

    if npoles == 1:
        shifts = jnp.asarray([sep * 1j], dtype=complex_dtype)
        weights = jnp.asarray([1 + 0j], dtype=complex_dtype)
    else:
        phi = jnp.pi / npoles * jnp.arange(1, 2 * npoles, 2)
        dy = dmax * 0.7
        ry = (sep + dy) / jnp.sin(phi[0])
        rx = ry * asp
        dx = rx * jnp.cos(phi[0])

        shifts = (
            rx * jnp.cos(phi)
            + 1j * ry * jnp.sin(phi)
            - (dx + 1j * dy)
        ).astype(complex_dtype)
        weights = (
            (ry * jnp.cos(phi) + 1j * rx * jnp.sin(phi)) / shifts / npoles
        ).astype(complex_dtype)

    if nblock == 1:
        bet = (op.rho[0] / 2 - 1, op.rho[0] / 2, dmax)
    elif nblock == 2:
        bet = (-1, dmax / 2 + (dmax**2 / 4 + op.rho[0]) ** 0.5, dmax)
    else:
        raise ValueError("nblock must be 1 or 2.")

    num = []
    q_values = []
    d_values = []
    for p in range(npoles):
        q, d, rate = exp_rate(complex(shifts[p]), bet, 1, 7)
        scaled_tol = tol_outer * float(jnp.abs(weights[0] / weights[p]) ** 0.5)
        count = int(jnp.ceil(jnp.log(scaled_tol) / jnp.log(rate)))
        num.append(max(count, 1))
        q_values.append(q)
        d_values.append(d)

    return FCIParameters(
        npoles=npoles,
        shifts=shifts,
        weights=weights,
        krylov_dim=krylov_dim,
        tol_outer=tol_outer,
        tol_inner=tol_inner,
        nblock=nblock,
        method=method,
        bet=bet,
        num=jnp.asarray(num, dtype=jnp.int32),
        q=jnp.asarray(q_values, dtype=jnp.int32),
        d=jnp.asarray(d_values, dtype=real_dtype),
    )


def fci_apply(rhs: Array, op: HelmholtzOperator, params: FCIParameters) -> FCIResult:
    """Apply one FCI approximation/correction step to ``rhs``.

    This ports ``Matlab/fcisol.m``. It intentionally keeps Python-level loops for
    the first validation pass; contour-pole parallelism can be added with ``vmap``
    once the numerical behavior is verified.
    """

    rhs = rhs.astype(jnp.result_type(rhs.dtype, jnp.complex64))
    nrm0 = jnp.linalg.norm(rhs)
    n = rhs.shape[0]
    u = jnp.zeros_like(rhs)

    shifted_rhs = rhs
    if params.nblock == 2:
        shifted_rhs = jnp.concatenate([jnp.zeros_like(rhs), rhs])

    nmvp = 0
    outer_relres = float("inf")
    for p in range(params.npoles):
        z = complex(params.shifts[p])
        tol = params.tol_outer * float(jnp.abs(params.weights[0] / params.weights[p]) ** 0.5)

        if params.method == 1:
            v, nmv1, relres = exp_poly(
                shifted_rhs,
                z,
                op,
                tol,
                int(params.num[p]),
                complex(params.bet[0], z.imag),
                float(params.d[p]),
                int(params.q[p]),
            )
        elif params.method == 2:
            v, nmv1, relres = cheby_poly(
                shifted_rhs,
                z,
                op,
                tol,
                int(params.num[p] * params.q[p]),
            )
            w = shifted_rhs - (jit_helmop(v, op) - z * v)
            w_norm = jnp.linalg.norm(w)
            if float(w_norm) > float(nrm0) * tol:
                correction = gmres(
                    lambda x, shift=z: jit_helmop(x, op) - shift * x,
                    w,
                    restart=params.krylov_dim,
                    tol=float(nrm0) * tol / float(w_norm),
                    max_matvecs=int(params.num[p] * params.q[p]),
                )
                v = v + correction.x
                nmv1 += correction.matvecs
                relres = correction.relres * float(w_norm / nrm0)
        else:
            raise ValueError("method must be 1 for exp_poly or 2 for cheby_poly+GMRES.")

        nmvp += nmv1
        outer_relres = min(outer_relres, relres)

        if params.nblock == 2:
            v = v[n : 2 * n]
        u = u + params.weights[p] * v

    matvecs = nmvp
    v = jit_helmop(u, op)
    c = jnp.vdot(v, rhs) / jnp.vdot(v, v)
    u = c * u
    residual = rhs - c * v

    max_inner = max(1, int(jnp.ceil(nmvp * 2 / params.krylov_dim)) * params.krylov_dim)
    inner = gmres(
        lambda x: jit_helmop(x, op),
        residual,
        restart=params.krylov_dim,
        tol=params.tol_inner,
        max_matvecs=max_inner,
    )
    u = u + inner.x
    matvecs += inner.matvecs + 1

    return FCIResult(
        u=u,
        matvecs=matvecs,
        outer_relres=outer_relres,
        inner_relres=inner.relres,
    )
