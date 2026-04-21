"""Polynomial iterations for shifted Helmholtz systems."""

from __future__ import annotations

import jax.numpy as jnp

from .operators import helmop, helmsym
from .setup import HelmholtzOperator


Array = jnp.ndarray


def exp_rate(
    z: complex,
    bet: tuple[float, float, float] | Array,
    qmin: int,
    qmax: int,
) -> tuple[int, float, float]:
    """Search polynomial order, step size, and convergence rate.

    Direct port of ``Matlab/exp_rate.m``.
    """

    bet = jnp.asarray(bet)
    z = complex(z)
    zre = z.real
    zim = z.imag

    b = jnp.asarray([bet[0] - bet[1], bet[0] + bet[1]], dtype=jnp.complex128) - z
    if zim < 0:
        b = b - 1j * bet[2]
    ba = jnp.abs(b)

    md = float(-1 / jnp.imag(jnp.sum(ba) / jnp.sum(ba / b)))
    mrate = float(jnp.abs(b[1] - b[0]) / jnp.sum(ba))
    mq = 1

    nlam = 20
    sample = jnp.arange(1, nlam + 1)
    lam = jnp.concatenate(
        [
            -zim + 1j * bet[1] * sample / nlam,
            -(zim + bet[2]) + 1j * bet[1] * sample / nlam,
            jnp.asarray([-1j * (zre - bet[0])]),
        ]
    )

    d0 = md / 10
    for q in range(qmin, qmax + 1):
        for i in range(1, 10001):
            d = d0 * i
            rate_per_matvec = float(_exp_convrate(d, lam, q) ** (1 / q))
            if rate_per_matvec < mrate:
                mq = q
                md = d
                mrate = rate_per_matvec
            if rate_per_matvec > 1:
                break

    return mq, md, mrate**mq


def exp_poly(
    rhs: Array,
    z: complex,
    op: HelmholtzOperator,
    tol: float,
    niter: int,
    z0: complex,
    d: float,
    q: int,
) -> tuple[Array, int, float]:
    """Solve a shifted system by Taylor polynomial fixed-point iteration."""

    wd = -1j * d
    wdz = wd * (z - z0)
    weights = [1 + 0j]
    for j in range(1, q + 1):
        weights.append(weights[-1] * (wdz / j))
    wv = jnp.asarray(weights, dtype=rhs.dtype)
    wsum = jnp.sum(wv)

    sol = jnp.zeros_like(rhs)
    nrm0 = jnp.linalg.norm(rhs)
    nmv = 0
    relres = float("inf")

    for _ in range(niter):
        k = sol
        for j in range(1, q + 1):
            k = (wd / j) * (helmop(k, op) - z0 * k - wv[j - 1] * rhs)
            nmv += 1
            if j == 1:
                relres = float(jnp.linalg.norm(k - wdz * sol) / nrm0 / abs(wd))
                if relres < tol:
                    break
            sol = sol + k
        if relres < tol:
            break
        sol = sol / wsum

    return sol, nmv, relres


def cheby_poly(
    rhs: Array,
    z: complex,
    op: HelmholtzOperator,
    tol: float,
    niter: int,
) -> tuple[Array, int, float]:
    """Chebyshev iteration for the shifted symmetric operator."""

    if rhs.shape[0] == op.size:
        c = op.rho[0] / 2
        a = c - (1 + z)
    else:
        c = op.rho[0] ** 0.5
        a = -(1 + z)

    r = rhs
    sol = jnp.zeros_like(rhs)
    dsol = jnp.zeros_like(rhs)

    nrm0 = jnp.linalg.norm(r)
    relres = 1.0
    nmv = 0
    beta = 0 + 0j

    while relres >= tol and nmv < niter:
        gamma = -(a + beta)
        dsol = (-r + beta * dsol) / gamma
        sol = sol + dsol
        r = rhs - (helmsym(sol, op) - z * sol)
        relres = float(jnp.linalg.norm(r) / nrm0)
        nmv += 1
        if nmv < 2:
            beta = -(c * c) / (2 * a)
        else:
            beta = (c / 2) ** 2 / gamma

    return sol, nmv, relres


def ric_rate(z: complex, bet: tuple[float, float, float] | Array) -> tuple[complex, float]:
    """Optimal stationary Richardson parameter for rectangular spectrum."""

    bet = jnp.asarray(bet)
    b = jnp.asarray([bet[0] - bet[1], bet[0] + bet[1]], dtype=jnp.complex128) - z
    if complex(z).imag < 0:
        b = b - 1j * bet[2]

    ba = jnp.abs(b)
    d = jnp.sum(ba / b) / jnp.sum(ba)
    rate = jnp.abs(b[1] - b[0]) / jnp.sum(ba)
    return complex(d), float(rate)


def ric_poly(
    rhs: Array,
    z: complex,
    op: HelmholtzOperator,
    tol: float,
    niter: int,
    d: complex,
) -> tuple[Array, int, float]:
    """Stationary Richardson iteration for shifted systems."""

    res = rhs
    sol = jnp.zeros_like(rhs)
    nrm0 = jnp.linalg.norm(rhs)
    relres = 1.0
    nmv = 0

    for it in range(1, niter + 1):
        sol = sol + d * res
        res = rhs - helmop(sol, op) + z * sol
        relres = float(jnp.linalg.norm(res) / nrm0)
        nmv = it
        if relres < tol:
            break

    return sol, nmv, relres


def _exp_convrate(d: float, z: Array, q: int) -> Array:
    c = jnp.ones_like(z)
    k = jnp.ones_like(z)
    for j in range(1, q + 1):
        k = k * z * (d / j)
        c = c + k
    return jnp.max(jnp.abs(c[:-1]) / jnp.abs(c[-1]))
