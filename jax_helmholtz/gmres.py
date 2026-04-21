"""Small restarted GMRES implementation for matrix-free operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(frozen=True)
class GMRESResult:
    x: Array
    relres: float
    matvecs: int


def gmres(
    matvec: Callable[[Array], Array],
    rhs: Array,
    *,
    restart: int = 20,
    tol: float = 1e-6,
    max_matvecs: int = 100,
    x0: Array | None = None,
) -> GMRESResult:
    """Restarted GMRES for small Krylov dimensions.

    This favors clarity and Matlab parity over full JIT compilation. The restart
    dimension in the original solver is small, so dense least-squares solves are
    acceptable here.
    """

    if restart < 1:
        raise ValueError("restart must be positive.")

    rhs_norm = jnp.linalg.norm(rhs)
    if float(rhs_norm) == 0.0:
        x = jnp.zeros_like(rhs) if x0 is None else x0
        return GMRESResult(x=x, relres=0.0, matvecs=0)

    if x0 is None:
        x = jnp.zeros_like(rhs)
        r = rhs
        matvecs = 0
    else:
        x = x0
        r = rhs - matvec(x)
        matvecs = 1
    beta = jnp.linalg.norm(r)
    relres = float(beta / rhs_norm)

    while relres > tol and matvecs < max_matvecs:
        m = min(restart, max_matvecs - matvecs)
        basis = []
        h_cols = []
        basis.append(r / beta)

        best_dx = jnp.zeros_like(rhs)
        best_relres = relres

        for j in range(m):
            w = matvec(basis[j])
            matvecs += 1

            h_col = []
            for i in range(j + 1):
                hij = jnp.vdot(basis[i], w)
                h_col.append(hij)
                w = w - hij * basis[i]

            h_next = jnp.linalg.norm(w)
            h_col.append(h_next.astype(w.dtype))
            if float(h_next) > 0.0:
                basis.append(w / h_next)
            else:
                basis.append(jnp.zeros_like(w))

            h_cols.append(_pad_h_col(h_col, m + 1))
            h = jnp.stack(h_cols, axis=1)[: j + 2, : j + 1]
            e1 = jnp.zeros((j + 2,), dtype=rhs.dtype).at[0].set(beta)
            y = jnp.linalg.lstsq(h, e1, rcond=None)[0]
            v = jnp.stack(basis[: j + 1], axis=0)
            dx = jnp.tensordot(y, v, axes=1)
            arnoldi_res = jnp.linalg.norm(e1 - h @ y)
            best_dx = dx
            best_relres = float(arnoldi_res / rhs_norm)

            if best_relres <= tol or matvecs >= max_matvecs:
                break

        x = x + best_dx
        relres = best_relres
        if relres <= tol or matvecs >= max_matvecs:
            break

        r = rhs - matvec(x)
        matvecs += 1
        beta = jnp.linalg.norm(r)
        relres = float(beta / rhs_norm)

    return GMRESResult(x=x, relres=relres, matvecs=matvecs)


def _pad_h_col(values: list[Array], length: int) -> Array:
    dtype = values[0].dtype
    out = jnp.zeros((length,), dtype=dtype)
    for i, value in enumerate(values):
        out = out.at[i].set(value)
    return out
