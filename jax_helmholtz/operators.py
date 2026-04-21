"""Matrix-free Helmholtz operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .setup import HelmholtzOperator


Array = jnp.ndarray


def flatten_grid(x: Array) -> Array:
    """Flatten a 3D grid with Matlab-compatible column-major ordering."""

    return jnp.reshape(x, (-1,), order="F")


def unflatten_grid(x: Array, n: tuple[int, int, int]) -> Array:
    """Reshape a vector into a 3D grid with Matlab-compatible ordering."""

    return jnp.reshape(x, n, order="F")


def stiffop(x: Array, op: HelmholtzOperator) -> Array:
    """Apply the stiffness matrix ``S`` without forming it."""

    x_grid = unflatten_grid(x, op.n)
    if op.mode == "spectral":
        if op.stiffness_eigs is None:
            raise ValueError("Spectral mode requires stiffness_eigs.")
        y_grid = jnp.fft.ifftn(op.stiffness_eigs * jnp.fft.fftn(x_grid))
    elif op.mode == "fd":
        y_grid = _fd_stiffness_grid(x_grid, op.kh_max)
    else:
        raise ValueError(f"Unknown operator mode: {op.mode}")
    return flatten_grid(y_grid)


def helmop(x: Array, op: HelmholtzOperator) -> Array:
    """Apply the Helmholtz operator.

    If ``x`` has length ``prod(op.n)``, this applies the original one-block
    operator. If ``x`` has double length, this applies the extended two-block system
    used by the paper and Matlab implementation.
    """

    n = op.size
    mass = flatten_grid(op.mass)
    damping = flatten_grid(op.damping)

    if x.shape[0] == n:
        return stiffop(x, op) - (mass + 1j * damping) * x

    if x.shape[0] != 2 * n:
        raise ValueError(f"Expected vector length {n} or {2 * n}; got {x.shape[0]}.")

    x1 = x[:n]
    x2 = x[n:]
    y1 = 1j * x2 - x1
    y2 = -1j * (stiffop(x1, op) + (1 - mass) * x1) - (1 + 1j * damping) * x2
    return jnp.concatenate([y1, y2])


jit_stiffop = jax.jit(stiffop)
jit_helmop = jax.jit(helmop)


def helmsym(x: Array, op: HelmholtzOperator) -> Array:
    """Hermitian part used by the Chebyshev shifted solve."""

    n = op.size
    mass = flatten_grid(op.mass)

    if x.shape[0] == n:
        return stiffop(x, op) - mass * x

    if x.shape[0] != 2 * n:
        raise ValueError(f"Expected vector length {n} or {2 * n}; got {x.shape[0]}.")

    x1 = x[:n]
    x2 = x[n:]
    return 1j * jnp.concatenate(
        [x2, -stiffop(x1, op) - (1 - mass) * x1]
    ) - x


def _fd_stiffness_grid(x: Array, kh_max: float) -> Array:
    padded = jnp.pad(x, ((1, 1), (1, 1), (1, 1)), mode="constant")
    center = padded[1:-1, 1:-1, 1:-1]
    y = 6 * center
    y = y - padded[:-2, 1:-1, 1:-1] - padded[2:, 1:-1, 1:-1]
    y = y - padded[1:-1, :-2, 1:-1] - padded[1:-1, 2:, 1:-1]
    y = y - padded[1:-1, 1:-1, :-2] - padded[1:-1, 1:-1, 2:]
    return y / kh_max**2
