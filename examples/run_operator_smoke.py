"""Small operator smoke run.

Run with:

    python examples/run_operator_smoke.py
"""

from __future__ import annotations

from jax import config
import jax.numpy as jnp

from jax_helmholtz import flatten_grid, helmop, mat_setup


def main() -> None:
    config.update("jax_enable_x64", True)
    op = mat_setup((12, 10, 8), jnp.pi / 2.25, 2 * jnp.pi / 2.25, sparse=False)
    x = flatten_grid(jnp.ones(op.n, dtype=jnp.complex128))
    y = helmop(x, op)
    print(f"mode={op.mode} n={op.n} size={op.size} rho={op.rho}")
    print(f"helmop shape={y.shape} norm={jnp.linalg.norm(y):.6e}")


if __name__ == "__main__":
    main()
