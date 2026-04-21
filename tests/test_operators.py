from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is not installed in this environment.",
)


def test_operator_shapes_and_linearity():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import flatten_grid, helmop, mat_setup

    config.update("jax_enable_x64", True)
    op = mat_setup((6, 5, 4), jnp.pi / 2.25, 2 * jnp.pi / 2.25)
    x = flatten_grid(jnp.arange(op.size, dtype=jnp.float64).reshape(op.n))
    x = x.astype(jnp.complex128)
    y = helmop(x, op)

    assert y.shape == (op.size,)
    assert jnp.allclose(helmop(2 * x, op), 2 * y)


def test_fd_operator_shapes():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import flatten_grid, helmop, mat_setup

    config.update("jax_enable_x64", True)
    op = mat_setup((6, 5, 4), jnp.pi / 9.0, 2 * jnp.pi / 9.0, sparse=True)
    x = flatten_grid(jnp.ones(op.n, dtype=jnp.complex128))

    assert helmop(x, op).shape == (op.size,)
    assert helmop(jnp.concatenate([x, x]), op).shape == (2 * op.size,)


def test_fci_setup_shapes():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import fci_setup, mat_setup

    config.update("jax_enable_x64", True)
    op = mat_setup((6, 5, 4), jnp.pi / 2.25, 2 * jnp.pi / 2.25)
    fci = fci_setup(3, 0.4, 0.5, 1, 1, op)

    assert fci.shifts.shape == (3,)
    assert fci.weights.shape == (3,)
    assert fci.num.shape == (3,)


def test_gmres_solves_diagonal_system():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import gmres

    config.update("jax_enable_x64", True)
    diag = jnp.asarray([2 + 1j, 3 - 0.5j, 4 + 0j], dtype=jnp.complex128)
    rhs = jnp.asarray([1, 2, 3], dtype=jnp.complex128)

    result = gmres(lambda x: diag * x, rhs, restart=3, tol=1e-10, max_matvecs=6)

    assert result.relres < 1e-10
    assert jnp.allclose(diag * result.x, rhs)


def test_gaussian_random_field_shape_and_stats():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import gaussian_random_field

    config.update("jax_enable_x64", True)
    field = gaussian_random_field((8, 8, 8), seed=1, correlation_length=2.0)

    assert field.shape == (8, 8, 8)
    assert jnp.abs(jnp.mean(field)) < 1e-10
    assert jnp.abs(jnp.std(field) - 1) < 1e-10


def test_setup_from_wavespeed_mass_convention():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import mat_setup_from_wavespeed

    config.update("jax_enable_x64", True)
    wavespeed = jnp.ones((4, 4, 4), dtype=jnp.float64).at[1, 1, 1].set(2.0)
    op = mat_setup_from_wavespeed(wavespeed, 2 * jnp.pi / 2.25)

    assert op.mass.shape == wavespeed.shape
    assert jnp.max(op.mass) == 1
    assert jnp.isclose(op.mass[1, 1, 1], 0.25)


def test_fast_spectral_fci_smoke():
    from jax import config
    import jax.numpy as jnp

    from jax_helmholtz import fci_apply_spectral_jit, fci_setup, flatten_grid, mat_setup

    config.update("jax_enable_x64", True)
    op = mat_setup((4, 4, 4), jnp.pi / 2.25, 2 * jnp.pi / 2.25)
    rhs_grid = jnp.zeros(op.n, dtype=jnp.complex128).at[2, 2, 2].set(1 + 0j)
    params = fci_setup(1, 0.4, 0.5, 1, 1, op, krylov_dim=4)

    result = fci_apply_spectral_jit(flatten_grid(rhs_grid), op, params)

    assert result.u.shape == (op.size,)
    assert jnp.all(jnp.isfinite(result.u))
