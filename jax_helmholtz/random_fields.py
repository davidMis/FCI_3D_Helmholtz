"""Random fields for test media and solver experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def gaussian_random_field(
    shape: tuple[int, int, int],
    *,
    seed: int = 0,
    correlation_length: float = 8.0,
    mean: float = 0.0,
    std: float = 1.0,
    dtype=jnp.float64,
) -> Array:
    """Generate a real-valued correlated Gaussian random field.

    ``correlation_length`` is measured in grid cells. Larger values produce a
    smoother field. The returned sample is normalized to the requested sample mean
    and standard deviation.
    """

    if len(shape) != 3:
        raise ValueError("shape must be a 3D tuple.")
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive.")
    if std < 0:
        raise ValueError("std must be non-negative.")

    key = jax.random.PRNGKey(seed)
    white = jax.random.normal(key, shape, dtype=dtype)
    spectrum = jnp.fft.fftn(white)
    kx, ky, kz = _angular_frequency_grid(shape, dtype=dtype)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    amplitude = jnp.exp(-0.5 * (correlation_length**2) * k2)
    field = jnp.real(jnp.fft.ifftn(spectrum * amplitude))

    field = field - jnp.mean(field)
    sample_std = jnp.std(field)
    field = jnp.where(sample_std > 0, field / sample_std, field)
    return field * std + mean


def _angular_frequency_grid(shape: tuple[int, int, int], *, dtype) -> tuple[Array, Array, Array]:
    return tuple(
        2 * jnp.pi * jnp.fft.fftfreq(count).astype(dtype) for count in shape
    )
