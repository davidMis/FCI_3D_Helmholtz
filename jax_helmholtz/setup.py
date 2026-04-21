"""Problem setup for the 3D Helmholtz operator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import jax
import jax.numpy as jnp


Array = jnp.ndarray
Mode = Literal["spectral", "fd"]


@dataclass(frozen=True)
class HelmholtzOperator:
    """Matrix-free Helmholtz operator data.

    The operator matches the Matlab form

        A = S - M - 1j * D

    and stores only fields plus the spectral Laplacian eigenvalues when FFT mode is
    selected. Finite-difference mode applies the 7-point stencil directly.
    """

    n: tuple[int, int, int]
    kh_min: float
    kh_max: float
    mode: Mode
    mass: Array
    damping: Array
    rho: tuple[float, float]
    stiffness_eigs: Array | None = None

    @property
    def size(self) -> int:
        return self.n[0] * self.n[1] * self.n[2]

    @property
    def sparse(self) -> bool:
        return self.mode == "fd"

    def tree_flatten(self):
        children = (self.mass, self.damping, self.stiffness_eigs)
        aux = (self.n, self.kh_min, self.kh_max, self.mode, self.rho)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mass, damping, stiffness_eigs = children
        n, kh_min, kh_max, mode, rho = aux
        return cls(
            n=n,
            kh_min=kh_min,
            kh_max=kh_max,
            mode=mode,
            mass=mass,
            damping=damping,
            rho=rho,
            stiffness_eigs=stiffness_eigs,
        )


jax.tree_util.register_pytree_node_class(HelmholtzOperator)


def mat_setup(
    n: tuple[int, int, int] | list[int],
    kh_min: float,
    kh_max: float,
    sparse: bool = False,
    *,
    dtype=jnp.float64,
) -> HelmholtzOperator:
    """Create the Matlab test operator in matrix-free JAX form.

    Parameters follow ``Matlab/mat_setup.m``:
    ``kh_min`` is ``2*pi / maximum_sampling_rate`` and ``kh_max`` is
    ``2*pi / minimum_sampling_rate``.
    """

    shape = tuple(int(v) for v in n)
    mode: Mode = "fd" if sparse else "spectral"
    contrast = (kh_min / kh_max) ** 2

    mass = jnp.ones(shape, dtype=dtype)
    for ix in (0.2, 0.6):
        for iy in (0.2, 0.6):
            for iz in (0.2, 0.6):
                mass = mass.at[
                    _matlab_frac_slice(shape[0], ix, ix + 0.2),
                    _matlab_frac_slice(shape[1], iy, iy + 0.2),
                    _matlab_frac_slice(shape[2], iz, iz + 0.2),
                ].set(contrast)
    mass = gaussian_smooth3(mass, size=9, sigma=1.0)

    return _operator_from_mass(
        mass,
        kh_min=kh_min,
        kh_max=kh_max,
        mode=mode,
        sparse=sparse,
        mass_lower_bound=contrast,
        dtype=dtype,
    )


def mat_setup_from_wavespeed(
    wavespeed: Array,
    kh_max: float,
    sparse: bool = False,
    *,
    dtype=jnp.float64,
) -> HelmholtzOperator:
    """Create an operator from a positive wavespeed grid.

    The field is normalized by its minimum value so that ``min(c)=1`` and
    ``mass = 1 / c**2`` satisfies the spectral assumption ``rho(M) <= 1``.
    """

    wavespeed = jnp.asarray(wavespeed, dtype=dtype)
    if wavespeed.ndim != 3:
        raise ValueError("wavespeed must be a 3D array.")
    if bool(jnp.any(wavespeed <= 0)):
        raise ValueError("wavespeed values must be positive.")

    normalized_speed = wavespeed / jnp.min(wavespeed)
    mass = 1 / normalized_speed**2
    kh_min = kh_max / float(jnp.max(normalized_speed))
    mode: Mode = "fd" if sparse else "spectral"
    return _operator_from_mass(
        mass,
        kh_min=kh_min,
        kh_max=kh_max,
        mode=mode,
        sparse=sparse,
        mass_lower_bound=float(jnp.min(mass)),
        dtype=dtype,
    )


def spectral_stiffness_eigs(
    n: tuple[int, int, int],
    kh_max: float,
    *,
    dtype=jnp.float64,
) -> Array:
    """Eigenvalues of the FFT spectral negative Laplacian."""

    axes = []
    for count in n:
        idx = jnp.arange(count, dtype=dtype)
        mirror = jnp.arange(count, 0, -1, dtype=dtype)
        axes.append(jnp.minimum(idx, mirror) * (2 * jnp.pi / count / kh_max))

    return (
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2
    )


def gaussian_smooth3(x: Array, *, size: int, sigma: float) -> Array:
    """Separable 3D Gaussian smoothing matching the Matlab setup intent."""

    if size % 2 != 1:
        raise ValueError("Gaussian smoothing size must be odd.")

    radius = size // 2
    offsets = jnp.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = jnp.exp(-(offsets**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)

    y = x
    for axis in range(3):
        y = _convolve_axis_same_zero(y, kernel, axis)
    return y


def taper(
    n: int,
    left: int,
    right: int,
    *,
    dtype=jnp.float64,
) -> Array:
    """Cosine taper at both ends, matching the helper in ``mat_setup.m``."""

    left = min(left, n)
    right = min(right, n)
    v = jnp.zeros((n,), dtype=dtype)
    if left > 0:
        i = jnp.arange(1, left + 1, dtype=dtype)
        values = 1 + jnp.cos(i / (left + 1) * jnp.pi)
        values = values / jnp.sum(values)
        v = v.at[:left].set(values)
    if right > 0:
        i = jnp.arange(1, right + 1, dtype=dtype)
        values = 1 + jnp.cos(i / (right + 1) * jnp.pi)
        values = values / jnp.sum(values)
        v = v.at[n - right :].set(values[::-1])
    return v


def _operator_from_mass(
    mass: Array,
    *,
    kh_min: float,
    kh_max: float,
    mode: Mode,
    sparse: bool,
    mass_lower_bound: float,
    dtype,
) -> HelmholtzOperator:
    shape = tuple(int(v) for v in mass.shape)
    damping = _absorbing_damping(shape, kh_max, sparse=sparse, dtype=dtype)

    if sparse:
        rho_hermitian = 12 / kh_max**2 + 1 - mass_lower_bound
        stiffness_eigs = None
    else:
        stiffness_eigs = spectral_stiffness_eigs(shape, kh_max, dtype=dtype)
        rho_hermitian = float(jnp.max(stiffness_eigs) + 1 - mass_lower_bound)

    return HelmholtzOperator(
        n=shape,
        kh_min=float(kh_min),
        kh_max=float(kh_max),
        mode=mode,
        mass=mass,
        damping=damping,
        rho=(float(rho_hermitian), float(jnp.max(damping))),
        stiffness_eigs=stiffness_eigs,
    )


def _absorbing_damping(
    shape: tuple[int, int, int],
    kh_max: float,
    *,
    sparse: bool,
    dtype,
) -> Array:
    nab = 20 if sparse else 10
    rab = 1e-2
    cab = abs(jnp.log(jnp.asarray(rab, dtype=dtype))) / kh_max * 2
    l1 = taper(shape[0], nab, nab, dtype=dtype) * cab
    l2 = taper(shape[1], nab, nab, dtype=dtype) * cab
    l3 = taper(shape[2], nab, nab, dtype=dtype) * cab
    return jnp.maximum(
        jnp.maximum(l1[:, None, None], l2[None, :, None]),
        l3[None, None, :],
    )


def _matlab_frac_slice(n: int, lo: float, hi: float) -> slice:
    start = math.ceil(n * lo) - 1
    stop = math.ceil(n * hi)
    return slice(start, stop)


def _convolve_axis_same_zero(x: Array, kernel: Array, axis: int) -> Array:
    radius = int(kernel.shape[0] // 2)
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (radius, radius)
    padded = jnp.pad(x, pad_width, mode="constant")
    out = jnp.zeros_like(x)
    for offset in range(kernel.shape[0]):
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(offset, offset + x.shape[axis])
        out = out + kernel[offset] * padded[tuple(slc)]
    return out
