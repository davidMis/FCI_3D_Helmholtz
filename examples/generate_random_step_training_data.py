"""Generate shifted-solve training data from in-memory GRFs.

This is a one-stage replacement for generating GRF files and then solving them.
For each Gaussian random field, it samples one refinement step and one FCI pole,
writes exactly one training tensor, and then discards the field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from jax import config
import jax.numpy as jnp
import numpy as np

from jax_helmholtz import (
    fci_apply_spectral_jit,
    fci_setup,
    flatten_grid,
    gaussian_random_field,
    mat_setup_from_wavespeed,
)
from jax_helmholtz.training_data import TRAINING_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64, help="Grid size in each dimension.")
    parser.add_argument(
        "--num-grfs",
        type=int,
        required=True,
        help="Number of in-memory GRFs to generate. One training sample is saved per GRF.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for GRF generation.")
    parser.add_argument(
        "--step-seed",
        type=int,
        default=None,
        help="Seed for random refinement-step and pole choices. Defaults to seed + 1000000.",
    )
    parser.add_argument("--correlation-length", type=float, default=8.0)
    parser.add_argument("--contrast-strength", type=float, default=0.20)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shifted_training_random_steps.npy"),
        help="Output .npy path with shape (num_grfs, channels, n, n, n).",
    )
    parser.add_argument("--max-refinement-steps", type=int, default=10)
    parser.add_argument("--npoles", type=int, default=1)
    parser.add_argument("--krylov-dim", type=int, default=20)
    parser.add_argument(
        "--inner-solver",
        choices=("gmres", "none", "richardson", "chebyshev"),
        default="gmres",
    )
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument(
        "--inner-alpha",
        type=float,
        default=None,
        help="Richardson step size. Omit to use an automatic residual-line step.",
    )
    parser.add_argument("--ppw-min", type=float, default=2.25)
    parser.add_argument(
        "--frequency-cycles",
        type=float,
        default=None,
        help="Nondimensional frequency omega/(2*pi) across the unit-width grid.",
    )
    parser.add_argument(
        "--profile-fci",
        action="store_true",
        help="Synchronize JAX phase boundaries and print per-step FCI timing.",
    )
    args = parser.parse_args()
    validate_args(parser, args)

    config.update("jax_enable_x64", args.precision == "float64")
    real_dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.precision == "float64" else jnp.complex64
    output_dtype = np.float64 if args.precision == "float64" else np.float32

    step_seed = args.step_seed if args.step_seed is not None else args.seed + 1_000_000
    rng = np.random.default_rng(step_seed)
    target_steps = rng.integers(
        1,
        args.max_refinement_steps + 1,
        size=args.num_grfs,
        endpoint=False,
    )
    target_poles = rng.integers(0, args.npoles, size=args.num_grfs, endpoint=False)

    if args.frequency_cycles is None:
        frequency_cycles = args.n / args.ppw_min
    else:
        frequency_cycles = args.frequency_cycles
    ppw_min = args.n / frequency_cycles
    kh_max = 2 * jnp.pi * frequency_cycles / args.n

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = np.lib.format.open_memmap(
        args.output,
        mode="w+",
        dtype=output_dtype,
        shape=(args.num_grfs, len(TRAINING_CHANNELS), args.n, args.n, args.n),
    )

    print("training data channel order: " + ",".join(TRAINING_CHANNELS), flush=True)
    print(
        f"writing {args.output} shape={dataset.shape} dtype={dataset.dtype}",
        flush=True,
    )
    print(f"random choices seed={step_seed}", flush=True)
    start_time = time.perf_counter()

    for grf_index in range(args.num_grfs):
        field_seed = args.seed + grf_index
        target_step = int(target_steps[grf_index])
        target_pole = int(target_poles[grf_index])
        tensor = generate_one_training_tensor(
            field_seed=field_seed,
            target_step=target_step,
            target_pole=target_pole,
            args=args,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            frequency_cycles=frequency_cycles,
            ppw_min=ppw_min,
            kh_max=kh_max,
        )
        dataset[grf_index] = tensor
        dataset.flush()
        print(
            f"wrote sample={grf_index + 1}/{args.num_grfs} "
            f"field_seed={field_seed} step={target_step} pole={target_pole}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    print(f"done output={args.output} elapsed_seconds={elapsed:.3f}", flush=True)


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.n < 1:
        parser.error("--n must be positive.")
    if args.num_grfs < 1:
        parser.error("--num-grfs must be positive.")
    if args.correlation_length <= 0:
        parser.error("--correlation-length must be positive.")
    if args.max_refinement_steps < 1:
        parser.error("--max-refinement-steps must be positive.")
    if args.npoles < 1:
        parser.error("--npoles must be positive.")
    if args.krylov_dim < 1:
        parser.error("--krylov-dim must be positive.")
    if args.inner_steps < 0:
        parser.error("--inner-steps must be non-negative.")
    if args.ppw_min <= 0:
        parser.error("--ppw-min must be positive.")
    if args.frequency_cycles is not None and args.frequency_cycles <= 0:
        parser.error("--frequency-cycles must be positive when set.")


def generate_one_training_tensor(
    *,
    field_seed: int,
    target_step: int,
    target_pole: int,
    args,
    real_dtype,
    complex_dtype,
    frequency_cycles: float,
    ppw_min: float,
    kh_max,
) -> np.ndarray:
    grf = gaussian_random_field(
        (args.n, args.n, args.n),
        seed=field_seed,
        correlation_length=args.correlation_length,
        mean=0.0,
        std=1.0,
        dtype=real_dtype,
    )
    wavespeed = grf_to_wavespeed(grf, contrast_strength=args.contrast_strength)
    op = mat_setup_from_wavespeed(wavespeed, kh_max, sparse=False, dtype=real_dtype)
    print(
        f"run field_seed={field_seed} n={op.n} size={op.size} "
        f"rho={op.rho} target_step={target_step} target_pole={target_pole}",
        flush=True,
    )

    rhs = point_source(op.n, dtype=complex_dtype)
    params = fci_setup(
        npoles=args.npoles,
        sep=0.4 * (40 / op.n[0]),
        asp=0.5,
        nblock=1,
        method=1,
        op=op,
        krylov_dim=args.krylov_dim,
        tol_outer=2e-1,
        tol_inner=2e-1,
    )

    u = jnp.zeros_like(rhs)
    residual = rhs
    rhs_norm = jnp.linalg.norm(rhs)
    mass_np = np.asarray(op.mass)
    damping_np = np.asarray(op.damping)
    captured: list[np.ndarray] = []

    for step in range(1, target_step + 1):
        relres_before = float(jnp.linalg.norm(residual) / rhs_norm)
        callback = None
        if step == target_step:
            callback = make_single_training_tensor_callback(
                samples=captured,
                target_pole=target_pole,
                mass=mass_np,
                damping=damping_np,
                kh_max=float(kh_max),
                step=step,
                relres_before=relres_before,
            )

        result = fci_apply_spectral_jit(
            residual,
            op,
            params,
            inner_solver=args.inner_solver,
            inner_steps=args.inner_steps,
            inner_alpha=args.inner_alpha,
            profile=args.profile_fci,
            shifted_sample_callback=callback,
        )
        u = u + result.u
        residual = result.residual
        relres = float(jnp.linalg.norm(residual) / rhs_norm)
        print(
            "refine "
            f"field_seed={field_seed} step={step} relres={relres:.6e} "
            f"step_matvecs={result.matvecs_estimate}",
            flush=True,
        )
        if args.profile_fci and result.profile is not None:
            print(format_fci_profile(step, result.profile), flush=True)

    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one captured sample for seed={field_seed}, "
            f"step={target_step}, pole={target_pole}; got {len(captured)}."
        )

    print(f"frequency_cycles={frequency_cycles:.6e} ppw_min={ppw_min:.6e}")
    print(f"wavespeed min={jnp.min(wavespeed):.6e} max={jnp.max(wavespeed):.6e}")
    print(f"pressure norm at sampled step={jnp.linalg.norm(u):.6e}")
    return captured[0]


def make_single_training_tensor_callback(
    *,
    samples: list[np.ndarray],
    target_pole: int,
    mass: np.ndarray,
    damping: np.ndarray,
    kh_max: float,
    step: int,
    relres_before: float,
):
    def save_sample(sample) -> None:
        if sample.pole_index != target_pole:
            return
        if samples:
            raise RuntimeError(
                f"Already captured a sample for step={step}, pole={target_pole}."
            )

        residual = np.asarray(sample.rhs)
        solution = np.asarray(sample.solution)
        shifted_residual = np.asarray(sample.shifted_residual)
        dtype = mass.dtype
        shape = mass.shape
        shift = complex(np.asarray(sample.shift).item())
        tensor = np.stack(
            [
                mass,
                damping,
                residual.real.astype(dtype, copy=False),
                residual.imag.astype(dtype, copy=False),
                solution.real.astype(dtype, copy=False),
                solution.imag.astype(dtype, copy=False),
                shifted_residual.real.astype(dtype, copy=False),
                shifted_residual.imag.astype(dtype, copy=False),
                np.full(shape, shift.real, dtype=dtype),
                np.full(shape, shift.imag, dtype=dtype),
                np.full(shape, kh_max, dtype=dtype),
                np.full(shape, relres_before, dtype=dtype),
                np.full(shape, step, dtype=dtype),
                np.full(shape, sample.pole_index, dtype=dtype),
            ],
            axis=0,
        )
        samples.append(tensor)
        print(f"training-sample captured step={step} pole={sample.pole_index}", flush=True)

    return save_sample


def grf_to_wavespeed(field: jnp.ndarray, *, contrast_strength: float) -> jnp.ndarray:
    speed = jnp.exp(contrast_strength * field)
    return speed / jnp.min(speed)


def point_source(shape: tuple[int, int, int], *, dtype) -> jnp.ndarray:
    source = jnp.zeros(shape, dtype=dtype)
    center = tuple(v // 2 for v in shape)
    source = source.at[center].set(1 + 0j)
    return flatten_grid(source)


def format_fci_profile(step: int, profile) -> str:
    shifted = profile.shifted_total_seconds
    total = profile.total_seconds
    shifted_pct = 100 * shifted / total if total > 0 else 0.0
    pole_times = ",".join(f"{value:.4f}" for value in profile.shifted_solve_seconds)
    return (
        "profile "
        f"step={step} total_seconds={total:.4f} "
        f"shifted_solve_seconds={shifted:.4f} "
        f"shifted_pct={shifted_pct:.1f} "
        f"pole_seconds=[{pole_times}] "
        f"contour_combine_seconds={profile.contour_combine_seconds:.4f} "
        f"postprocess_seconds={profile.postprocess_seconds:.4f} "
        f"inner_seconds={profile.inner_seconds:.4f} "
        f"sample_seconds={profile.sample_seconds:.4f}"
    )


if __name__ == "__main__":
    main()
