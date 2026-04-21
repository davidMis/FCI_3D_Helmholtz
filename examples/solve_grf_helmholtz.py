"""Solve a Helmholtz test problem using the generated Gaussian random field."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from jax import config
import jax.numpy as jnp
import numpy as np

from jax_helmholtz import fci_apply, fci_apply_spectral_jit, fci_setup, flatten_grid, jit_helmop
from jax_helmholtz import mat_setup_from_wavespeed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solve-tol", type=float, default=1e-3)
    parser.add_argument("--max-refinement-steps", type=int, default=10)
    parser.add_argument("--contrast-strength", type=float, default=0.20)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--fast-spectral", action="store_true")
    parser.add_argument("--npoles", type=int, default=1)
    parser.add_argument("--ppw-min", type=float, default=2.25)
    parser.add_argument(
        "--frequency-cycles",
        type=float,
        default=None,
        help="Nondimensional frequency omega/(2*pi) across the unit-width grid.",
    )
    args = parser.parse_args()

    config.update("jax_enable_x64", args.precision == "float64")
    real_dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.precision == "float64" else jnp.complex64

    solver_label = "fast_spectral" if args.fast_spectral else "reference"
    label = f"{args.n}x{args.n}x{args.n}_grf_seed{args.seed}_{args.precision}_{solver_label}"
    field_path = Path(f"data/grf_{args.n}x{args.n}x{args.n}_seed{args.seed}.npy")
    pressure_path = Path(f"data/pressure_{label}.npy")
    residual_path = Path(f"data/residual_history_{label}.npy")
    wavespeed_path = Path(f"data/wavespeed_{label}.npy")

    start_time = time.perf_counter()
    grf = jnp.asarray(np.load(field_path), dtype=real_dtype)
    wavespeed = grf_to_wavespeed(grf, contrast_strength=args.contrast_strength)
    np.save(wavespeed_path, np.asarray(wavespeed))

    if args.frequency_cycles is None:
        frequency_cycles = args.n / args.ppw_min
        ppw_min = args.ppw_min
    else:
        frequency_cycles = args.frequency_cycles
        ppw_min = args.n / frequency_cycles
    kh_max = 2 * jnp.pi * frequency_cycles / args.n
    op = mat_setup_from_wavespeed(wavespeed, kh_max, sparse=False, dtype=real_dtype)

    rhs = point_source(op.n, dtype=complex_dtype)
    params = fci_setup(
        npoles=args.npoles,
        sep=0.4 * (40 / op.n[0]),
        asp=0.5,
        nblock=1,
        method=1,
        op=op,
        tol_outer=2e-1,
        tol_inner=2e-1,
    )

    u = jnp.zeros_like(rhs)
    residual = rhs
    rhs_norm = jnp.linalg.norm(rhs)
    residual_history = []
    total_matvecs = 0
    for step in range(1, args.max_refinement_steps + 1):
        if args.fast_spectral:
            result = fci_apply_spectral_jit(residual, op, params)
            step_solution = result.u
            step_matvecs = result.matvecs_estimate
        else:
            result = fci_apply(residual, op, params)
            step_solution = result.u
            step_matvecs = result.matvecs + 1

        u = u + step_solution
        residual = rhs - jit_helmop(u, op)
        relres = float(jnp.linalg.norm(residual) / rhs_norm)
        total_matvecs += step_matvecs
        residual_history.append(relres)
        print(
            "refine "
            f"step={step} relres={relres:.6e} "
            f"step_matvecs={step_matvecs} total_matvecs={total_matvecs}",
            flush=True,
        )
        if relres < args.solve_tol:
            break

    pressure = jnp.reshape(u, op.n, order="F")
    np.save(pressure_path, np.asarray(pressure))
    np.save(residual_path, np.asarray(residual_history))
    elapsed = time.perf_counter() - start_time

    print(f"wavespeed wrote {wavespeed_path}")
    print(f"pressure wrote {pressure_path}")
    print(f"n={op.n} size={op.size} rho={op.rho}")
    print(f"frequency_cycles={frequency_cycles:.6e} ppw_min={ppw_min:.6e} kh_max={kh_max:.6e}")
    print(f"wavespeed min={jnp.min(wavespeed):.6e} max={jnp.max(wavespeed):.6e}")
    print(f"residual history wrote {residual_path}")
    print(f"total matvecs={total_matvecs}")
    print(f"final relres={residual_history[-1]:.6e}")
    print(f"pressure norm={jnp.linalg.norm(u):.6e}")
    print(f"elapsed seconds={elapsed:.3f}")


def grf_to_wavespeed(field: jnp.ndarray, *, contrast_strength: float) -> jnp.ndarray:
    speed = jnp.exp(contrast_strength * field)
    return speed / jnp.min(speed)


def point_source(shape: tuple[int, int, int], *, dtype) -> jnp.ndarray:
    source = jnp.zeros(shape, dtype=dtype)
    center = tuple(v // 2 for v in shape)
    source = source.at[center].set(1 + 0j)
    return flatten_grid(source)


if __name__ == "__main__":
    main()
