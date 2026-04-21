"""Solve a GRF Helmholtz test problem using plain restarted GMRES."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from jax import config
import jax.numpy as jnp
import numpy as np

from jax_helmholtz import flatten_grid, jit_helmop, mat_setup_from_wavespeed
from jax_helmholtz import solve_gmres_spectral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solve-tol", type=float, default=1e-3)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--contrast-strength", type=float, default=0.20)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--quiet", action="store_true")
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

    label = f"{args.n}x{args.n}x{args.n}_grf_seed{args.seed}_{args.precision}_gmres"
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
    result = solve_gmres_spectral(
        rhs,
        op,
        tol=args.solve_tol,
        restart=args.restart,
        max_cycles=args.max_cycles,
        verbose=not args.quiet,
    )
    residual = rhs - jit_helmop(result.u, op)
    relres = float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
    pressure = jnp.reshape(result.u, op.n, order="F")
    np.save(pressure_path, np.asarray(pressure))
    np.save(residual_path, np.asarray(result.residual_history))
    elapsed = time.perf_counter() - start_time

    print(f"wavespeed wrote {wavespeed_path}")
    print(f"pressure wrote {pressure_path}")
    print(f"n={op.n} size={op.size} rho={op.rho}")
    print(f"frequency_cycles={frequency_cycles:.6e} ppw_min={ppw_min:.6e} kh_max={kh_max:.6e}")
    print(f"wavespeed min={jnp.min(wavespeed):.6e} max={jnp.max(wavespeed):.6e}")
    print(f"residual history wrote {residual_path}")
    print(f"restart={args.restart} cycles={result.cycles}")
    print(f"estimated matvecs={result.matvecs_estimate}")
    print(f"final relres={relres:.6e}")
    print(f"pressure norm={jnp.linalg.norm(result.u):.6e}")
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
