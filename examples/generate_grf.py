"""Generate a 64x64x64 correlated Gaussian random field."""

from __future__ import annotations

import argparse
from pathlib import Path

from jax import config
import jax.numpy as jnp
import numpy as np

from jax_helmholtz import gaussian_random_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--correlation-length", type=float, default=8.0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    args = parser.parse_args()

    config.update("jax_enable_x64", args.precision == "float64")
    dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    field = gaussian_random_field(
        (args.n, args.n, args.n),
        seed=args.seed,
        correlation_length=args.correlation_length,
        mean=0.0,
        std=1.0,
        dtype=dtype,
    )
    arr = np.asarray(field)
    out = Path(f"data/grf_{args.n}x{args.n}x{args.n}_seed{args.seed}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"wrote {out}")
    print(f"shape={arr.shape} dtype={arr.dtype}")
    print(f"mean={arr.mean():.6e} std={arr.std():.6e}")
    print(f"min={arr.min():.6e} max={arr.max():.6e}")


if __name__ == "__main__":
    main()
