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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--correlation-length", type=float, default=8.0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to data/grf_* based on n, seed, and batch size.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    config.update("jax_enable_x64", args.precision == "float64")
    dtype = jnp.float64 if args.precision == "float64" else jnp.float32

    fields = []
    for offset in range(args.batch_size):
        fields.append(
            gaussian_random_field(
                (args.n, args.n, args.n),
                seed=args.seed + offset,
                correlation_length=args.correlation_length,
                mean=0.0,
                std=1.0,
                dtype=dtype,
            )
        )
    field = fields[0] if args.batch_size == 1 else jnp.stack(fields, axis=0)
    arr = np.asarray(field)
    out = args.output
    if out is None:
        if args.batch_size == 1:
            out = Path(f"data/grf_{args.n}x{args.n}x{args.n}_seed{args.seed}.npy")
        else:
            out = Path(
                f"data/grf_{args.n}x{args.n}x{args.n}"
                f"_seed{args.seed}_batch{args.batch_size}.npy"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"wrote {out}")
    print(f"shape={arr.shape} dtype={arr.dtype}")
    print(f"mean={arr.mean():.6e} std={arr.std():.6e}")
    print(f"min={arr.min():.6e} max={arr.max():.6e}")


if __name__ == "__main__":
    main()
