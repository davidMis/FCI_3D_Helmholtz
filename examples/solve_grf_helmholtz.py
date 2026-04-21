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


TRAINING_CHANNELS = (
    "mass",
    "damping",
    "residual_real",
    "residual_imag",
    "shifted_solution_real",
    "shifted_solution_imag",
    "shifted_residual_real",
    "shifted_residual_imag",
    "shift_real",
    "shift_imag",
    "kh_max",
    "relres_before",
    "step",
    "pole_index",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used to infer the default GRF path when --grf-path is omitted.",
    )
    parser.add_argument(
        "--grf-path",
        type=Path,
        default=None,
        help="Input GRF .npy path. Supports shape (n,n,n) or (batch,n,n,n).",
    )
    parser.add_argument("--solve-tol", type=float, default=1e-3)
    parser.add_argument("--max-refinement-steps", type=int, default=10)
    parser.add_argument("--contrast-strength", type=float, default=0.20)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--fast-spectral", action="store_true")
    parser.add_argument("--npoles", type=int, default=1)
    parser.add_argument("--krylov-dim", type=int, default=20)
    parser.add_argument(
        "--inner-solver",
        choices=("gmres", "none", "richardson", "chebyshev"),
        default="gmres",
        help="Inner correction used by --fast-spectral. Use 'none' for very large grids.",
    )
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument(
        "--inner-alpha",
        type=float,
        default=None,
        help="Richardson step size. Omit to use an automatic residual-line step.",
    )
    parser.add_argument("--no-save-pressure", action="store_true")
    parser.add_argument("--no-save-wavespeed", action="store_true")
    parser.add_argument("--save-slices-only", action="store_true")
    parser.add_argument(
        "--profile-fci",
        action="store_true",
        help="Synchronize JAX phase boundaries and print per-step FCI timing.",
    )
    parser.add_argument(
        "--save-shifted-samples",
        nargs="?",
        const="data/shifted_samples",
        default=None,
        metavar="DIR",
        help=(
            "Save shifted-solve training samples from the fast spectral FCI path. "
            "If DIR is omitted, writes to data/shifted_samples."
        ),
    )
    parser.add_argument(
        "--max-shifted-samples",
        type=int,
        default=None,
        help="Maximum number of shifted-solve samples to save for this run.",
    )
    parser.add_argument(
        "--training-data-out",
        type=Path,
        default=None,
        help=(
            "Write one stacked .npy tensor with shifted-solve training samples. "
            "Shape is (samples, channels, n, n, n)."
        ),
    )
    parser.add_argument("--ppw-min", type=float, default=2.25)
    parser.add_argument(
        "--frequency-cycles",
        type=float,
        default=None,
        help="Nondimensional frequency omega/(2*pi) across the unit-width grid.",
    )
    args = parser.parse_args()
    if args.save_shifted_samples is not None and not args.fast_spectral:
        parser.error("--save-shifted-samples currently requires --fast-spectral.")
    if args.training_data_out is not None and not args.fast_spectral:
        parser.error("--training-data-out currently requires --fast-spectral.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.max_shifted_samples is not None and args.max_shifted_samples < 1:
        parser.error("--max-shifted-samples must be positive when set.")

    config.update("jax_enable_x64", args.precision == "float64")
    real_dtype = jnp.float64 if args.precision == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.precision == "float64" else jnp.complex64

    solver_label = "fast_spectral" if args.fast_spectral else "reference"
    field_path = args.grf_path or default_grf_path(args.n, args.seed, args.batch_size)
    grf_batch = load_grf_batch(field_path, args.n)
    run_count = grf_batch.shape[0]

    start_time = time.perf_counter()
    if args.frequency_cycles is None:
        frequency_cycles = args.n / args.ppw_min
        ppw_min = args.ppw_min
    else:
        frequency_cycles = args.frequency_cycles
        ppw_min = args.n / frequency_cycles
    kh_max = 2 * jnp.pi * frequency_cycles / args.n

    print(f"loaded GRF batch {field_path} shape={grf_batch.shape}", flush=True)
    print_memory_estimate((args.n, args.n, args.n), real_dtype, complex_dtype)
    if args.profile_fci and args.fast_spectral:
        print("profile note: first fast-spectral step may include XLA compilation time", flush=True)

    shifted_sample_dir = None
    shifted_sample_counter = {"count": 0}
    if args.save_shifted_samples is not None:
        shifted_sample_dir = Path(args.save_shifted_samples)
        shifted_sample_dir.mkdir(parents=True, exist_ok=True)

    training_samples = [] if args.training_data_out is not None else None
    if training_samples is not None:
        print(
            "training data channel order: "
            + ",".join(TRAINING_CHANNELS),
            flush=True,
        )

    for batch_index, grf_np in enumerate(grf_batch):
        sample_seed = args.seed + batch_index
        label = (
            f"{args.n}x{args.n}x{args.n}_grf_seed{sample_seed}"
            f"_{args.precision}_{solver_label}"
        )
        solve_one_grf(
            grf_np,
            label=label,
            sample_seed=sample_seed,
            args=args,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
            frequency_cycles=frequency_cycles,
            ppw_min=ppw_min,
            kh_max=kh_max,
            shifted_sample_dir=shifted_sample_dir,
            shifted_sample_counter=shifted_sample_counter,
            training_samples=training_samples,
        )

    elapsed = time.perf_counter() - start_time

    if training_samples is not None:
        args.training_data_out.parent.mkdir(parents=True, exist_ok=True)
        if not training_samples:
            raise RuntimeError("No shifted-solve training samples were captured.")
        training_data = np.stack(training_samples, axis=0)
        np.save(args.training_data_out, training_data)
        print(
            f"training data wrote {args.training_data_out} "
            f"shape={training_data.shape} dtype={training_data.dtype}",
            flush=True,
        )

    print(f"solved batch_count={run_count}")
    print(f"elapsed seconds={elapsed:.3f}")


def default_grf_path(n: int, seed: int, batch_size: int) -> Path:
    if batch_size == 1:
        return Path(f"data/grf_{n}x{n}x{n}_seed{seed}.npy")
    return Path(f"data/grf_{n}x{n}x{n}_seed{seed}_batch{batch_size}.npy")


def load_grf_batch(path: Path, n: int) -> np.ndarray:
    arr = np.load(path)
    expected_shape = (n, n, n)
    if arr.shape == expected_shape:
        return arr[None, ...]
    if arr.ndim == 4 and arr.shape[1:] == expected_shape:
        return arr
    raise ValueError(
        f"Expected GRF shape {expected_shape} or (batch, {n}, {n}, {n}); "
        f"got {arr.shape} from {path}."
    )


def solve_one_grf(
    grf_np: np.ndarray,
    *,
    label: str,
    sample_seed: int,
    args,
    real_dtype,
    complex_dtype,
    frequency_cycles: float,
    ppw_min: float,
    kh_max,
    shifted_sample_dir: Path | None,
    shifted_sample_counter: dict[str, int],
    training_samples: list[np.ndarray] | None,
) -> None:
    pressure_path = Path(f"data/pressure_{label}.npy")
    residual_path = Path(f"data/residual_history_{label}.npy")
    wavespeed_path = Path(f"data/wavespeed_{label}.npy")

    grf = jnp.asarray(grf_np, dtype=real_dtype)
    wavespeed = grf_to_wavespeed(grf, contrast_strength=args.contrast_strength)
    if not args.no_save_wavespeed:
        if args.save_slices_only:
            np.save(sliced_path(wavespeed_path), center_slices(np.asarray(wavespeed)))
        else:
            np.save(wavespeed_path, np.asarray(wavespeed))

    op = mat_setup_from_wavespeed(wavespeed, kh_max, sparse=False, dtype=real_dtype)
    print(f"run label={label} n={op.n} size={op.size} rho={op.rho}", flush=True)

    shifted_medium_path = None
    if shifted_sample_dir is not None:
        shifted_medium_path = shifted_sample_dir / f"medium_{label}.npz"
        save_shifted_medium(
            shifted_medium_path,
            op,
            frequency_cycles=frequency_cycles,
            ppw_min=ppw_min,
            contrast_strength=args.contrast_strength,
            seed=sample_seed,
        )
        print(f"shifted-sample medium wrote {shifted_medium_path}", flush=True)

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
    residual_history = []
    total_matvecs = 0
    mass_np = np.asarray(op.mass)
    damping_np = np.asarray(op.damping)

    for step in range(1, args.max_refinement_steps + 1):
        shifted_sample_callback = None
        callbacks = []
        relres_before = None

        wants_npz_sample = (
            shifted_sample_dir is not None
            and (
                args.max_shifted_samples is None
                or shifted_sample_counter["count"] < args.max_shifted_samples
            )
        )
        wants_training_sample = training_samples is not None
        if wants_npz_sample or wants_training_sample:
            relres_before = float(jnp.linalg.norm(residual) / rhs_norm)

        if wants_npz_sample:
            callbacks.append(
                make_shifted_sample_callback(
                    sample_dir=shifted_sample_dir,
                    label=label,
                    medium_path=shifted_medium_path,
                    step=step,
                    relres_before=relres_before,
                    counter=shifted_sample_counter,
                    max_samples=args.max_shifted_samples,
                )
            )
        if wants_training_sample:
            callbacks.append(
                make_training_tensor_callback(
                    samples=training_samples,
                    mass=mass_np,
                    damping=damping_np,
                    kh_max=float(kh_max),
                    step=step,
                    relres_before=relres_before,
                )
            )
        if callbacks:
            shifted_sample_callback = combine_callbacks(callbacks)

        if args.fast_spectral:
            result = fci_apply_spectral_jit(
                residual,
                op,
                params,
                inner_solver=args.inner_solver,
                inner_steps=args.inner_steps,
                inner_alpha=args.inner_alpha,
                profile=args.profile_fci,
                shifted_sample_callback=shifted_sample_callback,
            )
            step_solution = result.u
            step_residual = result.residual
            step_matvecs = result.matvecs_estimate
        else:
            result = fci_apply(residual, op, params, profile=args.profile_fci)
            step_solution = result.u
            step_residual = None
            step_matvecs = result.matvecs + 1

        u = u + step_solution
        if step_residual is None:
            residual = rhs - jit_helmop(u, op)
            relres = float(jnp.linalg.norm(residual) / rhs_norm)
        else:
            residual = step_residual
            relres = float(jnp.linalg.norm(residual) / rhs_norm)
        total_matvecs += step_matvecs
        residual_history.append(relres)
        print(
            "refine "
            f"label={label} step={step} relres={relres:.6e} "
            f"step_matvecs={step_matvecs} total_matvecs={total_matvecs}",
            flush=True,
        )
        if args.profile_fci and result.profile is not None:
            print(format_fci_profile(step, result.profile), flush=True)
        if relres < args.solve_tol:
            break

    pressure = jnp.reshape(u, op.n, order="F")
    if not args.no_save_pressure:
        if args.save_slices_only:
            np.save(sliced_path(pressure_path), center_slices(np.asarray(pressure)))
        else:
            np.save(pressure_path, np.asarray(pressure))
    np.save(residual_path, np.asarray(residual_history))

    if args.no_save_wavespeed:
        print("wavespeed save skipped")
    elif args.save_slices_only:
        print(f"wavespeed slices wrote {sliced_path(wavespeed_path)}")
    else:
        print(f"wavespeed wrote {wavespeed_path}")
    if args.no_save_pressure:
        print("pressure save skipped")
    elif args.save_slices_only:
        print(f"pressure slices wrote {sliced_path(pressure_path)}")
    else:
        print(f"pressure wrote {pressure_path}")
    print(f"frequency_cycles={frequency_cycles:.6e} ppw_min={ppw_min:.6e} kh_max={kh_max:.6e}")
    print(f"wavespeed min={jnp.min(wavespeed):.6e} max={jnp.max(wavespeed):.6e}")
    print(f"residual history wrote {residual_path}")
    print(f"total matvecs={total_matvecs}")
    print(f"final relres={residual_history[-1]:.6e}")
    print(f"pressure norm={jnp.linalg.norm(u):.6e}")


def combine_callbacks(callbacks):
    def combined(sample) -> None:
        for callback in callbacks:
            callback(sample)

    return combined


def make_training_tensor_callback(
    *,
    samples: list[np.ndarray],
    mass: np.ndarray,
    damping: np.ndarray,
    kh_max: float,
    step: int,
    relres_before: float,
):
    def save_sample(sample) -> None:
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
        print(
            f"training-sample buffered count={len(samples)} "
            f"step={step} pole={sample.pole_index}",
            flush=True,
        )

    return save_sample


def grf_to_wavespeed(field: jnp.ndarray, *, contrast_strength: float) -> jnp.ndarray:
    speed = jnp.exp(contrast_strength * field)
    return speed / jnp.min(speed)


def point_source(shape: tuple[int, int, int], *, dtype) -> jnp.ndarray:
    source = jnp.zeros(shape, dtype=dtype)
    center = tuple(v // 2 for v in shape)
    source = source.at[center].set(1 + 0j)
    return flatten_grid(source)


def center_slices(arr: np.ndarray) -> np.ndarray:
    center = tuple(v // 2 for v in arr.shape[:3])
    return np.stack(
        [
            arr[center[0], :, :],
            arr[:, center[1], :],
            arr[:, :, center[2]],
        ],
        axis=0,
    )


def sliced_path(path: Path) -> Path:
    return path.with_name(path.stem + "_center_slices" + path.suffix)


def save_shifted_medium(
    path: Path,
    op,
    *,
    frequency_cycles: float,
    ppw_min: float,
    contrast_strength: float,
    seed: int,
) -> None:
    if op.stiffness_eigs is None:
        raise ValueError("shifted-solve sample logging requires spectral stiffness_eigs.")

    np.savez(
        path,
        mass=np.asarray(op.mass),
        damping=np.asarray(op.damping),
        stiffness_eigs=np.asarray(op.stiffness_eigs),
        n=np.asarray(op.n, dtype=np.int64),
        kh_min=np.asarray(op.kh_min),
        kh_max=np.asarray(op.kh_max),
        rho=np.asarray(op.rho),
        frequency_cycles=np.asarray(frequency_cycles),
        ppw_min=np.asarray(ppw_min),
        contrast_strength=np.asarray(contrast_strength),
        seed=np.asarray(seed, dtype=np.int64),
    )


def make_shifted_sample_callback(
    *,
    sample_dir: Path,
    label: str,
    medium_path: Path,
    step: int,
    relres_before: float,
    counter: dict[str, int],
    max_samples: int | None,
):
    def save_sample(sample) -> None:
        if max_samples is not None and counter["count"] >= max_samples:
            return

        sample_index = counter["count"]
        counter["count"] += 1
        path = sample_dir / (
            f"shifted_sample_{label}_idx{sample_index:06d}"
            f"_step{step:03d}_pole{sample.pole_index:02d}.npz"
        )

        rhs = np.asarray(sample.rhs)
        solution = np.asarray(sample.solution)
        shifted_residual = np.asarray(sample.shifted_residual)
        rhs_norm = np.linalg.norm(rhs)
        shifted_residual_norm = np.linalg.norm(shifted_residual)
        shifted_relres = (
            shifted_residual_norm / rhs_norm if rhs_norm > 0 else shifted_residual_norm
        )

        np.savez(
            path,
            residual=rhs,
            shifted_solution=solution,
            shifted_residual=shifted_residual,
            shift=np.asarray(sample.shift),
            weight=np.asarray(sample.weight),
            z0=np.asarray(sample.z0),
            d=np.asarray(sample.d),
            q=np.asarray(sample.q, dtype=np.int64),
            niter=np.asarray(sample.niter, dtype=np.int64),
            step=np.asarray(step, dtype=np.int64),
            pole_index=np.asarray(sample.pole_index, dtype=np.int64),
            sample_index=np.asarray(sample_index, dtype=np.int64),
            relres_before=np.asarray(relres_before),
            shifted_relres=np.asarray(shifted_relres),
            medium_path=np.asarray(str(medium_path)),
        )
        print(
            "shifted-sample "
            f"wrote {path} "
            f"shifted_relres={shifted_relres:.6e}",
            flush=True,
        )

    return save_sample


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


def print_memory_estimate(
    shape: tuple[int, int, int],
    real_dtype,
    complex_dtype,
) -> None:
    npoints = int(np.prod(shape))
    real_bytes = np.dtype(np.float32 if real_dtype == jnp.float32 else np.float64).itemsize
    complex_bytes = np.dtype(np.complex64 if complex_dtype == jnp.complex64 else np.complex128).itemsize
    real_gb = npoints * real_bytes / 1024**3
    complex_gb = npoints * complex_bytes / 1024**3
    print(
        "memory estimate "
        f"grid={shape} real_vector={real_gb:.3f}GB "
        f"complex_vector={complex_gb:.3f}GB "
        f"baseline_fields~{(3 * real_gb + 4 * complex_gb):.3f}GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
