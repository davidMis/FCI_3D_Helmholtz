"""Train a JAX neural surrogate for spectral shifted Helmholtz solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from flax import serialization
from flax.training import train_state
from jax import config
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from jax_helmholtz.setup import spectral_stiffness_eigs
from jax_helmholtz.surrogate import INPUT_CHANNELS, TARGET_CHANNELS, TRAINING_CHANNELS
from jax_helmholtz.surrogate import ShiftedUNet3D


EPS = 1e-12


class TrainState(train_state.TrainState):
    pass


class ShiftedTensorDataset:
    def __init__(self, path: Path):
        self.path = path
        self.array = np.load(path, mmap_mode="r")
        if self.array.ndim != 5:
            raise ValueError(
                f"Expected dataset shape (samples, channels, nx, ny, nz); got {self.array.shape}."
            )
        if self.array.shape[1] < len(TRAINING_CHANNELS):
            raise ValueError(
                f"Expected at least {len(TRAINING_CHANNELS)} channels; got {self.array.shape[1]}."
            )

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.array.shape[2:5])

    @property
    def kh_max(self) -> float:
        return float(self.array[0, TRAINING_CHANNELS.index("kh_max"), 0, 0, 0])

    def batches(self, batch_size: int, *, shuffle: bool, seed: int):
        indices = np.arange(self.array.shape[0])
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            yield np.asarray(self.array[batch_idx], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train_shifted_64.npy"))
    parser.add_argument("--val", type=Path, default=Path("data/val_shifted_64.npy"))
    parser.add_argument("--test", type=Path, default=Path("data/test_shifted_64.npy"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/shifted_unet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb-project", default="fci-shifted-surrogate")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.base_channels < 1:
        parser.error("--base-channels must be positive.")
    if args.depth < 1:
        parser.error("--depth must be positive.")

    config.update("jax_enable_x64", args.precision == "float64")

    train_data = ShiftedTensorDataset(args.train)
    val_data = ShiftedTensorDataset(args.val)
    test_data = ShiftedTensorDataset(args.test)
    validate_dataset_compatibility(train_data, val_data, "val")
    validate_dataset_compatibility(train_data, test_data, "test")

    stiffness_eigs = spectral_stiffness_eigs(
        train_data.grid_shape,
        train_data.kh_max,
        dtype=jnp.float32,
    )

    model = ShiftedUNet3D(base_channels=args.base_channels, depth=args.depth)
    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros(
        (1, len(INPUT_CHANNELS), *train_data.grid_shape),
        dtype=jnp.float32,
    )
    params = model.init(rng, dummy)["params"]
    tx = optax.adamw(args.learning_rate, weight_decay=args.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_config(args.checkpoint_dir / "config.json", args, train_data)

    wandb_config = {
        **serializable_args(args),
        "train_shape": train_data.shape,
        "val_shape": val_data.shape,
        "test_shape": test_data.shape,
        "training_channels": TRAINING_CHANNELS,
        "input_channels": INPUT_CHANNELS,
        "target_channels": TARGET_CHANNELS,
    }
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=wandb_config,
    )

    best_val = float("inf")
    start_time = time.perf_counter()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = []
        for batch in train_data.batches(
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            state, metrics = train_step(
                state,
                jnp.asarray(batch),
                stiffness_eigs,
                args.supervised_weight,
                args.physics_weight,
            )
            global_step += 1
            epoch_metrics.append(jax.device_get(metrics))
            if global_step % args.log_every == 0:
                wandb.log(prefix_metrics(metrics, "train_step"), step=global_step)

        train_metrics = average_metrics(epoch_metrics)
        val_metrics = evaluate(
            state,
            val_data,
            stiffness_eigs,
            args.batch_size,
            args.supervised_weight,
            args.physics_weight,
        )
        log_payload = {
            **prefix_metrics(train_metrics, "train"),
            **prefix_metrics(val_metrics, "val"),
            "epoch": epoch,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        wandb.log(log_payload, step=global_step)
        print(format_epoch(epoch, train_metrics, val_metrics), flush=True)

        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            save_checkpoint(args.checkpoint_dir / "best.msgpack", state.params)
        save_checkpoint(args.checkpoint_dir / "latest.msgpack", state.params)

    test_metrics = evaluate(
        state,
        test_data,
        stiffness_eigs,
        args.batch_size,
        args.supervised_weight,
        args.physics_weight,
    )
    wandb.log(prefix_metrics(test_metrics, "test"), step=global_step)
    print("test " + format_metrics(test_metrics), flush=True)
    save_checkpoint(args.checkpoint_dir / "final.msgpack", state.params)
    run.finish()


@jax.jit
def train_step(state, batch, stiffness_eigs, supervised_weight, physics_weight):
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, model_inputs(batch))
        loss, metrics = shifted_loss(
            pred,
            batch,
            stiffness_eigs,
            supervised_weight,
            physics_weight,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return state, metrics


@jax.jit
def eval_step(state, batch, stiffness_eigs, supervised_weight, physics_weight):
    pred = state.apply_fn({"params": state.params}, model_inputs(batch))
    loss, metrics = shifted_loss(
        pred,
        batch,
        stiffness_eigs,
        supervised_weight,
        physics_weight,
    )
    return {**metrics, "loss": loss}


def model_inputs(batch):
    return batch[:, jnp.asarray(INPUT_CHANNELS), ...]


def shifted_loss(pred, batch, stiffness_eigs, supervised_weight, physics_weight):
    target = batch[:, jnp.asarray(TARGET_CHANNELS), ...]
    pred_complex = pred[:, 0] + 1j * pred[:, 1]
    target_complex = target[:, 0] + 1j * target[:, 1]

    target_error = jnp.mean(jnp.abs(pred_complex - target_complex) ** 2)
    target_scale = jnp.mean(jnp.abs(target_complex) ** 2) + EPS
    supervised_rel_mse = target_error / target_scale

    shifted_residual = shifted_system_residual(pred_complex, batch, stiffness_eigs)
    rhs_complex = batch[:, 2] + 1j * batch[:, 3]
    residual_error = jnp.mean(jnp.abs(shifted_residual) ** 2)
    residual_scale = jnp.mean(jnp.abs(rhs_complex) ** 2) + EPS
    physics_rel_mse = residual_error / residual_scale

    loss = supervised_weight * supervised_rel_mse + physics_weight * physics_rel_mse
    metrics = {
        "supervised_rel_mse": supervised_rel_mse,
        "physics_rel_mse": physics_rel_mse,
        "target_rel_l2": jnp.sqrt(supervised_rel_mse),
        "shifted_residual_rel_l2": jnp.sqrt(physics_rel_mse),
    }
    return loss, metrics


def shifted_system_residual(v, batch, stiffness_eigs):
    mass = batch[:, 0]
    damping = batch[:, 1]
    rhs = batch[:, 2] + 1j * batch[:, 3]
    shift = batch[:, 8, 0, 0, 0] + 1j * batch[:, 9, 0, 0, 0]
    av = (
        jnp.fft.ifftn(
            stiffness_eigs[None, ...] * jnp.fft.fftn(v, axes=(1, 2, 3)),
            axes=(1, 2, 3),
        )
        - (mass + 1j * damping) * v
    )
    return av - shift[:, None, None, None] * v - rhs


def evaluate(
    state,
    dataset: ShiftedTensorDataset,
    stiffness_eigs,
    batch_size: int,
    supervised_weight: float,
    physics_weight: float,
):
    metrics = []
    for batch in dataset.batches(batch_size, shuffle=False, seed=0):
        metrics.append(
            jax.device_get(
                eval_step(
                    state,
                    jnp.asarray(batch),
                    stiffness_eigs,
                    supervised_weight,
                    physics_weight,
                )
            )
        )
    return average_metrics(metrics)


def average_metrics(metrics):
    out = {}
    for key in metrics[0]:
        out[key] = float(np.mean([float(m[key]) for m in metrics]))
    return out


def prefix_metrics(metrics, prefix: str):
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def format_epoch(epoch: int, train_metrics, val_metrics) -> str:
    return (
        f"epoch={epoch} "
        f"train({format_metrics(train_metrics)}) "
        f"val({format_metrics(val_metrics)})"
    )


def format_metrics(metrics) -> str:
    return " ".join(f"{key}={float(value):.6e}" for key, value in metrics.items())


def save_checkpoint(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def write_config(path: Path, args, train_data: ShiftedTensorDataset) -> None:
    payload = {
        **serializable_args(args),
        "train_shape": train_data.shape,
        "train_dtype": str(train_data.dtype),
        "training_channels": TRAINING_CHANNELS,
        "input_channels": INPUT_CHANNELS,
        "target_channels": TARGET_CHANNELS,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def serializable_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_dataset_compatibility(
    reference: ShiftedTensorDataset,
    other: ShiftedTensorDataset,
    name: str,
) -> None:
    if reference.shape[1:] != other.shape[1:]:
        raise ValueError(
            f"{name} dataset shape {other.shape} is incompatible with train shape {reference.shape}."
        )
    if abs(reference.kh_max - other.kh_max) > 1e-6:
        raise ValueError(
            f"{name} kh_max={other.kh_max} does not match train kh_max={reference.kh_max}."
        )


if __name__ == "__main__":
    main()
