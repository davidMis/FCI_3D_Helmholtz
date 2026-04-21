# 3D_Helmholtz_solver

Fast contour integration (FCI) method for solving the Helmholtz equation

Developers: Xiao Liu, Yuanzhe Xi

Consider the linear system (A-M)u=f, where A is the discrete Laplacian (here using FFT or finite difference), M is the potential (kh^2). The contour integration approach combines the solutions of multiple shifted systems (A-zM)u = f. The imaginary part of z makes the shifted systems easy to solve. A fixed-point iteration is used to solve shifted problems.

This repository contains Matlab and C code.

## Shifted-solve sample logging

The fast spectral JAX example can save training pairs for a neural shifted-solver
surrogate. Each sample records the residual input `r`, the shifted-solve target
`v ~= (A - zI)^-1 r`, and the diagnostic shifted residual `(A - zI)v - r`.

Example:

```bash
python3 examples/generate_grf.py \
  --n 64 \
  --seed 0 \
  --batch-size 8 \
  --precision float32

python3 examples/solve_grf_helmholtz.py \
  --n 64 \
  --seed 0 \
  --batch-size 8 \
  --precision float32 \
  --fast-spectral \
  --inner-solver none \
  --max-refinement-steps 10 \
  --training-data-out data/shifted_training_64_seed0_batch8.npy \
  --training-data-only
```

The batched generator writes one GRF file with shape `(batch, n, n, n)`.
The solver can write one training tensor with shape
`(samples, channels, n, n, n)`. The channel order is printed at runtime and is:
`mass`, `damping`, `residual_real`, `residual_imag`,
`shifted_solution_real`, `shifted_solution_imag`, `shifted_residual_real`,
`shifted_residual_imag`, `shift_real`, `shift_imag`, `kh_max`,
`relres_before`, `step`, `pole_index`.

Use `--training-data-only` to suppress the regular pressure, wavespeed,
residual-history, and per-sample side files while generating the tensor.

To generate training data without writing intermediate GRF files, use the
one-stage random-step generator:

```bash
python3 examples/generate_random_step_training_data.py \
  --n 64 \
  --num-grfs 8 \
  --seed 0 \
  --precision float32 \
  --inner-solver none \
  --max-refinement-steps 10 \
  --output data/shifted_training_64_seed0_random_steps.npy
```

This writes one tensor with shape `(num_grfs, channels, n, n, n)`. Each GRF is
generated in memory, one random refinement step and pole are captured, and the
GRF is discarded before the next sample.

For debugging or metadata-rich exports, the older `--save-shifted-samples`
option still writes one `medium_*.npz` file and one `shifted_sample_*.npz`
file per captured pole/refinement step.

## Shifted-solver surrogate training

Train the first JAX/Flax 3D U-Net surrogate with W&B logging:

```bash
python3 examples/train_shifted_surrogate.py \
  --train data/train_shifted_64.npy \
  --val data/val_shifted_64.npy \
  --test data/test_shifted_64.npy \
  --checkpoint-dir checkpoints/shifted_unet_64 \
  --epochs 50 \
  --batch-size 1 \
  --base-channels 16 \
  --depth 3 \
  --learning-rate 1e-4 \
  --physics-weight 1.0 \
  --wandb-project fci-shifted-surrogate
```

The model predicts `shifted_solution_real` and `shifted_solution_imag`
directly. The loss combines relative supervised error and the shifted-system
physics residual `||(A - zI)v_pred - r|| / ||r||`, using the spectral
Helmholtz operator in JAX.
