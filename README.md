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

For debugging or metadata-rich exports, the older `--save-shifted-samples`
option still writes one `medium_*.npz` file and one `shifted_sample_*.npz`
file per captured pole/refinement step.
