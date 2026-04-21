"""Neural shifted-solver surrogates."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from .training_data import TRAINING_CHANNELS

INPUT_CHANNELS = (0, 1, 2, 3, 8, 9, 10, 11, 12, 13)
TARGET_CHANNELS = (4, 5)


class ConvBlock3D(nn.Module):
    """Small 3D convolutional block for channel-last tensors."""

    features: int
    groups: int = 8

    @nn.compact
    def __call__(self, x):
        groups = _group_count(self.features, self.groups)
        x = nn.Conv(self.features, kernel_size=(3, 3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=groups)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3, 3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=groups)(x)
        return nn.gelu(x)


class ShiftedUNet3D(nn.Module):
    """Compact 3D U-Net predicting real/imag shifted-solve fields."""

    base_channels: int = 16
    depth: int = 3
    out_channels: int = 2

    @nn.compact
    def __call__(self, x):
        # Training tensors are channel-first: (batch, channels, nx, ny, nz).
        x = jnp.moveaxis(x, 1, -1)
        skips = []
        features = self.base_channels

        for level in range(self.depth):
            x = ConvBlock3D(features, name=f"enc_{level}")(x)
            skips.append(x)
            x = nn.Conv(
                features * 2,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                padding="SAME",
                name=f"down_{level}",
            )(x)
            x = nn.gelu(x)
            features *= 2

        x = ConvBlock3D(features, name="bottleneck")(x)

        for level in reversed(range(self.depth)):
            features //= 2
            x = nn.ConvTranspose(
                features,
                kernel_size=(2, 2, 2),
                strides=(2, 2, 2),
                padding="SAME",
                name=f"up_{level}",
            )(x)
            skip = skips[level]
            x = _match_spatial_shape(x, skip.shape[1:4])
            x = jnp.concatenate([x, skip], axis=-1)
            x = ConvBlock3D(features, name=f"dec_{level}")(x)

        x = nn.Conv(self.out_channels, kernel_size=(1, 1, 1), name="out")(x)
        return jnp.moveaxis(x, -1, 1)


def _match_spatial_shape(x, target_shape: Sequence[int]):
    slices = tuple(slice(0, target_shape[axis]) for axis in range(3))
    x = x[:, slices[0], slices[1], slices[2], :]
    pad_width = [(0, 0)]
    for axis, target in enumerate(target_shape):
        pad_width.append((0, max(0, target - x.shape[axis + 1])))
    pad_width.append((0, 0))
    return jnp.pad(x, pad_width)


def _group_count(features: int, max_groups: int) -> int:
    for groups in range(min(features, max_groups), 0, -1):
        if features % groups == 0:
            return groups
    return 1
