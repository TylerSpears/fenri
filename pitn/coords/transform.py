# -*- coding: utf-8 -*-
import functools
from typing import Tuple

import einops
import numpy as np
import torch

import pitn


@functools.lru_cache(maxsize=8)
def fraction_downsample_idx_weights(
    source_shape: tuple, source_vox_size: float, target_vox_size: float
):

    target_shape = np.floor(
        np.asarray(source_shape) * source_vox_size / target_vox_size
    ).astype(int)

    # Generate points for sampling along each axis.
    xs = (
        np.arange(
            0,
            (source_shape[0] * source_vox_size),
            target_vox_size,
        )
        / source_vox_size
    )

    ys = (
        np.arange(
            0,
            (source_shape[1] * source_vox_size),
            target_vox_size,
        )
        / source_vox_size
    )

    zs = (
        np.arange(
            0,
            (source_shape[2] * source_vox_size),
            target_vox_size,
        )
        / source_vox_size
    )

    # Remove any patches that would go out of bounds in the source image.
    xs = xs[: target_shape[0]]
    ys = ys[: target_shape[1]]
    zs = zs[: target_shape[2]]

    downsample_factor = target_vox_size / source_vox_size
    target_window_size = downsample_factor

    window_size = max(
        np.concatenate(
            [
                np.ceil(xs[1:]) - np.floor(xs[:-1]),
                np.ceil(ys[1:]) - np.floor(ys[:-1]),
                np.ceil(zs[1:]) - np.floor(zs[:-1]),
            ]
        ).astype(int)
    )

    g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"))
    g_iter = g.transpose(1, 2, 3, 0)

    full_indices = np.zeros(g.shape[1:] + (3,) + (window_size,) * 3, dtype=int)
    full_weights = np.zeros(g.shape[1:] + (window_size,) * 3, dtype=float)

    for i in np.ndindex(*g_iter.shape[:-1]):

        target_start_index = g_iter[i]
        target_end_index = target_start_index + target_window_size

        source_start_index = np.floor(target_start_index).astype(int)
        source_end_index = source_start_index + window_size

        ranges = list()
        for i_start, i_end in zip(source_start_index, source_end_index):
            ranges.append(np.arange(i_start, i_end))

        window_idx = np.stack(np.meshgrid(*ranges))

        vol_intersections = np.ones(window_idx.shape[1:])
        for dim_i in range(window_idx[0].ndim):

            axis_intersect = np.clip(
                np.min(
                    [
                        np.broadcast_to(
                            target_end_index[dim_i], window_idx[dim_i].shape
                        ),
                        window_idx[dim_i] + 1,
                    ],
                    axis=0,
                )
                - np.max(
                    [
                        np.broadcast_to(
                            target_start_index[dim_i], window_idx[dim_i].shape
                        ),
                        window_idx[dim_i],
                    ],
                    axis=0,
                ),
                0,
                np.inf,
            )

            vol_intersections = vol_intersections * axis_intersect

        weights = vol_intersections / vol_intersections.sum()

        full_weights[i] = weights
        full_indices[i] = window_idx

    full_indices = full_indices.reshape(
        *full_indices.shape[0:3],
        3,
        -1,
    )
    full_indices = full_indices.transpose(3, 4, 0, 1, 2)
    full_weights = full_weights.reshape(*full_weights.shape[0:3], -1)
    full_weights = full_weights.transpose(3, 0, 1, 2)

    return full_indices, full_weights


def int_downscale_patch_idx(
    idx: Tuple[torch.Tensor], downscale_factor: int, downscale_patch_shape: Tuple
):
    if not torch.is_tensor(idx):
        ndim_idx = torch.stack(idx, dim=0)
    else:
        ndim_idx = idx
    n_spatial_dims = len(downscale_patch_shape)
    span_dims = ndim_idx.shape[2:-n_spatial_dims]
    space_names = [f"space_{i}" for i in range(n_spatial_dims)]
    span_names = [f"span_{i}" for i in range(len(span_dims))]
    start_idx = einops.reduce(
        ndim_idx, f'ndim swatch ... {" ".join(space_names)} -> ndim swatch (...)', "min"
    )
    # Remove spanned dims
    start_idx = start_idx[len(span_names) :, :, 0]

    start_idx = torch.round(start_idx / downscale_factor).long()
    full_idx = pitn.utils.patch.extend_start_patch_idx(
        start_idx,
        patch_shape=downscale_patch_shape,
        span_extra_dims_sizes=tuple(span_dims),
    )

    return full_idx


def _map_fr_init_idx_to_lr(
    fr_idx, downsample_factor, low_res_sample_extension, fr_patch_size
) -> np.ndarray:
    """Maps starting indices of patches in FR space to LR space."""

    # Batch-ify the input, if not done already.
    if len(fr_idx.shape) == 1:
        batch_fr_idx = fr_idx.reshape(1, -1)
    else:
        batch_fr_idx = fr_idx

    # Find offset between the original FR patch index and the "oversampled" FR patch
    # index.
    idx_delta = np.round(
        (
            np.round(np.asarray(fr_patch_size) * low_res_sample_extension)
            - np.asarray(fr_patch_size)
        )
        / 2
    )
    expanded_fr_idx = batch_fr_idx - idx_delta

    # Now perform simple downscaling.
    lr_idx = np.floor(expanded_fr_idx / downsample_factor)

    # Undo batch-ification, if necessary.
    if len(fr_idx.shape) == 1:
        lr_idx = lr_idx.squeeze()

    return lr_idx.astype(int)


def _map_fr_coord_to_lr(
    fr_coords, downsample_factor, low_res_sample_extension, fr_patch_size
):
    fr_index_ini = fr_coords[:, :3]
    lr_index_ini = _map_fr_init_idx_to_lr(
        fr_index_ini,
        downsample_factor=downsample_factor,
        low_res_sample_extension=low_res_sample_extension,
        fr_patch_size=fr_patch_size,
    )
    lr_lengths = np.floor(
        np.round(np.asarray(fr_patch_size) * low_res_sample_extension)
        / downsample_factor
    )
    lr_coord = np.concatenate([lr_index_ini, lr_index_ini + lr_lengths], axis=-1)

    return lr_coord
