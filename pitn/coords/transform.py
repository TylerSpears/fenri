# -*- coding: utf-8 -*-
import functools

import numpy as np


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
