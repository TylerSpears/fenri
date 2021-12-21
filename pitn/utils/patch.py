# -*- coding: utf-8 -*-
import math

from pitn._lazy_loader import LazyLoader

import numpy as np
import torch

monai = LazyLoader("monai", globals(), "monai")


def patch_center(
    patch: torch.Tensor, sub_sample_strategy="lower", keepdim=False
) -> torch.Tensor:
    """Extract 3D multi-channel patch center.

    Expects patch of shape '[B x C x] W x H x D'

    sub_sample_strategy: str
        Strategy for handling center coordinates of even-sized dimensions.
        Options:
            Strategies over indices:
                'lower': Take the voxel to the left of the center.
                'upper': Take the voxel to the right of the center.

            Strategies over multi-dimensional voxels:
                'max': Take the max of all center voxels.
                'min': Take the minimum of all center voxels.
                'mean': Take the average of all center voxels.
                'agg': Don't reduce at all, and return the center voxels.
    """
    strategy_fn = {
        "idx_fns": {
            "lower".casefold(): lambda i: int(i),
            "upper".casefold(): lambda i: int(i) + 1,
        },
        "vox_fns": {
            "max".casefold(): lambda p: torch.amax(p, dim=(-3, -2, -1), keepdim=True),
            "min".casefold(): lambda p: torch.amin(p, dim=(-3, -2, -1), keepdim=True),
            "mean".casefold(): lambda p: p.mean(dim=(-3, -2, -1), keepdim=True),
            "agg".casefold(): lambda p: p,
        },
    }

    strat = sub_sample_strategy.casefold()
    if (strat not in strategy_fn["idx_fns"].keys()) and (
        strat not in strategy_fn["vox_fns"].keys()
    ):
        raise ValueError(
            f"ERROR: Invalid strategy; got {sub_sample_strategy}, expected one of"
            + f"{list(strategy_fn['idx_fns'].keys()) + list(strategy_fn['vox_fns'].keys())}"
        )
    patch_spatial_shape = patch.shape[-3:]
    centers = torch.as_tensor(patch_spatial_shape) / 2
    center = list()
    for dim in centers:
        if int(dim) != dim:
            dim = slice(int(math.floor(dim)), int(math.ceil(dim)))
        elif strat in strategy_fn["idx_fns"].keys():
            dim = int(strategy_fn["idx_fns"][strat](dim))
            dim = slice(dim, dim + 1)
        elif strat in strategy_fn["vox_fns"].keys():
            dim = slice(int(dim), int(dim) + 2)
        else:
            raise ValueError("ERROR: Invalid strategy")
        center.append(dim)

    center_patches = patch[..., center[0], center[1], center[2]]

    if (
        center_patches.shape[-3:] != (1, 1, 1)
        and strat in strategy_fn["vox_fns"].keys()
    ):
        center_patches = strategy_fn["vox_fns"][strat](center_patches)

    if not keepdim:
        center_patches = center_patches.squeeze()

    return center_patches


def swatched_patch_coords_iter(
    spatial_shape, patch_shape, stride, max_swatch_size: int
):
    """Yields swatches (chunks) of patch coordinates for indexing into large tensors.

    "Swatches" are just batches of patches that come from the same tensor (image, volume,
    etc.). Because swatches are cloth-related. Like patches.

    Parameters
    ----------
    spatial_shape : tuple

    patch_shape : tuple

    stride : tuple

    max_swatch_size : int

    Yields
    -------
    [type]
        [description]
    """
    spatial_shape = tuple(spatial_shape)
    ndim = len(spatial_shape)

    patch_shape = monai.utils.misc.ensure_tuple_rep(patch_shape, ndim)
    stride = monai.utils.misc.ensure_tuple_rep(stride, ndim)
    # Cut off all patches that would go beyond the spatial size.
    lower_bounds = np.asarray(spatial_shape)
    lower_bounds = lower_bounds - (np.asarray(patch_shape) - 1)

    # Create a multi-dimensional index array that contains the first (top-most,
    # left-most, etc.) element of every patch in the swatch.
    idx_grid = np.meshgrid(
        *[np.arange(0, lb, st) for (lb, st) in zip(lower_bounds, stride)], copy=False
    )
    # Combine and reshape to be (n_patches, ndim).
    idx_grid = np.stack(idx_grid, -1).reshape(-1, ndim).astype(np.uint16)
    n_patches = idx_grid.shape[0]

    # Expand the patch starting idx to encompass all elements in the patch. This works
    # by emulating an n-dim 'arange()' via broadcasting addition of index values.
    patch_range_broadcast = np.indices(patch_shape).reshape(ndim, -1)
    # Iterate over swatches at most of size 'max_swatch_size'.
    for swatch_start_idx in range(0, n_patches, max_swatch_size):
        swatch_size = min(max_swatch_size, n_patches - swatch_start_idx)
        swatch_end_idx = swatch_start_idx + swatch_size
        # Select only patches in this swatch.
        patch_start_idx = idx_grid[swatch_start_idx:swatch_end_idx]

        # Expand the start_idx to the full extent of the patch.
        full_swatch_idx = torch.from_numpy(
            (patch_start_idx[..., None] + patch_range_broadcast)
            .swapaxes(0, 1)
            .reshape(ndim, swatch_size, *patch_shape)
        )
        yield tuple(full_swatch_idx)


def batched_patches_iter(im, patch_shape, stride, max_swatch_size: int):
    """Yields batched patches and patch indices.

    Parameters
    ----------
    im : torch.Tensor
        N-dimensional tensor to index into, channel-first.

        Must be of shape [B, C, dim_1, dim_2, ...], where B is the batch dimension and C
        is the channel dimension, both of which are *not* reduced by indexing.
    patch_shape : tuple

    stride : tuple

    max_batch_size : int

    """
    spatial_shape = tuple(im.shape[2:])
    batch_size = im.shape[0]
    channel_size = im.shape[1]

    # Create the swatch generator.
    idx_gen = swatched_patch_coords_iter(
        spatial_shape=spatial_shape,
        patch_shape=patch_shape,
        stride=stride,
        max_swatch_size=max_swatch_size,
    )

    # Iterate over all valid patches as swatches.
    for swatched_idx in idx_gen:
        # We must index into the B x C x [...] tensor B * C times for every swatch to
        # keep the swatch generation code manageable. So, keep a list of retrieved
        # patches for every (B x C) and stack them at the end of each swatch iteration.
        batch_channel_swatch = list()
        # Iterate over all batches.
        for b in range(batch_size):
            # Iterate over all channels.
            for c in range(channel_size):
                batch_channel_swatch.append(im[b, c][swatched_idx])
        batch_channel_swatch = torch.stack(batch_channel_swatch, dim=0)
        batch_channel_swatch = batch_channel_swatch.view(
            batch_size, channel_size, *batch_channel_swatch.shape[1:]
        )

        # Output is of shape [B x C x Swatch x (patch_shape_1, patch_shape_2, ...)]
        # Swatched idx is a tuple with length N_spatial_dim, with each element indexing
        # into the corresponding spatial dimension with a shape of
        # [Swatch x (patch_shape_1, patch_shape_2, ...)].
        yield batch_channel_swatch, swatched_idx
