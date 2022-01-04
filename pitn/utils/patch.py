# -*- coding: utf-8 -*-
import math
import collections
from typing import Union, Sequence, Optional, List, Tuple

from pitn._lazy_loader import LazyLoader

import numpy as np
import torch
import einops

monai = LazyLoader("monai", globals(), "monai")

SwatchIdxSample = collections.namedtuple("SwatchIdxSample", ("full_idx", "start_idx"))
SwatchSample = collections.namedtuple(
    "SwatchSample", ("swatch", "full_idx", "start_idx")
)


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


def extend_start_patch_idx(
    patch_start_idx, patch_shape, span_extra_dims_sizes: Tuple[int] = tuple()
):
    """Extend N-dim patch indices across the size of the patch.

    Final output will be a tuple of Tensors whose values comprise a set of patches.
    This tuple will have length `N_dim`, with each entry being a Tensor of size:

    swatch_size x (spanned_extra_dim_1 x spanned_extra_dim_2 x ...) x (spatial_dim_1 x spatial_dim_2 x ...)

    Only the spatial dims will be reduced through indexing; extra spanned dims will be
    fully spanned by the values in the indices.

    Here, 'N_dim' is the total number of dimensions for each patch in the output, which
    is `len(span_extra_dims_sizes) + len(patch_start_idx)`.

    Parameters
    ----------
    patch_start_idx : tuple
        Tuple containing the starting index of a swatch of patches.

        Has length N_spatial_dims, each element is a 1D Tensor that has a length of
        "num swatches".

    patch_shape : tuple
        Shape of each patch in the spatial dimensions.

    span_extra_dims_sizes: tuple
        Tuple of dims and sizes that each patch fully spans, by default empty tuple()

        Typically used to index into non-spatial dimensions, such as a channel dimension.

        These dims *will* be included in the full output coordinates/index, where each
        patch will index into the entire dimension(s). These dims will be prefixed to
        the shape of the output indices.
    """

    n_span_dims = len(span_extra_dims_sizes)
    n_spatial_dims = len(patch_start_idx)
    # Define the number of unique patches (each of which will be the seed for a swatch)
    # as the number of elements that were given starting indices.
    n_swatch = patch_start_idx[0].numel()

    spatial_patch_shape = monai.utils.misc.ensure_tuple_rep(patch_shape, n_spatial_dims)

    if not torch.is_tensor(patch_start_idx):
        start_idx = torch.stack(patch_start_idx, dim=0)
    else:
        start_idx = patch_start_idx
    if n_span_dims > 0:
        start_span_dims = torch.zeros(n_span_dims, *start_idx.shape[1:]).to(start_idx)
        start_idx = torch.concat([start_span_dims, start_idx], dim=0)

    ndim = n_span_dims + n_spatial_dims
    full_patch_shape = span_extra_dims_sizes + spatial_patch_shape
    # Expand the patch starting idx to encompass all elements in the patch. This works
    # by emulating an n-dim 'arange()' via broadcasting addition of index values.
    # Use `np.indices` as the addends for the start_idx coordinates.

    # Reshape to ndim x [n_swatch * spanned dim sizes * patch dim sizes]
    patch_range_broadcast = torch.from_numpy(
        np.indices(full_patch_shape).reshape(ndim, -1)
    )
    # Reorder to be n_swatch x ndim, so the ndim-sized dimension can be matched for
    # broadcasting with the `patch_range_broadcast`.
    start_idx = einops.rearrange(start_idx, "ndim patch_elems -> patch_elems ndim 1")
    # Perform the expansion of indices by broadcasting over an extra dimension in the
    # start_idx.
    full_swatch_idx = start_idx + patch_range_broadcast

    # Reorder to ndim x ..., and reshape to be
    # ndim x n_swatch x (spanned dim_1 x spanned_dim 2 x ... x spatial dim_1 x spatial dim_2 x ...)
    # Group the n-dimensional patch dims into a dynamically-sized group of names & their
    # respective lengths.
    full_patch_names_map = {
        f"patch_dim_{i}": size for (i, size) in enumerate(full_patch_shape)
    }
    # String of patch dim names as "patch_dim_0 patch_dim_1 patch_dim_2 ..."
    patch_dim_group_names = " ".join(full_patch_names_map.keys())
    full_swatch_idx = einops.rearrange(
        full_swatch_idx,
        f"s ndim ({patch_dim_group_names}) -> ndim s {patch_dim_group_names}",
        **full_patch_names_map,
    )

    return tuple(full_swatch_idx.long())


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

    # Create a multi-dimensional index array that contains the first (lowest,
    # left-most, etc.) element of every patch in the swatch.
    idx_grid = np.meshgrid(
        *[np.arange(0, lb, st) for (lb, st) in zip(lower_bounds, stride)], copy=False
    )
    # Combine and reshape to be (n_patches, ndim).
    idx_grid = einops.rearrange(idx_grid, "ndim ... -> (...) ndim").astype(np.int16)
    n_patches = idx_grid.shape[0]

    # Iterate over swatches at most of size 'max_swatch_size'.
    for swatch_start_idx in range(0, n_patches, max_swatch_size):
        swatch_size = min(max_swatch_size, n_patches - swatch_start_idx)
        swatch_end_idx = swatch_start_idx + swatch_size
        # Select only patches in this swatch.
        patch_start_idx = idx_grid[swatch_start_idx:swatch_end_idx]
        # Convert to a tuple of size N_dim of swatched patches.
        patch_start_idx = tuple(torch.from_numpy(patch_start_idx.swapaxes(0, 1)))

        # Expand the start_idx to the full extent of the patch.
        full_swatch_idx = extend_start_patch_idx(patch_start_idx, patch_shape)
        yield SwatchIdxSample(full_idx=full_swatch_idx, start_idx=patch_start_idx)


def batched_patches_iter(
    ims: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    patch_shape,
    stride,
    max_swatch_size: int,
):
    """Yields batched patches and patch indices.

    Parameters
    ----------
    ims : Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
        N-dimensional tensor or sequence of N-dimensional tensors to index into.

        Must be of shape [B, C, dim_1, dim_2, ...], where B is the batch dimension and C
        is the channel dimension, both of which are *not* reduced by indexing. If a
        (non-Tensor) sequence of tensors is given, then each element is assumed to be
        a Tensor with the *same indexing dimension size*; i.e., B and C may be different,
        but dim_1, dim_2, ..., must be the same.
    patch_shape : tuple

    stride : tuple

    max_batch_size : int

    """
    if torch.is_tensor(ims):
        ims = (ims,)
        yield_tensor = True
    else:
        yield_tensor = False
    spatial_shape = tuple(ims[0].shape[2:])

    # Create the swatch generator.
    idx_gen = swatched_patch_coords_iter(
        spatial_shape=spatial_shape,
        patch_shape=patch_shape,
        stride=stride,
        max_swatch_size=max_swatch_size,
    )

    batch_sizes = tuple(im.shape[0] for im in ims)
    channel_sizes = tuple(im.shape[1] for im in ims)
    # Iterate over all valid patches as swatches.
    for full_swatch_idx, start_idx in idx_gen:
        yield_ims = list()
        for im, batch_size, channel_size in zip(ims, batch_sizes, channel_sizes):
            # We must index into the B x C x [...] tensor B * C times for every swatch to
            # keep the swatch generation code manageable. So, keep a list of retrieved
            # patches for every (B x C) and stack them at the end of each swatch iteration.
            batch_channel_swatch = list()
            # Iterate over all batches.
            for b in range(batch_size):
                # Iterate over all channels.
                for c in range(channel_size):
                    batch_channel_swatch.append(im[b, c][full_swatch_idx])
            batch_channel_swatch = einops.rearrange(
                batch_channel_swatch,
                "(b c) ... -> b c ...",
                b=batch_size,
                c=channel_size,
            )
            yield_ims.append(batch_channel_swatch)

        yield_ims = tuple(yield_ims)
        if yield_tensor:
            yield_ims = yield_ims[0]
        # Output is of shape [B x C x Swatch x (patch_shape_1, patch_shape_2, ...)]
        # Swatched idx is a tuple with length N_spatial_dim, with each element indexing
        # into the corresponding spatial dimension with a shape of
        # [Swatch x (patch_shape_1, patch_shape_2, ...)].
        yield SwatchSample(
            swatch=yield_ims, full_idx=full_swatch_idx, start_idx=start_idx
        )
