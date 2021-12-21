# -*- coding: utf-8 -*-
import math
from pathlib import Path

from pitn._lazy_loader import LazyLoader

import numpy as np
import torch

monai = LazyLoader("monai", globals(), "monai")
GPUtil = LazyLoader("GPUtil", globals(), "GPUtil")
tabulate = LazyLoader("tabulate", globals(), "tabulate")


def get_gpu_specs():
    """Return string describing GPU specifications.

    Taken from
    <https://www.thepythoncode.com/article/get-hardware-system-information-python>.

    Returns
    -------
    str
        Human-readable string of specifications.
    """

    gpus = GPUtil.getGPUs()
    specs = list()
    specs.append("".join(["=" * 50, "GPU Specs", "=" * 50]))
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        driver_version = gpu.driver
        cuda_version = torch.version.cuda
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_uuid = gpu.uuid
        list_gpus.append(
            (
                gpu_id,
                gpu_name,
                driver_version,
                cuda_version,
                gpu_total_memory,
                gpu_uuid,
            )
        )

    table = tabulate.tabulate(
        list_gpus,
        headers=(
            "id",
            "Name",
            "Driver Version",
            "CUDA Version",
            "Total Memory",
            "uuid",
        ),
    )

    specs.append(table)

    return "\n".join(specs)


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
    """Yields chunks of patch coordinates for indexing into large tensors.

    "Swatches" are just batches of patches contained within the same tensor. Because
    swatches are cloth-related. Like patches.

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
    lower_bounds = np.asarray(spatial_shape)
    lower_bounds = lower_bounds - (np.asarray(patch_shape) - 1)
    idx_grid = np.meshgrid(
        *[np.arange(0, lb, st) for (lb, st) in zip(lower_bounds, stride)], copy=False
    )
    # Combine and reshape to be (n_patches, ndim).
    idx_grid = np.stack(idx_grid, -1).reshape(-1, ndim).astype(np.uint16)
    n_patches = idx_grid.shape[0]

    patch_range_broadcast = np.indices(patch_shape).reshape(ndim, -1)
    for swatch_start_idx in range(0, n_patches, max_swatch_size):
        swatch_size = min(max_swatch_size, n_patches - swatch_start_idx)
        swatch_end_idx = swatch_start_idx + swatch_size
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

    idx_gen = swatched_patch_coords_iter(
        spatial_shape=spatial_shape,
        patch_shape=patch_shape,
        stride=stride,
        max_swatch_size=max_swatch_size,
    )

    for swatched_idx in idx_gen:
        batch_channel_swatch = list()
        for b in range(batch_size):
            for c in range(channel_size):
                batch_channel_swatch.append(im[b, c][swatched_idx])
        batch_channel_swatch = torch.stack(batch_channel_swatch, dim=0)
        batch_channel_swatch = batch_channel_swatch.view(
            batch_size, channel_size, *batch_channel_swatch.shape[1:]
        )

        yield batch_channel_swatch, swatched_idx


def conv3d_out_shape(
    out_channels,
    input_shape,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
):
    in_shape = np.asarray(input_shape)
    kernel = np.asarray(kernel_size)
    stride = np.asarray(stride)
    pad = np.asarray(padding)
    dilate = np.asarray(dilation)

    spatial = np.floor(((in_shape + 2 * pad - dilate * (kernel - 1) - 1) / stride) + 1)

    return (out_channels,) + tuple(spatial)


def get_file_glob_unique(root_path: Path, glob_pattern: str) -> Path:
    root_path = Path(root_path)
    glob_pattern = str(glob_pattern)
    files = list(root_path.glob(glob_pattern))

    invalid_num_files = False
    if len(files) == 0:
        files = list(root_path.rglob(glob_pattern))
        if len(files) != 1:
            invalid_num_files = True
    elif len(files) > 1:
        invalid_num_files = True

    if invalid_num_files:
        raise RuntimeError(
            "ERROR: More than one file matches glob pattern "
            + f"{glob_pattern} under directory {str(root_path)}."
        )

    return files[0]
