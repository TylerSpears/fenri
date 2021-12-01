# -*- coding: utf-8 -*-
import math

import numpy as np
import torch


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
