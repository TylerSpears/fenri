# -*- coding: utf-8 -*-

import einops
import numpy as np
import torch
import torch.nn.functional as F

import pitn


# world_extent = (affine[:3, :3] @ vox_grid.T) + (affine[:3, 3:4])
def coord_transform_3d(coords: torch.Tensor, affine_a2b: torch.Tensor) -> torch.Tensor:
    # For now, this function only accepts coords that are 1 or 2-dimensional
    if coords.ndim == 1 and affine_a2b.ndim > 2:
        new_shape = (1,) * (affine_a2b.ndim - 2)
        new_shape = new_shape + (-1,)
        c = coords.expand(*new_shape)
        affine = affine_a2b
    elif coords.ndim > 1 and affine_a2b.ndim == 2:
        new_shape = (1,) * (coords.ndim - 1)
        new_shape = new_shape + (-1, -1)
        affine = affine_a2b.expand(*new_shape)
        c = coords
    else:
        c = coords
        affine = affine_a2b

    if c.ndim == 1:
        c = c[None]
    if affine.ndim == 2:
        affine = affine[None]

    c = c.to(torch.result_type(c, affine))
    affine = affine.to(torch.result_type(c, affine))
    p = einops.einsum(affine[..., :3, :3], c, "... i j, ... j -> ... i")
    p = p + affine[..., :3, -1]

    return p.reshape(coords.shape)


def sample_3d(
    vol: torch.Tensor,
    coords_mm_zyx: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
) -> torch.Tensor:

    if vol.ndim == 3:
        v = vol[None, None]
    elif vol.ndim == 4:
        v = vol[None]
    elif vol.ndim == 5 and vol.shape[0] != 1:
        raise ValueError(
            f"ERROR: Volume must only have batch size 1, got {vol.shape[0]}"
        )
    else:
        v = vol

    # Construct an affine transformation that maps from voxel space to [-1, 1].
    spatial_shape = v.shape[-3:]
    aff_vox2grid = torch.eye(4).to(affine_vox2mm)
    # Scale all coordinates to be in range [0, 2].
    aff_diag = 2 / (torch.as_tensor(spatial_shape) - 1)
    aff_diag = torch.cat([aff_diag, aff_diag.new_ones(1)], 0)
    aff_vox2grid = aff_vox2grid.diagonal_scatter(aff_diag)
    # Translate coords "back" by 1 arbitrary unit to make all coords within range
    # [-1, 1], rather than [0, 2].
    aff_vox2grid[:3, 3:4] = -1

    # Invert the typical vox->mm affine.
    aff_mm2vox = torch.linalg.inv(affine_vox2mm)
    # Merge transformations to map mm to normalized grid space.
    aff_mm2grid = aff_vox2grid @ aff_mm2vox
    grid_coords = coord_transform_3d(coords_mm_zyx, aff_mm2grid)

    # Reverse the order of the coordinate dimension to work properly with grid_sample!
    grid_coords = torch.flip(grid_coords, dims=(-1,))
    # We don't want a batch of samples coming from different sources, so set the batch
    # size to 1 and move all samples to an arbitrary spatial dimension.
    grid = einops.rearrange(grid_coords, "... coord -> 1 1 (...) 1 coord", coord=3)
    if not torch.is_floating_point(vol):
        v = v.to(torch.promote_types(torch.float32, grid.dtype))
    samples = F.grid_sample(
        v, grid=grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )
    point_shape = tuple(coords_mm_zyx.shape[:-1])
    # If the volume contained channels, account for those.
    if vol.ndim > 3:
        point_shape = (v.shape[-4],) + point_shape

    samples = samples.reshape(*point_shape)
    if vol.ndim > 3:
        samples = samples.movedim(0, -1)
    # Change samples back to the input dtype only if the input was not a floating point,
    # and the interpolation was nearest-neighbor.
    if not torch.is_floating_point(vol) and mode == "nearest":
        samples = samples.round().to(vol.dtype)
    return samples
