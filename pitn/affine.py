# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F

import pitn


def affine_coordinate_grid(
    affine_vox2b: torch.Tensor, vox_fov_shape: Union[torch.Tensor, int, Tuple[int]]
) -> torch.Tensor:

    if torch.is_tensor(vox_fov_shape):
        fov_shape = tuple(vox_fov_shape.shape[-3:])
    elif isinstance(vox_fov_shape, int):
        fov_shape = (vox_fov_shape,) * 3
    else:
        fov_shape = tuple(vox_fov_shape)

    vox_coords = torch.stack(
        torch.meshgrid(
            *[
                torch.arange(d, dtype=affine_vox2b.dtype, device=affine_vox2b.device)
                for d in fov_shape
            ],
            indexing="ij",
        ),
        dim=-1,
    )

    return transform_coords(vox_coords, affine_vox2b)


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
    elif coords.ndim > affine_a2b.ndim + 1 and affine_a2b.ndim == 3:
        new_shape = (
            (affine_a2b.shape[0],)
            + ((1,) * (coords.ndim - 2))
            + tuple(affine_a2b.shape[1:])
        )
        affine = affine_a2b.view(*new_shape)
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


def _canonicalize_coords_3d_affine(
    coords_3d: torch.Tensor,
    affine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape and/or repeat tensor elements to a compatible shape.

    coords_3d : torch.Tensor
        Coordinates in a coordinate-last format with shape `[B] x [s_1, s2, s3] x 3`.

        The last dimension must always be size 3.
        If coords_3d is 1D, then it will be expanded according to the common batch
            size.
        If coords_3d is 2D, then the first dimension is assumed to be a batch size.
        If coords_3d is 3D, then the first dimension is assumed to be the batch size,
            and the second is assumed to be a `within-batch` sample dimension.
        If coords_3d is 4D, then the first 3 dimensions are assumed to be spatial
            dimensions, and the batch size is assumed to be 1.
        If coords_3D is 5D, then the first dimension is assumed to be the batch size,
            and the next 3 dimensions are assumed to be spatial dimensions that
            correspond to their respective batch indices.
        Otherwise, a RuntimeError is raised.

    affine : torch.Tensor
        Affine matrix (or matrices) of shape `[B] x 4 x 4`.

    Batch size will be determined as the broadcasted size of the batch sizes given
        in the shapes of `coords_3d` and `affine`. If the determined batch sizes are
        incompatible, a ValueError is raised. For example if coords_3d is `4 x 3`, and
        `affine` is `4 x 4`, then the batch size is set to `broadcast_shape(4, 1) = 4`.
        If `coords_3d` is `4 x 3` and `affine` is `2 x 4 x 4`, the batch sizes are
        incompatible, and an error is raised.
    """

    c = coords_3d
    if c.ndim < 1 or c.ndim > 5:
        raise RuntimeError(
            "ERROR: Expected coords_3d to have shape `[B] x [s_1, s2, s3] x 3`, but",
            f"got shape {tuple(c.shape)}.",
        )
    if c.shape[-1] != 3:
        raise RuntimeError(
            "ERROR: Expected coords_3d to have the last dimension be a coordinate",
            f"dimension of size 3, got {tuple(c.shape[-1])},",
            f"with full shape {tuple(c.shape)}",
        )

    # Add a batch dimension.
    if c.ndim == 1 or c.ndim == 4:
        c = torch.unsqueeze(c, 0)

    # Add or expand spatial dimensions, if necessary.
    if c.ndim == 2:
        c_spatial_shape = (1, 1, 1)
    elif c.ndim == 3:
        c_spatial_shape = (c.shape[1], 1, 1)
    else:
        c_spatial_shape = tuple(c.shape[1:-1])
    c = c.reshape(c.shape[0], *c_spatial_shape, c.shape[-1])
    batch_c = c.shape[0]

    a = affine
    if a.ndim < 2 or a.ndim > 3 or tuple(a.shape[-2:]) != (4, 4):
        raise RuntimeError(
            f"ERROR: Expected affine of shape `[B] x 4 x 4`, got {tuple(a.shape)}"
        )
    if a.ndim == 2:
        a = torch.unsqueeze(a, 0)

    batch_a = a.shape[0]

    if batch_c != 1 and batch_a != 1 and batch_c != batch_a:
        raise RuntimeError("ERROR: Batch sizes are not broadcastable")

    common_batch_size = max(batch_c, batch_a)
    if batch_c != batch_a:
        if batch_c == 1:
            c = einops.repeat(
                c, "1 s1 s2 s3 coord -> b s1 s2 s3 coord", b=common_batch_size
            )
        elif batch_a == 1:
            a = einops.repeat(
                a,
                "1 homog_aff_1 homog_aff_2 -> b homog_aff_1 homog_aff_2",
                b=common_batch_size,
            )

    c = c.to(torch.result_type(c, a))
    a = a.to(torch.result_type(c, a))

    return c, a


def transform_coords(coords_3d: torch.Tensor, affine_a2b: torch.Tensor) -> torch.Tensor:
    c, a = _canonicalize_coords_3d_affine(coords_3d, affine_a2b)
    # Expand the affine matrices to be broadcastable over the coordinate spatial
    # dimensions.
    a = a[:, None, None, None]

    p = einops.einsum(a[..., :3, :3], c, "... i j, ... j -> ... i")
    p += a[..., :3, -1]

    if c.numel() != coords_3d.numel():
        p = p.reshape(-1, *tuple(coords_3d.shape))
    else:
        p = p.reshape(coords_3d.shape)
    if torch.is_floating_point(coords_3d) and p.dtype != coords_3d.dtype:
        p = p.to(dtype=coords_3d.dtype)

    return p


def affine_vox2normalized_grid(
    spatial_fov_vox: tuple, lower: float, upper: float, to_tensor=None
) -> torch.Tensor:
    """Construct an affine transformation that maps from voxel space to [lower, upper]."""
    diff = upper - lower

    if to_tensor is not None:
        aff_vox2grid = torch.eye(4, dtype=to_tensor.dtype, device=to_tensor.device)
    else:
        aff_vox2grid = torch.eye(4, dtype=torch.float)
    # Scale all coordinates to be in range [0, diff].
    aff_diag = diff / (aff_vox2grid.new_tensor(spatial_fov_vox) - 1)
    aff_diag = torch.cat([aff_diag, aff_diag.new_ones(1)], 0)
    aff_vox2grid = aff_vox2grid.diagonal_scatter(aff_diag)
    # Translate coords "back" by `lower` arbitrary unit(s) to make all coords within
    # range [lower, upper], rather than [0, diff].
    aff_vox2grid[:3, 3:4] = lower

    return aff_vox2grid


def sample_vol(
    vol: torch.Tensor,
    coords_mm_xyz: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    override_out_of_bounds_val: Optional[float] = None,
) -> torch.Tensor:

    if vol.ndim == 3:
        v = vol[None, None]
    elif vol.ndim == 4:
        v = vol[None]
    else:
        v = vol
    assert v.ndim == 5

    if mode.strip().lower() == "trilinear" or mode.strip().lower() == "linear":
        mode = "bilinear"

    coords, aff_vox2mm = _canonicalize_coords_3d_affine(coords_mm_xyz, affine_vox2mm)
    # Invert the typical vox->mm affine.
    aff_mm2vox = torch.linalg.inv(aff_vox2mm)
    # Transform from vox to normalized grid coordinates in range [-1, 1] as required by
    # grid_sample().
    # spatial_fov_vox = tuple(vol.shape[1:-1])
    spatial_fov_vox = tuple(v.shape[2:])
    aff_vox2grid = affine_vox2normalized_grid(
        spatial_fov_vox, lower=-1.0, upper=1.0, to_tensor=aff_vox2mm
    )
    aff_vox2grid.unsqueeze_(0)
    # Merge transformations to map mm to normalized grid space, broadcasting the
    # vox2grid affine over the batches in the input.
    aff_mm2grid = torch.matmul(aff_vox2grid, aff_mm2vox)
    grid_coords = transform_coords(coords, aff_mm2grid)

    # Re-sample the volume with pytorch's grid_sample().
    if not torch.is_floating_point(vol):
        v = v.to(torch.promote_types(torch.float32, grid_coords.dtype))
    # Reverse the order of the coordinate dimension to work properly with grid_sample!
    grid_coords = torch.flip(grid_coords, dims=(-1,))
    samples = F.grid_sample(
        v,
        grid=grid_coords,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # If out-of-bounds samples should be overridden, set those sample values now.
    # Otherwise, the grid_sample() only interpolates with the padded values, which
    # allows out-of-bounds samples to appear to be in bounds.
    if override_out_of_bounds_val is not None:
        samples.masked_fill_(
            (grid_coords < -1).any(dim=-1)[:, None]
            | (grid_coords > 1).any(dim=-1)[:, None],
            override_out_of_bounds_val,
        )
    # Change samples back to the input dtype only if the input was not a floating point,
    # and the interpolation was nearest-neighbor.
    if not torch.is_floating_point(vol) and mode == "nearest":
        samples = samples.round().to(vol.dtype)

    # Un-canonicalize the sample result shape.
    if (
        (samples.ndim != coords_mm_xyz.ndim)
        or (samples.shape[0] != coords_mm_xyz.shape[0])
        or (samples.ndim != vol.ndim)
    ):
        undo_shape = list(samples.shape)
        b = samples.shape[0]
        if b == 1:
            if (
                coords_mm_xyz.ndim in (1, 4)
                and vol.ndim in (3, 4)
                and affine_vox2mm.ndim == 2
            ):
                undo_shape[0] = 0
        c = samples.shape[1]
        if c == 1 and vol.ndim == 3:
            undo_shape[1] = 0
        s = tuple(samples.shape[2:])
        if all(s_ == 1 for s_ in s) and coords_mm_xyz.ndim in (1, 2):
            for i in (2, 3, 4):
                undo_shape[i] = 0
        elif sum(s) == max(s) + 2:
            if coords_mm_xyz.ndim == 3:
                undo_shape[2] = coords_mm_xyz.shape[1]
                undo_shape[3] = 0
                undo_shape[4] = 0
        undo_shape = tuple(filter(lambda x: x > 0, undo_shape))
        samples = samples.reshape(undo_shape)

    return samples


def sample_3d(
    vol: torch.Tensor,
    coords_mm_zyx: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    override_out_of_bounds_val: Optional[float] = None,
) -> torch.Tensor:
    #!DEPRECATED
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
    # If out-of-bounds samples should be overridden, set those sample values now.
    # Otherwise, the grid_sample() only interpolates with the padded values, which
    # still makes them valid.
    if override_out_of_bounds_val is not None:
        samples.masked_fill_(
            (grid < -1).any(dim=-1)[:, None] | (grid > 1).any(dim=-1)[:, None],
            override_out_of_bounds_val,
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
