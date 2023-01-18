# -*- coding: utf-8 -*-
import collections
from typing import Callable, Optional, Tuple

import einops
import numpy as np
import torch

import pitn
import pitn.affine
import pitn.tract
import pitn.tract.local
import pitn.tract.peak

_SeedDirectionContainer = collections.namedtuple(
    "_PointContainer", ("origin", "theta", "phi")
)


def seeds_directions_from_peaks(
    max_peaks_per_voxel: int,
    seed_coords_mm: torch.Tensor,
    peaks: torch.Tensor,
    theta_peak: torch.Tensor,
    phi_peak: torch.Tensor,
    valid_peak_mask: torch.Tensor,
) -> _SeedDirectionContainer:

    topk = pitn.tract.peak.topk_peaks(
        max_peaks_per_voxel,
        peaks,
        theta_peak=theta_peak,
        phi_peak=phi_peak,
        valid_peak_mask=valid_peak_mask,
    )
    topk_valid = topk.valid_peak_mask
    seed_coord = einops.rearrange(seed_coords_mm, "... 3 -> (...) 3")
    k = topk.peaks.shape[-1]

    # Only keep the directions with valid peaks.
    seed_coord = einops.repeat(seed_coord, "b 3 -> b k 3", k=k)
    seed_coord = torch.masked_select(seed_coord, topk_valid[..., None])
    seed_theta = topk.theta[topk_valid]
    seed_phi = topk.phi[topk_valid]

    # Additionally, invert each peak to account for bi-polar symmetry.
    seed_theta = torch.cat([seed_theta, (seed_theta + torch.pi / 2) % torch.pi], dim=0)
    # Phi's range does not include -pi, so if phi is exactly pi, it must be rounded
    # the the next lowest value.
    seed_phi = torch.cat(
        [
            seed_phi,
            torch.clamp_min(-seed_phi, -torch.pi + torch.finfo(seed_phi.dtype).eps),
        ],
        dim=0,
    )
    seed_coord = torch.cat([seed_coord, seed_coord], dim=0)

    return _SeedDirectionContainer(origin=seed_coord, theta=seed_theta, phi=seed_phi)


def expand_seeds_from_topk_peaks_rk4(
    seed_coords_mm: torch.Tensor,
    max_peaks_per_voxel: int,
    seed_peak_vals: torch.Tensor,
    theta_peak: torch.Tensor,
    phi_peak: torch.Tensor,
    valid_peak_mask: torch.Tensor,
    step_size: float,
    fn_zyx_direction_t2theta_phi: Callable[
        [torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]
    ],
) -> Tuple[torch.Tensor, torch.Tensor]:

    topk = pitn.tract.peak.topk_peaks(
        max_peaks_per_voxel,
        seed_peak_vals,
        theta_peak=theta_peak,
        phi_peak=phi_peak,
        valid_peak_mask=valid_peak_mask,
    )
    topk_valid = topk.valid_peak_mask
    seed_coord = einops.rearrange(
        seed_coords_mm, "... zyx_coord -> (...) zyx_coord", zyx_coord=3
    )
    k = topk.peaks.shape[-1]

    # Only keep the directions with valid peaks.
    seed_coord = einops.repeat(
        seed_coord, "b zyx_coord -> b k zyx_coord", k=k, zyx_coord=3
    )
    seed_coord = seed_coord[topk_valid]
    seed_theta = topk.theta[topk_valid]
    seed_phi = topk.phi[topk_valid]

    # Additionally, invert each peak to account for bi-polar symmetry.
    full_sphere_coords = pitn.tract.direction.fodf_duplicate_hemisphere2sphere(
        seed_theta, seed_phi, [seed_coord], [0]
    )
    # seed_theta = torch.cat([seed_theta, (seed_theta + torch.pi / 2) % torch.pi], dim=0)
    # # Phi's range does not include -pi, so if phi is exactly pi, it must be rounded
    # # the the next lowest value.
    # seed_phi = torch.cat(
    #     [
    #         seed_phi,
    #         torch.clamp_min(-seed_phi, -torch.pi + torch.finfo(seed_phi.dtype).eps),
    #     ],
    #     dim=0,
    # )
    seed_theta = full_sphere_coords.theta
    seed_phi = full_sphere_coords.phi
    seed_coord = full_sphere_coords.vals[0]
    init_theta_phi = torch.stack([seed_theta, seed_phi], -1)

    tangent_t1_zyx = pitn.tract.local.gen_tract_step_rk4(
        seed_coord,
        step_size=step_size,
        fn_zyx_direction_t2theta_phi=fn_zyx_direction_t2theta_phi,
        init_direction_theta_phi=init_theta_phi,
    )

    seeds_t0_t1 = torch.stack([seed_coord, seed_coord + tangent_t1_zyx], dim=0)
    return seeds_t0_t1, tangent_t1_zyx


def seeds_from_mask(
    mask: torch.Tensor, seeds_per_vox_axis: int, affine_vox2mm: torch.Tensor
) -> torch.Tensor:
    # Copied from dipy's seeds_from_mask() function, just adapted for pytorch.
    # <https://dipy.org/documentation/1.5.0/reference/dipy.tracking/#seeds-from-mask>
    # <https://github.com/dipy/dipy/blob/master/dipy/tracking/utils.py#L372>

    # Assume that (0, 0, 0) is the *center* of each voxel, *not* the corner!
    # Get offsets in each dimension in local voxel coordinates.
    within_vox_offsets = torch.meshgrid(
        [
            torch.linspace(
                -0.5,
                0.5,
                steps=seeds_per_vox_axis,
                dtype=affine_vox2mm.dtype,
                device=affine_vox2mm.device,
            )
        ]
        * 3,
        indexing="ij",
    )

    within_vox_offsets = torch.stack(within_vox_offsets, -1).reshape(1, -1, 3)
    # within_vox_offsets_mm = pitn.affine.coord_transform_3d(
    #     within_vox_offsets, affine_vox2mm
    # )
    # Only allow batch=1 and channel=1 mask Tensors!
    if mask.ndim == 5:
        assert mask.shape[0] == 1
        mask = mask[0]
    if mask.ndim == 4:
        assert mask.shape[0] == 1
        mask = mask[0]

    # Broadcast mask coordinates against all offsets.
    mask_coords_vox = torch.stack(torch.where(mask), -1).reshape(-1, 1, 3)
    dense_mask_coords_vox = mask_coords_vox + within_vox_offsets
    dense_mask_coords_vox = dense_mask_coords_vox.reshape(-1, 3)
    seeds_mm = pitn.affine.coord_transform_3d(dense_mask_coords_vox, affine_vox2mm)

    return seeds_mm
