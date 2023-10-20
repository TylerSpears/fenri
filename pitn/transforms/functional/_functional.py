# -*- coding: utf-8 -*-
from typing import Literal, NamedTuple, Optional, Tuple, TypedDict, Union

import einops
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import torch

import pitn
from pitn.affine import AffineSpace


def _ensure_vol_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        y = x.unsqueeze(0)
    elif x.ndim == 4:
        y = x
    else:
        raise ValueError(
            "ERROR: Tensor must be shape (i, j, k) or shape (channel, i, j, k), "
            + f"got shape {tuple(x.shape)}"
        )
    return y


def _undo_vol_channels(x_ch: torch.Tensor, orig_x: torch.Tensor) -> torch.Tensor:
    if orig_x.ndim == 3:
        y = x_ch[0]
    else:
        y = x_ch
    return y.to(x_ch)


def _inv_affine(aff, rounding_decimals=6):
    if torch.is_tensor(aff):
        a = aff.detach().cpu().numpy()
    else:
        a = aff

    a_inv = np.linalg.inv(a)
    # Clean up any numerical instability artifacts.
    a_inv = np.round(a_inv, decimals=rounding_decimals)
    a_inv[..., -1, -1] = 1.0

    if torch.is_tensor(aff):
        ret = torch.from_numpy(a_inv).to(aff)
    else:
        ret = a_inv
    return ret


def fov_coord_grid(
    fov_bb_coords: torch.Tensor,
    affine_vox2real: torch.Tensor,
):
    spacing = torch.tensor(
        nib.affines.voxel_sizes(affine_vox2real.detach().cpu().numpy())
    ).to(fov_bb_coords)
    extent = fov_bb_coords[1] - fov_bb_coords[0]
    # Go in the negative direction for flipped axes.
    spacing = torch.where(extent < 0, -spacing, spacing)
    coord_axes = [
        torch.arange(
            fov_bb_coords[0, i],
            fov_bb_coords[1, i] + (spacing[i] / 100),  # include endpoint
            step=spacing[i],
            dtype=fov_bb_coords.dtype,
            device=fov_bb_coords.device,
        )
        for i in range(spacing.shape[0])
    ]
    coord_grid = torch.stack(torch.meshgrid(coord_axes, indexing="ij"), dim=-1)

    return coord_grid


def crop_vox(
    vol_vox: torch.Tensor,
    affine_vox2real: torch.Tensor,
    *crops_low_high: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    v_ch = _ensure_vol_channels(vol_vox)
    crop_low = [0] * (v_ch.ndim - 1)
    crop_high = [0] * (v_ch.ndim - 1)
    v_spatial_shape = tuple(v_ch.shape[-3:])
    for i, dim_i_crops in enumerate(crops_low_high):
        crop_low[i] = dim_i_crops[0]
        crop_high[i] = dim_i_crops[1]

    # Subset the volume tensor.
    v_ch = v_ch[
        ...,
        crop_low[0] : (v_spatial_shape[0] - crop_high[0]),
        crop_low[1] : (v_spatial_shape[1] - crop_high[1]),
        crop_low[2] : (v_spatial_shape[2] - crop_high[2]),
    ]

    # Low crops require a translation of the affine matrix to maintain vox->real
    # mappings.
    crop_low_vox_aff = torch.eye(affine_vox2real.shape[-1]).to(affine_vox2real)
    crop_low_vox_aff[:-1, -1] = torch.Tensor(crop_low).to(affine_vox2real)
    new_affine = einops.einsum(
        affine_vox2real, crop_low_vox_aff, "... i j, ... j k -> ... i k"
    )

    v = _undo_vol_channels(v_ch, vol_vox)

    return (v, new_affine)


def pad_vox(
    vol_vox: torch.Tensor,
    affine_vox2real: torch.Tensor,
    *spatial_pads_low_high: Tuple[int, int],
    **np_pad_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:

    v_ch = _ensure_vol_channels(vol_vox)
    pad_widths = np.zeros((v_ch.ndim, 2), dtype=int).tolist()
    # Add a (0, 0) padding at the start to not pad the channel dimension.
    if len(spatial_pads_low_high) != 4:
        spatial_pads_low_high = [(0, 0)] + list(spatial_pads_low_high)

    for i, dim_i_pads in enumerate(spatial_pads_low_high):
        pad_widths[i][0] = dim_i_pads[0]
        pad_widths[i][1] = dim_i_pads[1]

    # Pad the volume in voxel space.
    v_ch = torch.from_numpy(
        np.pad(v_ch.cpu().numpy(), pad_width=pad_widths, **np_pad_kwargs)
    ).to(v_ch)

    # Low pads require a translation of the affine matrix to maintain vox->real
    # mappings.
    pad_low = [p[0] for p in pad_widths[1:]]
    pad_low_vox_aff = torch.eye(affine_vox2real.shape[-1]).to(affine_vox2real)
    pad_low_vox_aff[:-1, -1] = -torch.Tensor(pad_low).to(affine_vox2real)
    new_affine = einops.einsum(
        affine_vox2real, pad_low_vox_aff, "... i j, ... j k -> ... i k"
    )

    v = _undo_vol_channels(v_ch, vol_vox)

    return (v, new_affine)


def vox_shape_from_fov(
    fov_real_bb_coords: torch.Tensor, affine_vox2real: torch.Tensor, tol=1e-4
) -> Tuple[int, ...]:

    fov_vox = pitn.affine.transform_coords(
        fov_real_bb_coords, _inv_affine(affine_vox2real, rounding_decimals=8)
    )

    # While edges of the fov may not lie directly on a voxel's center, the size of the
    # fov should be unit length.
    fov_vox_len = fov_vox[1] - fov_vox[0]
    if torch.max(torch.abs(fov_vox_len - fov_vox_len.round())) > tol:
        raise RuntimeError("ERROR: Vox fov rounding not within tol")
    fov_vox_shape = tuple(fov_vox_len.round().int().tolist())

    return fov_vox_shape


def scale_fov_spacing(
    fov_bb_coords: torch.Tensor,
    affine_vox2real: torch.Tensor,
    spacing_scale_factors: Tuple[float, ...],
    set_affine_orig_to_fov_orig: bool,
    new_fov_align_direction: Literal["interior", "exterior"] = "interior",
) -> Tuple[torch.Tensor, torch.Tensor]:

    scale_transform = np.ones(affine_vox2real.shape[-1])
    for i, scale in enumerate(spacing_scale_factors):
        scale_transform[i] = scale
    scale_transform = torch.diag(torch.Tensor(scale_transform)).to(affine_vox2real)
    affine_unit2rescaled_space = einops.einsum(
        affine_vox2real, scale_transform, "... i j, ... j k -> ... i k"
    )
    # Round for slightly better numerical stability with the affine transforms.
    # affine_unit2rescaled_space = torch.round(affine_unit2rescaled_space, decimals=7)
    # Treat the vox space as just a unit, positive directed space, for convenience.
    unit_fov_bb = pitn.affine.transform_coords(
        fov_bb_coords, _inv_affine(affine_unit2rescaled_space, rounding_decimals=8)
    )
    # Clamp almost-zero values to be 0, as they should almost certainly be 0.
    unit_fov_bb[0] = torch.where(
        torch.isclose(
            unit_fov_bb[0], torch.zeros_like(unit_fov_bb[0]), atol=1e-4, rtol=1e-5
        ),
        torch.zeros_like(unit_fov_bb[0]),
        unit_fov_bb[0],
    )

    unit_fov_extent = unit_fov_bb[1] - unit_fov_bb[0]
    new_unit_fov_bb = unit_fov_bb.clone()
    residual_fov = unit_fov_extent % 1
    # Dims that fall on multiples of the new spacing do not need to be rounded, such
    # as when a spacing is scaled by an integer factor.
    translate_dim_indicator = ~torch.isclose(residual_fov, residual_fov.new_tensor([0]))

    # Align the new fov such that the length of each side is evenly divisible by the
    # new spacing. The alignment may be pushed "inside" the original fov, or "outside"
    # of it.
    fov_align = new_fov_align_direction.lower().strip()
    if fov_align in {"in", "inside", "internal", "interior"}:
        fov_align = "interior"
    elif fov_align in {"out", "outside", "external", "exterior"}:
        fov_align = "exterior"
    else:
        raise ValueError(
            f"ERROR: Invalid new_fov_align_direction: {new_fov_align_direction}"
        )

    residual_side = residual_fov / 2
    # Translate lower bound "up"
    new_unit_fov_bb[0] = new_unit_fov_bb[0] + (translate_dim_indicator * residual_side)
    # Translate upper bound "down"
    new_unit_fov_bb[1] = new_unit_fov_bb[1] - (translate_dim_indicator * residual_side)
    if fov_align == "exterior":
        # Move each side by 0.5 units
        # Translate lower bound "down"
        new_unit_fov_bb[0] = new_unit_fov_bb[0] - (translate_dim_indicator * 0.5)
        # Translate upper bound "up"
        new_unit_fov_bb[1] = new_unit_fov_bb[1] + (translate_dim_indicator * 0.5)

    # Bring new fov bb coordinates back into real space.
    new_fov_bb = pitn.affine.transform_coords(
        new_unit_fov_bb, affine_unit2rescaled_space
    )
    if set_affine_orig_to_fov_orig:
        affine_out = affine_unit2rescaled_space.clone()
        affine_out[:-1, -1] = new_fov_bb[0]
    else:
        affine_out = affine_unit2rescaled_space

    return new_fov_bb, affine_out


def resample_dwi_directions(
    dwi: torch.Tensor,
    src_grad_mrtrix_table: torch.Tensor,
    target_grad_mrtrix_table: torch.Tensor,
    bval_round_decimals=-2,
    k_nearest_points=5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_g = src_grad_mrtrix_table[:, :-1].detach().cpu().numpy()
    src_bvals = src_grad_mrtrix_table[:, -1].detach().cpu().numpy()
    target_g = target_grad_mrtrix_table[:, :-1].detach().cpu().numpy()
    target_bvals = target_grad_mrtrix_table[:, -1].detach().cpu().numpy()

    src_shells = np.round(src_bvals, decimals=bval_round_decimals).astype(int)
    target_shells = np.round(target_bvals, decimals=bval_round_decimals).astype(int)
    # Force all vectors to have unit norm.
    src_g_norm = np.linalg.norm(src_g, ord=2, axis=-1, keepdims=True)
    src_g_norm = np.where(np.isclose(src_g_norm, 0, atol=1e-4), 1.0, src_g_norm)
    src_g = src_g / src_g_norm
    target_g_norm = np.linalg.norm(target_g, ord=2, axis=-1, keepdims=True)
    target_g_norm = np.where(
        np.isclose(target_g_norm, 0, atol=1e-4), 1.0, target_g_norm
    )
    target_g = target_g / target_g_norm
    # Project all vectors into the top hemisphere, as we have antipodal symmetry in dwi.
    src_g = np.where(src_g[:, -1, None] < 0, -src_g, src_g)
    target_g = np.where(target_g[:, -1, None] < 0, -target_g, target_g)
    # src_g[:, -1] = np.abs(src_g[:, -1])
    # target_g[:, -1] = np.abs(target_g[:, -1])

    # Double precision floats are more numerically stable, so explicitly cast to that
    # inside the arccos.
    p_arc_len = np.arccos(
        einops.einsum(
            target_g.astype(np.float64), src_g.astype(np.float64), "b1 d, b2 d -> b1 b2"
        )
    )
    p_arc_len = p_arc_len.astype(target_g.dtype)
    p_arc_len = np.nan_to_num(p_arc_len, nan=0)
    p_arc_w = np.pi - p_arc_len
    # Zero-out weights between dissimilar shells.
    p_arc_w[target_shells[:, None] != src_shells[None, :]] = 0.0

    # b0 volumes are just assigned a copy of the source b0 that is nearest to the
    # relative index of the target b0.
    # Zero-out all b0 weights
    p_arc_w[(target_shells[:, None] == 0) & (src_shells[None, :] == 0)] = 0.0
    src_b0_idx = np.where(np.isclose(src_shells, 0))[0]
    src_b0_relative_idx = src_b0_idx / len(src_shells)
    target_b0_idx = np.where(np.isclose(target_shells, 0))[0]
    target_b0_relative_idx = target_b0_idx / len(target_shells)

    for (
        i_target_b0,
        rel_target_b0_i_idx,
    ) in zip(target_b0_idx, target_b0_relative_idx):
        j_selected_src_b0 = src_b0_idx[
            np.argmin(np.abs(rel_target_b0_i_idx - src_b0_relative_idx))
        ]
        # pi is the max weight that can be selected before normalization.
        p_arc_w[i_target_b0, j_selected_src_b0] = np.pi

    # Zero-out any weights lower than the top k weights.
    p_arc_w[
        p_arc_w < (np.flip(np.sort(p_arc_w, -1), -1))[:, (k_nearest_points - 1), None]
    ] = 0.0
    # If any weights are close to pi, then zero-out all other weights and just produce
    # a copy of the DWI that is (nearly) angularly identical to the target.
    for i_target in range(p_arc_w.shape[0]):
        if np.isclose(p_arc_w[i_target], np.pi, atol=1e-5).any():
            pi_idx = np.argmax(p_arc_w[i_target])
            p_arc_w[i_target] = p_arc_w[i_target] * 0.0
            p_arc_w[i_target, pi_idx] = np.pi

    # Normalize weights to sum to 1.0.
    norm_p_arc_w = p_arc_w / p_arc_w.sum(1, keepdims=True)
    # Weigh and combine the source DWIs to make the target DWIs.
    target_dwis = list()
    src_dwi = dwi.detach().cpu().numpy()
    for i_target in range(p_arc_w.shape[0]):
        norm_w_i = norm_p_arc_w[i_target]
        tmp_dwi_i = np.zeros_like(src_dwi[0])
        for j_src, w_j in enumerate(norm_w_i):
            if np.isclose(w_j, 0):
                continue
            tmp_dwi_i += src_dwi[j_src] * w_j
        target_dwis.append(tmp_dwi_i)

    target_dwis = torch.from_numpy(np.stack(target_dwis, 0)).to(dwi)

    return target_dwis


def distance_transform_mask(
    m: torch.Tensor, spacing: Tuple[float, ...]
) -> torch.Tensor:
    m_np = m.cpu().numpy()[0]
    dt = scipy.ndimage.distance_transform_edt(
        m_np, sampling=spacing, return_distances=True
    )
    return torch.from_numpy(dt).to(device=m.device).expand_as(m).to(torch.float64)


def prefilter_gaussian_blur(
    vol: torch.Tensor,
    src_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    sigma_scale_coeff: float = 2.0,
    sigma_truncate: float = 4.0,
):
    v = vol.detach().cpu().numpy()
    # Assume isotropic resampling.
    scale_ratio_high_to_low = (
        torch.mean(torch.Tensor(src_spacing)) / torch.mean(torch.Tensor(target_spacing))
    ).item()
    # Assume the src spacing is lower (i.e., higher spatial resolution) than the target
    # spacing.
    assert scale_ratio_high_to_low <= 1.0
    sigma = 1 / (sigma_scale_coeff * scale_ratio_high_to_low)
    if len(v.shape) == 4:
        sigma = (0, sigma, sigma, sigma)
    else:
        sigma = (sigma,) * 3
    v_filter = scipy.ndimage.gaussian_filter(
        v, sigma=sigma, order=0, mode="nearest", truncate=sigma_truncate
    )

    vol_blur = torch.from_numpy(v_filter).to(vol)

    return vol_blur


def add_rician_noise(
    dwi: torch.Tensor,
    grad_table: pd.DataFrame,
    snr: Union[float, torch.Tensor],
    S0: Union[float, torch.Tensor],
    rng: torch.Generator,
    dwi_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Generator]:
    """Adds Rician-sampled complex noise to DWI volumes based on given SNR and S0.

    Based on implementations found in dipy
    <https://dipy.org/documentation/1.7.0/reference/dipy.sims/#add-noise> and
    <https://github.com/bchimagine/fODF_deep_learning/blob/b93a40b2de841aec5c619bbf7bd7fc4edc626ea6/crl_aux.py#L41>,
    the implementation of:

        D. Karimi, L. Vasung, C. Jaimes, F. Machado-Rivas, S. K. Warfield, and A. Gholipour,
        "Learning to estimate the fiber orientation distribution function from
        diffusion-weighted MRI," NeuroImage, vol. 239, p. 118316, Oct. 2021,
        doi: 10.1016/j.neuroimage.2021.118316.

    Parameters
    ----------
    dwi : torch.Tensor
        DWI channel-first volume Tensor.
    grad_table : pd.DataFrame
        MRtrix-style gradient table as a DataFrame.
    snr : Union[float, torch.Tensor]
        Target signal-to-noise ratio
    S0 : Union[float, torch.Tensor]
        Give S0 reference value for calculating sigma
    rng : torch.Generator
        Pytorch random number generator
    dwi_mask : Optional[torch.Tensor]
        Channel-first spatial mask, used in b0 quantile if given, by default None

    Returns
    -------
    Tuple[torch.Tensor, torch.Generator]
        Returns noised DWI Tensor and the random number generator.

    """
    rng_fork = torch.Generator(device=rng.device)
    rng_fork.set_state(rng.get_state())
    b = torch.from_numpy(grad_table.b.to_numpy()).to(dwi)
    shells = torch.round(b, decimals=-2)

    if not torch.is_tensor(S0):
        S0 = torch.ones_like(dwi) * S0
    if not torch.is_tensor(snr):
        snr = torch.ones_like(dwi) * snr
    # SNR = S0 / sigma
    sigma = S0 / snr
    sigma = sigma.broadcast_to(dwi.shape)
    N_real = torch.normal(mean=0, std=sigma, generator=rng_fork)
    N_complex = torch.normal(mean=0, std=sigma, generator=rng_fork)

    S = torch.sqrt((dwi + N_real) ** 2 + N_complex**2)
    if dwi_mask is not None:
        S = S * dwi_mask

    return S, rng_fork


def transform_affine_space(
    s: AffineSpace,
    affine_transform: torch.Tensor,
) -> AffineSpace:

    aff_p = einops.einsum(affine_transform, s.affine, "... i j, ... j k -> ... i k")
    bb_p = pitn.affine.transform_coords(s.fov_bb_coords, affine_transform)

    return AffineSpace(affine=aff_p, fov_bb_coords=bb_p)


def _reorientation_transform(
    input_ornt_code: tuple[str, ...],
    output_ornt_code: tuple[str, ...],
    input_vox_space_shape: tuple[int, ...],
):
    a_code = input_ornt_code
    b_code = output_ornt_code

    a_ornt = nib.orientations.axcodes2ornt(a_code)
    b_ornt = nib.orientations.axcodes2ornt(b_code)
    a2b_ornt = nib.orientations.ornt_transform(a_ornt, b_ornt)

    transform_aff_inv = nib.orientations.inv_ornt_aff(
        a2b_ornt, shape=input_vox_space_shape
    )
    transform_aff = _inv_affine(transform_aff_inv)

    return transform_aff


def reorient_affine_space(s: AffineSpace, target_orientation_code: str) -> AffineSpace:
    """Reorient the axes of a 3D affine space according to standard ornt codes.

    Parameters
    ----------
    s : AffineSpace
        Source affine space matrix and fov coordinates.
    target_orientation_code : str
        Target axis orientation. This must go by the MRI convention of R/L, A/P, and
        I/S, in any order, as one single upper-cased string. Ex. 'LAS'.

    Returns
    -------
    AffineSpace
        Reoriented affine matrix and fov boundary coordinates.
    """
    target_ornt_code = tuple(target_orientation_code.upper().strip())
    current_ornt_code = nib.orientations.aff2axcodes(s.affine.detach().cpu().numpy())

    # Get shape for output space when discretized.
    unit_sizes = nib.affines.voxel_sizes(s.affine.detach().cpu().numpy())
    fov_sizes = torch.abs(s.fov_bb_coords[1] - s.fov_bb_coords[0]).cpu().numpy()
    # Add 1 to get the actual shape, instead of the "length" of each "side."
    input_shape = (
        np.round(fov_sizes / unit_sizes, decimals=0).astype(int) + 1
    ).tolist()

    transform_aff = _reorientation_transform(
        current_ornt_code, target_ornt_code, input_vox_space_shape=input_shape
    )

    transform_aff = torch.from_numpy(transform_aff).to(s.affine)

    return transform_affine_space(s, transform_aff)


# def crop_affine_space_by_vox(
#     s: AffineSpace, *vox_crops_low_high: Tuple[int, int]
# ) -> AffineSpace:
#     """Transform an affine space when cropping by some number of voxels in any dim.

#     Parameters
#     ----------
#     s : AffineSpace
#         Source AffineSpace object that contains some vox -> target transform and the
#         target space coordinates of the bounding box/fov.

#     Returns
#     -------
#     AffineSpace
#         Cropped AffineSpace with updated transformation and fov.
#     """

#     crop_low = [0] * s.fov_bb_coords.shape[-1]
#     crop_high = [0] * s.fov_bb_coords.shape[-1]

#     for i, dim_i_crops in enumerate(vox_crops_low_high):
#         crop_low[i] = dim_i_crops[0]
#         crop_high[i] = dim_i_crops[1]
#     crop_low = torch.Tensor(crop_low).to(s.affine)
#     crop_high = torch.Tensor(crop_high).to(s.affine)

#     # Low crops require a translation of the lower bound and a translation of the
#     # affine matrix.
#     # Translate the fov lower bounds.
#     fov_s_low = s.fov_bb_coords[0]
#     fov_vox_low = pitn.affine.transform_coords(fov_s_low, _inv_affine(s.affine))
#     new_fov_s_low = pitn.affine.transform_coords(fov_vox_low + crop_low, s.affine)
#     # Also transform the affine matrix.
#     crop_low_vox_aff = torch.eye(s.affine.shape[0]).to(s.affine)
#     crop_low_vox_aff[:-1, -1] = crop_low
#     new_affine = einops.einsum(
#         s.affine, crop_low_vox_aff, "... i j, ... j k -> ... i k"
#     )

#     # Translate the fov upper bounds.
#     fov_s_up = s.fov_bb_coords[1]
#     fov_vox_up = pitn.affine.transform_coords(fov_s_up, _inv_affine(s.affine))
#     new_fov_s_up = pitn.affine.transform_coords(fov_vox_up - crop_high, s.affine)

#     new_s = AffineSpace(
#         affine=new_affine,
#         fov_bb_coords=torch.stack(
#             [new_fov_s_low, new_fov_s_up],
#             dim=0,
#         ),
#     )

#     return new_s


# def pad_affine_space_by_vox(
#     s: AffineSpace, *vox_pads_low_high: Tuple[int, int]
# ) -> AffineSpace:
#     """Transform an affine space when padding by some number of voxels in any dim.

#     Parameters
#     ----------
#     s : AffineSpace
#         Source AffineSpace object that contains some vox -> target transform and the
#         target space coordinates of the bounding box/fov.

#     Returns
#     -------
#     AffineSpace
#         Padded AffineSpace with updated transformation and fov.
#     """

#     # Padding is just negative cropping...
#     vox_crops_low_high = [(-p[0], -p[1]) for p in vox_pads_low_high]
#     return crop_affine_space_by_vox(s, *vox_crops_low_high)


# def prefilter_gaussian_blur(
#     vol: torch.Tensor,
#     src_spacing: Tuple[float, ...],
#     target_spacing: Tuple[float, ...],
#     sigma_scale_coeff: float = 2.5,
#     sigma_truncate=4.0,
# ):
#     v = vol.detach().cpu().numpy()
#     # Assume isotropic resampling.
#     scale_ratio_high_to_low = (
#         torch.mean(torch.Tensor(src_spacing)) / torch.mean(torch.Tensor(target_spacing))
#     ).item()
#     # Assume the src spacing is lower (i.e., higher spatial resolution) than the target
#     # spacing.
#     assert scale_ratio_high_to_low <= 1.0
#     sigma = 1 / (sigma_scale_coeff * scale_ratio_high_to_low)
#     if len(v.shape) == 4:
#         sigma = (0, sigma, sigma, sigma)
#     else:
#         sigma = (sigma,) * 3
#     v_filter = scipy.ndimage.gaussian_filter(
#         v, sigma=sigma, order=0, mode="nearest", truncate=sigma_truncate
#     )

#     vol_blur = torch.from_numpy(v_filter).to(vol)

#     return vol_blur


# def resample_to_affine_space(
#     x: torch.Tensor, target_s: AffineSpace, **scipy_map_coordinates_kwargs
# ) -> torch.Tensor:
#     n_channels = x.shape[0] if x.ndim == 4 else 0
#     spacing_ = nib.affines.voxel_sizes(target_s.affine.detach().cpu().numpy())
#     spacing = list()
#     for i, s in enumerate(spacing_):
#         if target_s.fov_bb_coords[1, i] < target_s.fov_bb_coords[0, i]:
#             spacing.append(-spacing_[i])
#         else:
#             spacing.append(spacing_[i])
#     spatial_coords = torch.stack(
#         torch.meshgrid(
#             *[
#                 torch.arange(
#                     target_s.fov_bb_coords[0, i],
#                     target_s.fov_bb_coords[1, i] + (spacing[i] / 10),
#                     spacing[i],
#                 )
#                 for i in range(len(spacing))
#             ],
#             indexing="ij",
#         ),
#         dim=-1,
#     ).to(target_s.fov_bb_coords)

#     vox_coords = pitn.affine.transform_coords(
#         spatial_coords, _inv_affine(target_s.affine)
#     )
#     # Scipy needs coordinates first.
#     vox_coords = np.moveaxis(vox_coords.detach().cpu().numpy(), -1, 0)

#     a = x.detach().cpu().numpy()
#     if n_channels == 0:
#         a = a[None]
#     y = list()
#     for i in range(max(n_channels, 1)):
#         a_resample = scipy.ndimage.map_coordinates(
#             a[i], coordinates=vox_coords, **scipy_map_coordinates_kwargs
#         )
#         y.append(a_resample)

#     y = torch.from_numpy(np.stack(y, axis=0)).to(x)
#     if n_channels == 0:
#         y = y[0]

#     return y


# def _random_iso_center_scale_affine(
#     src_affine: torch.Tensor,
#     src_spatial_sample: torch.Tensor,
#     scale_low: float,
#     scale_high: float,
#     n_delta_buffer_scaled_vox: int = 1,
# ) -> torch.Tensor:
#     # Randomly increase the spacing of src affine isotropically, while adding
#     # translations to center the resamples into the src FoV.
#     scale = np.random.uniform(scale_low, scale_high)
#     scaling_affine = monai.transforms.utils.create_scale(
#         3,
#         [scale] * 3,
#     )
#     scaling_affine = torch.from_numpy(scaling_affine).to(src_affine)
#     # Calculate the offset in target space voxels such that the target FoV will be
#     # centered in the src FoV, and have 1 target voxel buffer between the LR fov and src
#     # fov.
#     src_spatial_shape = np.array(tuple(src_spatial_sample.shape[1:]))
#     src_spacing = monai.data.utils.affine_to_spacing(src_affine, r=3).cpu().numpy()
#     src_spatial_extent = src_spacing * src_spatial_shape
#     target_n_vox_in_src_fov = src_spatial_extent / (scale * src_spacing)
#     fov_delta = target_n_vox_in_src_fov - np.floor(target_n_vox_in_src_fov)
#     evenly_distribute_fov_delta = fov_delta / 2
#     # Add 1 scaled voxel between the inner (scaled) fov and the outer (src) fov, while
#     # also keeping the delta spacing to keep the scaled fov centered wrt the src fov.
#     evenly_distribute_fov_delta = (
#         evenly_distribute_fov_delta + n_delta_buffer_scaled_vox
#     )
#     translate_aff = torch.eye(4).to(scaling_affine)
#     translate_aff[:-1, -1] = torch.from_numpy(evenly_distribute_fov_delta).to(
#         translate_aff
#     )

#     # Delta is in target space voxels, so we need to scale first, then translate.
#     target_affine = src_affine @ (translate_aff @ scaling_affine)
#     return target_affine


# def resample_dwi_to_grad_directions(
#     dwi: torch.Tensor,
#     src_bvec: torch.Tensor,
#     src_bval: torch.Tensor,
#     target_bvec: torch.Tensor,
# ):
#     K = 5
#     bval_round_decimals = -2
#     # Assume that the src and target bvecs are referring to the same gradient strengths
#     # (bvals), just different orientations.
#     x_g = src_bvec.detach().cpu().numpy().T
#     y_g = target_bvec.detach().cpu().numpy().T
#     bval = src_bval.detach().cpu().numpy()
#     shells = np.round(bval, decimals=bval_round_decimals).astype(int)
#     # If target and source are b0s, then no re-weighting should be done, as there is no
#     # gradient.

#     d_cos = scipy.spatial.distance.cdist(y_g, x_g, "cosine")
#     sim = 1 - d_cos
#     sim = np.clip(np.abs(sim), a_min=None, a_max=1 - 1e-5)
#     l = np.arccos(sim)
#     # For each shell (excluding b=0), restrict the available dwis to only the matching
#     # shell.
#     unique_shells = set(np.unique(shells).tolist()) - {0}
#     for s in unique_shells:
#         s_mask = np.isclose(shells, s)
#         shell_intersection_mask = np.logical_and(s_mask[:, None], s_mask[None, :])
#         shell_dissimilar_mask = ~shell_intersection_mask
#         l[shell_dissimilar_mask * s_mask[:, None]] = np.inf

#     top_k_idx = np.argsort(l, axis=1, kind="stable")[:, :K]

#     w = np.take_along_axis(1 / np.clip(l, a_min=1e-5, a_max=None), top_k_idx, axis=1)
#     w = w / w.sum(axis=1, keepdims=True)

#     w = torch.from_numpy(w).to(dwi)
#     top_k_idx = torch.from_numpy(top_k_idx).to(dwi).long()
#     # Start with identity convolution, which will be left for the b0s.
#     w_conv = torch.eye(dwi.shape[0]).to(dwi)
#     shells = torch.from_numpy(shells).to(dwi.device)
#     for i_y in range(dwi.shape[0]):
#         shell_i = shells[i_y]
#         if shell_i == 0:
#             continue
#         top_k_i = top_k_idx[i_y]
#         w_i = w[i_y]
#         # zero-out the row for y_i
#         w_conv[i_y] = 0
#         w_conv[i_y, top_k_i] = w_i

#     w_conv = w_conv[:, :, None, None, None]
#     dwi_target = torch.nn.functional.conv3d(dwi[None], w_conv)
#     dwi_target = dwi_target[0]

#     # w = torch.from_numpy(w).to(dwi)

#     # top_k_idx = torch.from_numpy(top_k_idx).to(src_bval).long()
#     # # dwi_np = dwi.detach().cpu().numpy()
#     # dwi_target = monai.inferers.sliding_window_inference(
#     #     dwi[None],
#     #     roi_size=(72, 72, 72),
#     #     sw_batch_size=1,
#     #     predictor=lambda dw: torch.from_numpy(
#     #         einops.einsum(
#     #             np.take(dw[0].cpu().numpy(), top_k_idx, axis=0),
#     #             w,
#     #             "i j x y z,i j -> i x y z",
#     #         )
#     #     ).to(dw)[None],
#     #     overlap=0,
#     #     padding_mode="replicate",
#     # )[0]
#     # # Use ein. notation to weight and sum over K closest gradient directions.
#     # dwi_target = einops.einsum(
#     #     torch.take(dwi_np, top_k_idx, axis=0), w, "i j x y z,i j -> i x y z"
#     # )
#     # Re-assign the b0s
#     # b0_mask = torch.from_numpy(np.isclose(y_g, 0.0).all(1)).to(src_bval).bool()
#     # dwi_target[b0_mask] = dwi[b0_mask]

#     return dwi_target, target_bvec.to(src_bvec)


# def sub_select_dwi_from_bval(
#     dwi: torch.Tensor,
#     bval: torch.Tensor,
#     bvec: torch.Tensor,
#     shells_to_remove: list[float] = list(),
#     within_shell_idx_to_keep: dict[float, tuple[int]] = dict(),
#     bval_round_decimals: int = -2,
# ) -> Tuple[torch.Tensor, torch.Tensor]:

#     keep_mask = torch.ones_like(bval).bool()

#     shells = torch.round(bval, decimals=bval_round_decimals)

#     for s in shells_to_remove:
#         keep_mask[torch.isclose(shells, shells.new_tensor(float(s)))] = False

#     for s, idx_to_keep in within_shell_idx_to_keep.items():
#         # Sub-select only bvals in this shell.
#         shell_mask = torch.isclose(shells, shells.new_tensor(float(s))).bool()
#         within_shell_idx_to_keep = bval.new_tensor(tuple(idx_to_keep)).long()
#         within_shell_mask = shell_mask[shell_mask]
#         within_shell_mask[within_shell_idx_to_keep] = False
#         within_shell_mask_to_keep = ~within_shell_mask
#         # Merge current running mask with the sub-selected shell.
#         keep_mask[shell_mask] = keep_mask[shell_mask] * within_shell_mask_to_keep

#     return {"dwi": dwi[keep_mask], "bval": bval[keep_mask], "bvec": bvec[:, keep_mask]}
