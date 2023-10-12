# -*- coding: utf-8 -*-
import collections
import functools
import itertools
import math
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, TypedDict, Union

import einops
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import skimage
import torch

import pitn
import pitn.transforms.functional as ptf


class PreprocedSuperResSubjDict(pitn.data.LoadedSuperResSubjSampleDict):
    S0_noise: float
    patch_sampling_cumulative_weights: torch.Tensor


class SuperResLRFRSample(TypedDict):
    subj_id: str
    affine_lr_vox2real: torch.Tensor
    lr_real_coords: torch.Tensor
    lr_spacing: Tuple[float, ...]
    lr_fov_coords: torch.Tensor
    lr_dwi: torch.Tensor
    grad_table: np.ndarray
    full_res_real_coords: torch.Tensor
    full_res_spacing: Tuple[float, ...]
    full_res_fov_coords: torch.Tensor
    odf: torch.Tensor
    brain_mask: torch.Tensor
    wm_mask: torch.Tensor
    gm_mask: torch.Tensor
    csf_mask: torch.Tensor


def _is_vol(x: torch.Tensor, ch=True, batch=False):
    if not torch.is_tensor(x):
        s = False
    else:
        if ch:
            if batch:
                s = x.ndim == 5
            else:
                s = x.ndim == 4
        else:
            if not batch:
                s = x.ndim == 3
            else:
                raise ValueError("ERROR: Must assume channel dim if assuming batch dim")
    return s


def preproc_loaded_super_res_subj(
    loaded_super_res_subj: pitn.data.LoadedSuperResSubjSampleDict,
    S0_noise_b0_quantile: float,
    resample_target_grad_table: Optional[pd.DataFrame] = None,
    patch_sampling_w_erosion: Optional[int] = None,
) -> PreprocedSuperResSubjDict:
    grad_table = loaded_super_res_subj["grad_table"]
    dwi = loaded_super_res_subj["dwi"]
    brain_mask = loaded_super_res_subj["brain_mask"]
    update_subj_dict = dict()

    # Calculate S0 value needed for noise injection later on.
    shells = grad_table.b.to_numpy().round(-2)
    # Take all b0s.
    b0_idx = (np.where(shells == 0)[0][:9],)
    b0_select = np.zeros_like(shells).astype(bool).copy()
    b0_select[b0_idx] = True
    b0s = dwi[shells == 0]
    b0s = b0s[brain_mask.broadcast_to(b0s.shape)].flatten()
    S0_noise = float(np.quantile(b0s.detach().cpu().numpy(), q=S0_noise_b0_quantile))
    del b0s

    # Resample the DWIs according to the given gradient table.
    if resample_target_grad_table is not None:
        src_grad = torch.from_numpy(grad_table.to_numpy()).to(dwi)
        target_grad = torch.from_numpy(resample_target_grad_table.to_numpy()).to(
            src_grad
        )
        dwi = ptf.resample_dwi_directions(
            dwi, src_grad_mrtrix_table=src_grad, target_grad_mrtrix_table=target_grad
        )
        grad_table = resample_target_grad_table.copy(deep=True)
        update_subj_dict["dwi"] = dwi
        update_subj_dict["grad_table"] = grad_table

    # Create patch sampling weights volume according to the brain mask.
    spacing = nib.affines.voxel_sizes(
        loaded_super_res_subj["affine_vox2real"].detach().cpu().numpy()
    )
    sample_w = ptf.distance_transform_mask(brain_mask, spacing=spacing)
    mask_w = sample_w > 0
    if patch_sampling_w_erosion is not None:
        m = mask_w.detach().cpu().clone().numpy().astype(bool)
        # Take out channel dim.
        m = m[0]
        m = scipy.ndimage.binary_fill_holes(m)
        m = scipy.ndimage.binary_erosion(m, iterations=patch_sampling_w_erosion)
        m = m[None]
        mask_w = torch.from_numpy(m).to(mask_w) & mask_w
    sample_w = sample_w * mask_w
    norm_sample_w = sample_w / sample_w.sum()

    cumul_norm_sample_w = torch.cumsum(norm_sample_w.view(-1), dim=0).view(
        norm_sample_w.shape
    )

    return PreprocedSuperResSubjDict(
        **(loaded_super_res_subj | update_subj_dict),
        patch_sampling_cumulative_weights=cumul_norm_sample_w,
        S0_noise=S0_noise,
    )


class _CallablePromisesList(collections.UserList):
    """Utility class that calls a callable when using indexing/using __getitem__.

    Used for lazily accessing items in a (potentially large) list of (potentially
    large) objects/containers.
    """

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ret = list()
            for i in idx:
                if callable(self.data[i]):
                    ret_i = self.data[i]()
                    # Cache any items already accessed.
                    self.data[i] = ret_i
                    ret.append(ret_i)
                else:
                    ret.append(self.data[i])
        else:
            if callable(self.data[idx]):
                ret = self.data[idx]()
                # Cache any items already accessed.
                self.data[idx] = ret
            else:
                ret = self.data[idx]
        return ret


def lazy_sample_patch_from_super_res_sample(
    sample_dict: PreprocedSuperResSubjDict,
    patch_size: Tuple[int, ...],
    num_samples: int,
    rng: Union[str, torch.Generator] = "default",
    skip_crop_keys: Optional[Tuple[str]] = None,
) -> _CallablePromisesList[PreprocedSuperResSubjDict]:
    def sample_single_patch(
        sample_dict: PreprocedSuperResSubjDict,
        patch_size,
        rng,
        skip_crop_keys: set,
        drop_keys: set,
    ) -> PreprocedSuperResSubjDict:

        cumul_w = sample_dict["patch_sampling_cumulative_weights"][0]

        rng_fork = torch.Generator(device=rng.device)
        rng_fork.set_state(rng.get_state())
        r_v = torch.rand(
            1, generator=rng_fork, dtype=cumul_w.dtype, device=cumul_w.device
        )

        patch_center_flat_idx = torch.searchsorted(cumul_w.view(-1), r_v, side="right")
        patch_center_idx = np.unravel_index(
            patch_center_flat_idx.cpu().item(), shape=tuple(cumul_w.shape)
        )
        patch_center_idx = np.array(patch_center_idx)
        p_s = np.array(patch_size)
        bb_vox_high = np.array(tuple(cumul_w.shape)) - 1
        patch_size_lower = np.ceil(p_s / 2).astype(int)
        patch_size_upper = np.floor(p_s / 2).astype(int)
        patch_crop_low = patch_center_idx - patch_size_lower
        patch_crop_high = bb_vox_high - (patch_center_idx + patch_size_upper - 1)

        affine_key = "affine_vox2real"
        affine_vox2real = sample_dict[affine_key]
        out_dict = dict()
        for k, v in sample_dict.items():
            if k == affine_key or k in drop_keys:
                continue

            if k in skip_crop_keys or not _is_vol(v, ch=True, batch=False):
                out_dict[k] = v
            else:
                crop_vol, new_aff = ptf.crop_vox(
                    v,
                    affine_vox2real,
                    *[
                        (patch_crop_low[i], patch_crop_high[i])
                        for i in range(len(patch_crop_low))
                    ],
                )

                out_dict[k] = crop_vol
        out_dict[affine_key] = new_aff

        return PreprocedSuperResSubjDict(**out_dict)

    if rng == "default":
        state_rng = torch.default_generator.get_state()
    else:
        state_rng = rng.get_state()
    rng_fork = torch.Generator()
    rng_fork.set_state(state_rng)

    if skip_crop_keys is None:
        skip_crop_keys = set()
    else:
        skip_crop_keys = set(skip_crop_keys)
    drop_keys = set()
    lazy_samples = _CallablePromisesList()
    for i in range(num_samples):
        # Iterate the rng state.
        torch.randint(10, size=(1,), generator=rng_fork)
        # Create a new "forked" rng
        rng_i = torch.Generator(device=rng_fork.device)
        rng_i.set_state(rng_fork.get_state())
        lazy_samples.append(
            partial(
                sample_single_patch,
                sample_dict=sample_dict,
                patch_size=patch_size,
                rng=rng_i,
                skip_crop_keys=skip_crop_keys,
                drop_keys=drop_keys,
            )
        )

    if rng == "default":
        torch.default_generator.set_state(rng_fork.get_state())

    return lazy_samples


class _VoxRealAffineSpace(NamedTuple):
    vox_vol: torch.Tensor
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


class _BatchVoxRealAffineSpace(NamedTuple):
    vox_vols: Tuple[torch.Tensor, ...]
    affine_vox2real: torch.Tensor
    fov_bb_real: torch.Tensor


def _crop_lr_inside_smallest_lr(
    lr_vox_vol: torch.Tensor,
    affine_lr_vox2real: torch.Tensor,
    fr_fov_bb_real: torch.Tensor,
    affine_fr_vox2real: torch.Tensor,
    max_spacing_scale_factor: float,
) -> _VoxRealAffineSpace:

    min_lr_fov_bb, affine_min_lr_vox2real = ptf.scale_fov_spacing(
        fr_fov_bb_real,
        affine_fr_vox2real,
        spacing_scale_factors=(max_spacing_scale_factor,) * 3,
        new_fov_align_direction="interior",
    )
    min_lr_shape = ptf.vox_shape_from_fov(min_lr_fov_bb, affine_min_lr_vox2real)

    # Determine current LR shape and crop bb to match the minimum LR shape.
    # Assume that the lr real-coordinate fov bb lines up with the vox-coordinate fov bb.
    lr_fov_bb_real = pitn.affine.fov_bb_coords_from_vox_shape(
        affine_lr_vox2real, vox_vol=lr_vox_vol
    )
    lr_shape = ptf.vox_shape_from_fov(lr_fov_bb_real, affine_lr_vox2real)

    crop_lr_low = np.floor((np.array(lr_shape) - np.array(min_lr_shape)) / 2).astype(
        int
    )
    crop_lr_high = np.ceil((np.array(lr_shape) - np.array(min_lr_shape)) / 2).astype(
        int
    )
    crops_low_high = [
        (crop_lr_low[i], crop_lr_high[i]) for i in range(len(crop_lr_low))
    ]

    vox_conform_lr_vox_vol, vox_conform_lr_affine_vox2real = ptf.crop_vox(
        lr_vox_vol, affine_lr_vox2real, *crops_low_high
    )
    vox_conform_lr_fov_bb = pitn.affine.fov_bb_coords_from_vox_shape(
        vox_conform_lr_affine_vox2real, vox_vol=vox_conform_lr_vox_vol
    )

    return _VoxRealAffineSpace(
        vox_vol=vox_conform_lr_vox_vol,
        affine_vox2real=vox_conform_lr_affine_vox2real,
        fov_bb_real=vox_conform_lr_fov_bb,
    )


def _crop_frs_inside_lr(
    *fr_vox_vols: torch.Tensor,
    affine_fr_vox2real: torch.Tensor,
    lr_fov_bb_coords: torch.Tensor,
) -> _BatchVoxRealAffineSpace:
    fr_vol = fr_vox_vols[0]
    fr_fov_bb = pitn.affine.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real, vox_vol=fr_vol
    )
    affine_real2fr_vox = torch.linalg.inv(affine_fr_vox2real)
    fr_fov_in_fr_vox_space = pitn.affine.transform_coords(fr_fov_bb, affine_real2fr_vox)
    lr_fov_in_fr_vox_space = pitn.affine.transform_coords(
        lr_fov_bb_coords, affine_real2fr_vox
    )

    crop_low = lr_fov_in_fr_vox_space[0] - fr_fov_in_fr_vox_space[0]
    crop_low = torch.clip(torch.ceil(crop_low), min=0, max=torch.inf).int().tolist()
    crop_high = fr_fov_in_fr_vox_space[1] - lr_fov_in_fr_vox_space[1]
    crop_high = torch.clip(torch.ceil(crop_high), min=0, max=torch.inf).int().tolist()

    crops_low_high = [(crop_low[i], crop_high[i]) for i in range(len(crop_low))]

    vols = list()
    for v in fr_vox_vols:
        v_crop, affine_fr_vox2real_crop = ptf.crop_vox(
            v, affine_fr_vox2real, *crops_low_high
        )
        vols.append(v_crop)
    fr_crop_fov_bb = pitn.affine.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real_crop, vox_vol=vols[0]
    )

    return _BatchVoxRealAffineSpace(
        vox_vols=tuple(vols),
        affine_vox2real=affine_fr_vox2real_crop,
        fov_bb_real=fr_crop_fov_bb,
    )


def preproc_super_res_patch_sample(
    super_res_sample_dict: PreprocedSuperResSubjDict,
    downsample_factor_range: Tuple[float, float],
    noise_snr_range: Optional[Tuple[float, float]],
    rng: Union[str, torch.Generator] = "default",
    prefilter_sigma_scale_coeff: float = 2.5,
    prefilter_sigma_truncate: float = 4.0,
) -> SuperResLRFRSample:

    if rng == "default":
        state_rng = torch.default_generator.get_state()
    else:
        state_rng = rng.get_state()
    rng_fork = torch.Generator()
    rng_fork.set_state(state_rng)
    print(
        f"{torch.utils.data.get_worker_info().id}: {rng_fork.get_state().float().mean()}",
        flush=True,
    )

    affine_fr_vox2real = super_res_sample_dict["affine_vox2real"]
    fr_spacing = np.array(
        nib.affines.voxel_sizes(affine_fr_vox2real.detach().cpu().numpy())
    )
    dwi = super_res_sample_dict["dwi"]

    downsample_min = float(downsample_factor_range[0])
    downsample_max = float(downsample_factor_range[1])
    downsample_factor = torch.rand(1, generator=rng_fork, dtype=torch.float64).item()
    downsample_factor = (
        downsample_factor * (downsample_max - downsample_min) + downsample_min
    )
    lr_spacing = fr_spacing * downsample_factor

    # Set spacing for LR fov.
    orig_fr_bb_coords = pitn.affine.fov_bb_coords_from_vox_shape(
        affine_fr_vox2real, vox_vol=dwi
    )
    lr_fov_coords, affine_lr_vox2real = ptf.scale_fov_spacing(
        orig_fr_bb_coords,
        affine_fr_vox2real,
        spacing_scale_factors=(downsample_factor,) * 3,
        new_fov_align_direction="interior",
    )
    # Prefilter/blur DWI patch before downsampling.
    blur_dwi = ptf.prefilter_gaussian_blur(
        dwi,
        src_spacing=tuple(fr_spacing),
        target_spacing=tuple(lr_spacing),
        sigma_scale_coeff=prefilter_sigma_scale_coeff,
        sigma_truncate=prefilter_sigma_truncate,
    )
    # Downsample DWI patch.
    lr_real_coord_grid = ptf.fov_coord_grid(lr_fov_coords, affine_lr_vox2real)
    lr_dwi = pitn.affine.sample_vol(
        blur_dwi,
        coords_mm_xyz=lr_real_coord_grid,
        affine_vox2mm=affine_fr_vox2real,
        mode="linear",
        align_corners=True,
        override_out_of_bounds_val=torch.nan,
    )
    # Crop the LR fov s.t. the smallest possible LR shape is the same as all LR samples
    # (assuming the same FR input shape).
    if downsample_min != downsample_max:
        lr_space = _crop_lr_inside_smallest_lr(
            lr_dwi,
            affine_lr_vox2real,
            fr_fov_bb_real=orig_fr_bb_coords,
            affine_fr_vox2real=affine_fr_vox2real,
            max_spacing_scale_factor=downsample_max,
        )

        lr_dwi = lr_space.vox_vol
        affine_lr_vox2real = lr_space.affine_vox2real
        lr_fov_coords = lr_space.fov_bb_real

    # Add Rician noise to downsampled patch.
    if noise_snr_range is not None:
        snr_min = float(noise_snr_range[0])
        snr_max = float(noise_snr_range[1])
        snr = torch.rand(1, generator=rng_fork, dtype=torch.float64).item()
        snr = snr * (snr_max - snr_min) + snr_min
        lr_dwi, rng_fork = ptf.add_rician_noise(
            lr_dwi,
            super_res_sample_dict["grad_table"],
            snr=snr,
            rng=rng_fork,
            S0=super_res_sample_dict["S0_noise"],
            # Just hack together an LR brain mask, not worth resampling.
            dwi_mask=torch.all(
                ~torch.isclose(lr_dwi, lr_dwi.new_zeros(1)), dim=0, keepdim=True
            ),
        )

    # Crop the input FR sample to be contained within the cropped LR space.
    fr_vol_keys = ("odf", "brain_mask", "wm_mask", "gm_mask", "csf_mask")
    fr_vols = tuple([super_res_sample_dict[k] for k in fr_vol_keys])
    crop_frs_space = _crop_frs_inside_lr(
        *fr_vols,
        affine_fr_vox2real=affine_fr_vox2real,
        lr_fov_bb_coords=lr_fov_coords,
    )
    fr_vols = crop_frs_space.vox_vols
    affine_fr_vox2real = crop_frs_space.affine_vox2real
    fr_bb_coords = crop_frs_space.fov_bb_real

    # Generate coordinates for both full-res and low-res spaces.
    lr_real_coords = ptf.fov_coord_grid(lr_fov_coords, affine_lr_vox2real)
    fr_real_coords = ptf.fov_coord_grid(fr_bb_coords, affine_fr_vox2real)
    # Collate function needs channel-first tensors.
    lr_real_coords = einops.rearrange(lr_real_coords, "i j k coord -> coord i j k")
    fr_real_coords = einops.rearrange(fr_real_coords, "i j k coord -> coord i j k")

    out_dict = SuperResLRFRSample(
        subj_id=super_res_sample_dict["subj_id"],
        affine_lr_vox2real=affine_lr_vox2real,
        lr_real_coords=lr_real_coords,
        lr_spacing=lr_spacing,
        lr_fov_coords=lr_fov_coords,
        lr_dwi=lr_dwi,
        grad_table=super_res_sample_dict["grad_table"].to_numpy(),  # must be ndarray
        full_res_real_coords=fr_real_coords,
        full_res_spacing=fr_spacing,
        full_res_fov_coords=fr_bb_coords,
        **{fr_vol_keys[i]: fr_vols[i] for i in range(len(fr_vols))},
    )

    # If the generator was the default pytorch generator, then update the global rng
    # state.
    if rng == "default":
        torch.default_generator.set_state(rng_fork.get_state())

    return out_dict
