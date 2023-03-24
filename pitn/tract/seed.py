# -*- coding: utf-8 -*-
import collections
import math
from functools import partial
from typing import Callable, Optional, Tuple

import dipy
import einops
import numpy as np
import torch

from pitn._lazy_loader import LazyLoader

# Make pitn lazy load to avoid circular imports.
pitn = LazyLoader("pitn", globals(), "pitn")
# import pitn
# import pitn.affine
# import pitn.tract
# import pitn.tract.local
# import pitn.tract.peak

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
        theta_peaks=theta_peak,
        phi_peaks=phi_peak,
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
        theta_peaks=theta_peak,
        phi_peaks=phi_peak,
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


class BatchSeedSequenceSampler:
    def __init__(
        self,
        max_batch_size: int,
        unique_seed_coords_zyx_mm: torch.Tensor,
        fodf_coeffs_brain_vol: torch.Tensor,
        affine_vox2mm: torch.Tensor,
        max_peaks_per_voxel: int,
        tracking_step_size: float,
        fn_zyx_direction_t2theta_phi,
        pytorch_device="cpu",
        sh_order=8,
        **dipy_peak_finder_kwargs,
    ):
        self.max_batch_size = max_batch_size
        self.unique_seed_coords_zyx_mm = unique_seed_coords_zyx_mm
        self.target_tensor = self.unique_seed_coords_zyx_mm
        self.max_peaks_per_voxel = max_peaks_per_voxel
        self._max_peak_expansion_batch_size = math.floor(
            self.max_batch_size / (self.max_peaks_per_voxel * 2)
        )
        self._max_peak_expansion_batch_size = max(
            1, self._max_peak_expansion_batch_size
        )

        self.fodf_coeffs_brain_vol = fodf_coeffs_brain_vol
        self.affine_vox2mm = affine_vox2mm
        self.tracking_step_size = tracking_step_size
        self.fn_zyx_direction_t2theta_phi = fn_zyx_direction_t2theta_phi

        self.sh_order = sh_order

        self._fn_dipy_peak_finder_trilinear = partial(
            self._dipy_peak_finder_fn_linear_interp_zyx,
            fodf_coeffs_brain_vol=self.fodf_coeffs_brain_vol,
            affine_vox2mm=self.affine_vox2mm,
            sh_order=self.sh_order,
            **dipy_peak_finder_kwargs,
        )

        self._fn_rk4_expansion = partial(
            expand_seeds_from_topk_peaks_rk4,
            max_peaks_per_voxel=self.max_peaks_per_voxel,
            step_size=self.tracking_step_size,
            fn_zyx_direction_t2theta_phi=self.fn_zyx_direction_t2theta_phi,
        )

        self.device = pytorch_device
        self.seed_buffer = list()
        self.tangent_buffer = list()

        self._init_buffers(self.device)

    def _init_buffers(self, device):

        for b_start_idx in range(
            0,
            self.unique_seed_coords_zyx_mm.shape[0],
            self._max_peak_expansion_batch_size,
        ):
            print(
                f"{b_start_idx}/{self.unique_seed_coords_zyx_mm.shape[0]}",
                end=" | ",
                flush=True,
            )
            b_end_idx = b_start_idx + self._max_peak_expansion_batch_size
            unique_seed_batch = self.unique_seed_coords_zyx_mm[b_start_idx:b_end_idx]
            batch_peaks = self._fn_dipy_peak_finder_trilinear(unique_seed_batch)

            seeds_expanded_t_to_tp1, tangent_expanded_tp1 = self._fn_rk4_expansion(
                unique_seed_batch,
                seed_peak_vals=batch_peaks.peaks,
                theta_peak=batch_peaks.theta,
                phi_peak=batch_peaks.phi,
                valid_peak_mask=batch_peaks.valid_peak_mask,
            )
            self.seed_buffer.append(
                seeds_expanded_t_to_tp1.detach().to(device=device, dtype=torch.float32)
            )
            self.tangent_buffer.append(
                tangent_expanded_tp1.detach().to(device=device, dtype=torch.float32)
            )

        self.seed_buffer = torch.cat(self.seed_buffer, dim=1).to(device)
        self.tangent_buffer = torch.cat(self.tangent_buffer, dim=0).to(device)

    @staticmethod
    def _dipy_peak_finder_fn_linear_interp_zyx(
        target_coords_mm_zyx: torch.Tensor,
        fodf_coeffs_brain_vol: torch.Tensor,
        affine_vox2mm: torch.Tensor,
        seed_sphere_theta: torch.Tensor,
        seed_sphere_phi: torch.Tensor,
        sh_order: int,
        fodf_sample_min_val: Optional[float] = None,
        fodf_sample_min_quantile_thresh: Optional[float] = None,
        **dipy_peak_directions_kwargs,
    ) -> "pitn.tract.peak.PeaksContainer":
        # Initial interpolation of fodf coefficients at the target points.
        pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
            target_coords_mm_zyx,
            fodf_coeff_vol=fodf_coeffs_brain_vol,
            affine_vox2mm=affine_vox2mm,
        )

        # Transform to fodf spherical samples.
        target_sphere_samples = pitn.odf.sample_sphere_coords(
            pred_sample_fodf_coeffs,
            theta=seed_sphere_theta,
            phi=seed_sphere_phi,
            sh_order=sh_order,
        )

        # Threshold spherical function values.
        if fodf_sample_min_val is not None:
            target_sphere_samples = pitn.odf.thresh_fodf_samples_by_value(
                target_sphere_samples, fodf_sample_min_val
            )
        if fodf_sample_min_quantile_thresh is not None:
            target_sphere_samples = pitn.odf.thresh_fodf_samples_by_quantile(
                target_sphere_samples, fodf_sample_min_quantile_thresh
            )

        dipy_sphere = dipy.data.Sphere(
            theta=seed_sphere_theta.detach().cpu().numpy(),
            phi=seed_sphere_phi.detach().cpu().numpy(),
        )

        np_sphere_samples = target_sphere_samples.detach().cpu().numpy()

        # Set size to 50 arbitrarily; seems unlikely a real-world point would have 50 unique
        # peaks...can be set to the number of sphere samples, but that will almost certainly
        # never be used, and would be wasteful of memory.
        peak_vals = target_sphere_samples.new_zeros(target_sphere_samples.shape[0], 50)
        peak_theta = torch.clone(peak_vals)
        peak_phi = torch.clone(peak_vals)
        peak_valid_mask = torch.clone(peak_vals).bool()

        n_max_peaks = 0
        for i_sample in range(np_sphere_samples.shape[0]):
            directions, vals, sphere_indices = dipy.direction.peak_directions(
                np_sphere_samples[i_sample : i_sample + 1].flatten(),
                dipy_sphere,
                **dipy_peak_directions_kwargs,
            )
            n_peaks_i = directions.shape[0]

            peak_vals[i_sample, :n_peaks_i] = torch.from_numpy(vals).to(peak_vals)
            peak_idx_i = torch.from_numpy(sphere_indices).to(peak_vals.device).long()
            peak_theta[i_sample, :n_peaks_i] = torch.take(seed_sphere_theta, peak_idx_i)
            peak_phi[i_sample, :n_peaks_i] = torch.take(seed_sphere_phi, peak_idx_i)
            peak_valid_mask[i_sample, :n_peaks_i] = True

            n_max_peaks = max(n_max_peaks, n_peaks_i)

        peak_vals = peak_vals[:, :n_max_peaks]
        peak_theta = peak_theta[:, :n_max_peaks]
        peak_phi = peak_phi[:, :n_max_peaks]
        peak_valid_mask = peak_valid_mask[:, :n_max_peaks]

        return pitn.tract.peak.PeaksContainer(
            peak_vals, theta=peak_theta, phi=peak_phi, valid_peak_mask=peak_valid_mask
        )

    def sample_direction_seeds_sequential(self, start_idx: int, end_idx: int):

        if start_idx >= self.tangent_buffer.shape[0]:
            raise IndexError(
                f"ERROR: Start idx {start_idx} out of bounds"
                + f"{int(self.tangent_buffer.shape[0])}"
            )

        sample_coords_t_tp1 = self.seed_buffer[:, start_idx:end_idx].to(
            self.target_tensor
        )
        sample_tangent_tp1 = self.tangent_buffer[start_idx:end_idx].to(
            self.target_tensor
        )

        return sample_coords_t_tp1, sample_tangent_tp1


class SequentialSeedDirectionSampler:
    def __init__(
        self,
        max_batch_size: int,
        unique_seed_coords_zyx_mm: torch.Tensor,
        fodf_coeffs_brain_vol: torch.Tensor,
        affine_vox2mm: torch.Tensor,
        max_peaks_per_voxel: int,
        tracking_step_size: float,
        fn_zyx_direction_t2theta_phi,
        sh_order=8,
        **dipy_peak_finder_kwargs,
    ):
        self.max_batch_size = max_batch_size
        self.unique_seed_coords_zyx_mm = unique_seed_coords_zyx_mm
        self.target_tensor = self.unique_seed_coords_zyx_mm
        self._curr_unique_seed_coords_batch_position = 0
        self.max_peaks_per_voxel = max_peaks_per_voxel
        self._max_peak_expansion_batch_size = math.floor(
            self.max_batch_size / (self.max_peaks_per_voxel * 2)
        )
        self._max_peak_expansion_batch_size = max(
            1, self._max_peak_expansion_batch_size
        )

        self.fodf_coeffs_brain_vol = fodf_coeffs_brain_vol
        self.affine_vox2mm = affine_vox2mm
        self.tracking_step_size = tracking_step_size
        self.fn_zyx_direction_t2theta_phi = fn_zyx_direction_t2theta_phi

        self.sh_order = sh_order

        self._fn_dipy_peak_finder_trilinear = partial(
            self._dipy_peak_finder_fn_linear_interp_zyx,
            fodf_coeffs_brain_vol=self.fodf_coeffs_brain_vol,
            affine_vox2mm=self.affine_vox2mm,
            sh_order=self.sh_order,
            **dipy_peak_finder_kwargs,
        )

        self._fn_rk4_expansion = partial(
            expand_seeds_from_topk_peaks_rk4,
            max_peaks_per_voxel=self.max_peaks_per_voxel,
            step_size=self.tracking_step_size,
            fn_zyx_direction_t2theta_phi=self.fn_zyx_direction_t2theta_phi,
        )

        self.seed_buffer = list()
        self.tangent_buffer = list()
        self.buffer_item_lens = list()
        self.buffer_len_sums = list()
        self._buffer_max_idx = list()

    @staticmethod
    def _dipy_peak_finder_fn_linear_interp_zyx(
        target_coords_mm_zyx: torch.Tensor,
        fodf_coeffs_brain_vol: torch.Tensor,
        affine_vox2mm: torch.Tensor,
        seed_sphere_theta: torch.Tensor,
        seed_sphere_phi: torch.Tensor,
        sh_order: int,
        fodf_sample_min_val: Optional[float] = None,
        fodf_sample_min_quantile_thresh: Optional[float] = None,
        **dipy_peak_directions_kwargs,
    ) -> "pitn.tract.peak.PeaksContainer":
        # Initial interpolation of fodf coefficients at the target points.
        pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
            target_coords_mm_zyx,
            fodf_coeff_vol=fodf_coeffs_brain_vol,
            affine_vox2mm=affine_vox2mm,
        )

        # Transform to fodf spherical samples.
        target_sphere_samples = pitn.odf.sample_sphere_coords(
            pred_sample_fodf_coeffs,
            theta=seed_sphere_theta,
            phi=seed_sphere_phi,
            sh_order=sh_order,
        )

        # Threshold spherical function values.
        if fodf_sample_min_val is not None:
            target_sphere_samples = pitn.odf.thresh_fodf_samples_by_value(
                target_sphere_samples, fodf_sample_min_val
            )
        if fodf_sample_min_quantile_thresh is not None:
            target_sphere_samples = pitn.odf.thresh_fodf_samples_by_quantile(
                target_sphere_samples, fodf_sample_min_quantile_thresh
            )

        dipy_sphere = dipy.data.Sphere(
            theta=seed_sphere_theta.detach().cpu().numpy(),
            phi=seed_sphere_phi.detach().cpu().numpy(),
        )

        np_sphere_samples = target_sphere_samples.detach().cpu().numpy()

        # Set size to 50 arbitrarily; seems unlikely a real-world point would have 50 unique
        # peaks...can be set to the number of sphere samples, but that will almost certainly
        # never be used, and would be wasteful of memory.
        peak_vals = target_sphere_samples.new_zeros(target_sphere_samples.shape[0], 50)
        peak_theta = torch.clone(peak_vals)
        peak_phi = torch.clone(peak_vals)
        peak_valid_mask = torch.clone(peak_vals).bool()

        n_max_peaks = 0
        for i_sample in range(np_sphere_samples.shape[0]):
            directions, vals, sphere_indices = dipy.direction.peak_directions(
                np_sphere_samples[i_sample : i_sample + 1].flatten(),
                dipy_sphere,
                **dipy_peak_directions_kwargs,
            )
            n_peaks_i = directions.shape[0]

            peak_vals[i_sample, :n_peaks_i] = torch.from_numpy(vals).to(peak_vals)
            peak_idx_i = torch.from_numpy(sphere_indices).to(peak_vals.device).long()
            peak_theta[i_sample, :n_peaks_i] = torch.take(seed_sphere_theta, peak_idx_i)
            peak_phi[i_sample, :n_peaks_i] = torch.take(seed_sphere_phi, peak_idx_i)
            peak_valid_mask[i_sample, :n_peaks_i] = True

            n_max_peaks = max(n_max_peaks, n_peaks_i)

        peak_vals = peak_vals[:, :n_max_peaks]
        peak_theta = peak_theta[:, :n_max_peaks]
        peak_phi = peak_phi[:, :n_max_peaks]
        peak_valid_mask = peak_valid_mask[:, :n_max_peaks]

        return pitn.tract.peak.PeaksContainer(
            peak_vals, theta=peak_theta, phi=peak_phi, valid_peak_mask=peak_valid_mask
        )

    def _continuous_sample_batch_buffers(self, start_idx, end_idx):
        if (
            start_idx > self._buffer_max_idx[-1] + 1
            or end_idx > self._buffer_max_idx[-1] + 1
            or start_idx > end_idx
        ):
            raise RuntimeError(f"Error: Indices {start_idx}, {end_idx} are invalid")

        incl_end_idx = end_idx - 1
        start_buffer_idx = np.searchsorted(
            self._buffer_max_idx, start_idx, side="right"
        )
        if start_buffer_idx == 0:
            start_idx_in_start_buffer = start_idx
        elif start_buffer_idx == len(self._buffer_max_idx):
            start_idx_in_start_buffer = self.buffer_item_lens[-1] - 1
            start_buffer_idx = start_buffer_idx - 1
        else:
            start_idx_in_start_buffer = (
                start_idx - self._buffer_max_idx[start_buffer_idx - 1]
            )

        end_buffer_idx = np.searchsorted(
            self._buffer_max_idx, incl_end_idx, side="left"
        )
        if end_buffer_idx == 0:
            incl_end_idx_in_end_buffer = incl_end_idx
        else:
            incl_end_idx_in_end_buffer = (
                incl_end_idx - self.buffer_len_sums[end_buffer_idx - 1]
            )

        assert (start_idx_in_start_buffer >= 0) and (incl_end_idx_in_end_buffer >= 0)
        assert end_buffer_idx >= start_buffer_idx

        end_idx_in_end_buffer = incl_end_idx_in_end_buffer + 1

        # Start and end are in the same buffer item.
        if start_buffer_idx == end_buffer_idx:
            sample_coord_t_to_tp1 = self.seed_buffer[start_buffer_idx][
                :, start_idx_in_start_buffer:end_idx_in_end_buffer
            ]
            sample_tangent = self.tangent_buffer[start_buffer_idx][
                start_idx_in_start_buffer:end_idx_in_end_buffer
            ]
        else:
            sample_coord_start = self.seed_buffer[start_buffer_idx][
                :, start_idx_in_start_buffer:
            ]
            sample_coord_end = self.seed_buffer[end_buffer_idx][
                :, :end_idx_in_end_buffer
            ]

            sample_tangent_start = self.tangent_buffer[start_buffer_idx][
                start_idx_in_start_buffer:
            ]
            sample_tangent_end = self.tangent_buffer[end_buffer_idx][
                :end_idx_in_end_buffer
            ]

            intermediate_sample_coords = self.seed_buffer[
                start_buffer_idx + 1 : end_buffer_idx
            ]
            intermediate_sample_coords = (
                list()
                if len(intermediate_sample_coords) == 0
                else intermediate_sample_coords
            )
            intermediate_sample_tangent = self.tangent_buffer[
                start_buffer_idx + 1 : end_buffer_idx
            ]
            intermediate_sample_tangent = (
                list()
                if len(intermediate_sample_tangent) == 0
                else intermediate_sample_tangent
            )

            sample_coord_t_to_tp1 = torch.cat(
                [sample_coord_start] + intermediate_sample_coords + [sample_coord_end],
                dim=1,
            )
            sample_tangent = torch.cat(
                [sample_tangent_start]
                + intermediate_sample_tangent
                + [sample_tangent_end],
                dim=0,
            )

        return sample_coord_t_to_tp1.to(self.target_tensor), sample_tangent.to(
            self.target_tensor
        )

    def _append_to_buffer(self, seed_coord_batch, seed_tangent_batch):
        self.seed_buffer.append(seed_coord_batch.detach().cpu())
        self.tangent_buffer.append(seed_tangent_batch.detach().cpu())
        self.buffer_item_lens.append(seed_tangent_batch.shape[0])
        self._buffer_max_idx.append(np.sum(self.buffer_item_lens) - 1)
        self.buffer_len_sums.append(np.sum(self.buffer_item_lens))

    def sample_direction_seeds_sequential(self, start_idx: int, end_idx: int):

        assert end_idx > start_idx

        # Check if the starting index is out of bounds.
        if (
            self._curr_unique_seed_coords_batch_position
            >= self.unique_seed_coords_zyx_mm.shape[0]
            and start_idx > self._buffer_max_idx[-1]
        ):
            raise IndexError(
                f"ERROR: Starting idx {start_idx} "
                + f"is out of max range {self._buffer_max_idx[-1]}"
            )
        # Fill the buffers with at most `max_batch_size`-sized tensors until the
        # requested range can be sampled.
        if len(self.seed_buffer) == 0:
            current_end_idx = 1
        else:
            current_end_idx = self._buffer_max_idx[-1] + 1
        while current_end_idx < end_idx:
            unique_seed_batch = self.unique_seed_coords_zyx_mm[
                self._curr_unique_seed_coords_batch_position : self._curr_unique_seed_coords_batch_position
                + self._max_peak_expansion_batch_size
            ]
            self._curr_unique_seed_coords_batch_position = (
                self._curr_unique_seed_coords_batch_position
                + unique_seed_batch.shape[0]
            )
            batch_peaks = self._fn_dipy_peak_finder_trilinear(unique_seed_batch)

            seeds_expanded_t_to_tp1, tangent_expanded_tp1 = self._fn_rk4_expansion(
                unique_seed_batch,
                seed_peak_vals=batch_peaks.peaks,
                theta_peak=batch_peaks.theta,
                phi_peak=batch_peaks.phi,
                valid_peak_mask=batch_peaks.valid_peak_mask,
            )

            self._append_to_buffer(seeds_expanded_t_to_tp1, tangent_expanded_tp1)
            current_end_idx = self._buffer_max_idx[-1] + 1
            # The end idx exceeds the number of unique seeds, so only return what can be
            # sampled.
            if (
                self._curr_unique_seed_coords_batch_position
                >= self.unique_seed_coords_zyx_mm.shape[0]
            ):
                end_idx = min(end_idx, current_end_idx)
                break

        if start_idx > self._buffer_max_idx[-1]:
            raise IndexError(
                f"ERROR: Starting idx {start_idx} "
                + f"is out of max range {self._buffer_max_idx[-1]}"
            )
        (
            sample_coords_t_tp1,
            sample_tangents,
        ) = self._continuous_sample_batch_buffers(start_idx, end_idx)

        return sample_coords_t_tp1, sample_tangents


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
                steps=seeds_per_vox_axis + 2,
                dtype=affine_vox2mm.dtype,
                device=affine_vox2mm.device,
            )
        ]
        * 3,
        indexing="ij",
    )

    # Remove the endpoints to avoid intersection with neighboring voxels.
    within_vox_offsets = tuple(
        offsets[1:-1, 1:-1, 1:-1] for offsets in within_vox_offsets
    )
    within_vox_offsets = torch.stack(within_vox_offsets, -1).reshape(1, -1, 3)
    # If only 1 seed is requested per voxel, give the 0,0,0 offset instead. The linspace
    # function would return -0.5 if there is only 'steps=1'
    if seeds_per_vox_axis == 1:
        within_vox_offsets *= 0
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
