# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import collections
import functools
import itertools
import math
import os
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

# Change default behavior of jax GPU memory allocation.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"

# visualization libraries
# %matplotlib inline
from pprint import pprint

import dipy
import dipy.denoise
import dipy.io
import dipy.io.streamline
import dipy.reconst
import dipy.reconst.csdeconv
import dipy.reconst.shm
import dipy.viz
import einops
import functorch
import jax
import jax.config
import jax.dlpack

# Disable jit for debugging.
# jax.config.update("jax_disable_jit", True)
# Enable 64-bit precision.
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_default_matmul_precision", 32)
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import scipy
import seaborn as sns
import skimage
import torch
import torch.nn.functional as F
from jax import lax

import pitn

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})
plt.rcParams.update({"image.cmap": "gray"})
plt.rcParams.update({"image.interpolation": "antialiased"})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)
torch.set_printoptions(sci_mode=False, threshold=100, linewidth=88)

# %%
# torch setup
# allow for CUDA usage, if available
if torch.cuda.is_available():
    # Pick only one device for the default, may use multiple GPUs for training later.
    dev_idx = 0
    device = torch.device(f"cuda:{dev_idx}")
    print("CUDA Device IDX ", dev_idx)
    torch.cuda.set_device(device)
    print("CUDA Current Device ", torch.cuda.current_device())
    print("CUDA Device properties: ", torch.cuda.get_device_properties(device))
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # See
    # <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>
    # for details.

    # Activate cudnn benchmarking to optimize convolution algorithm speed.
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        print("CuDNN convolution optimization enabled.")
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

else:
    device = torch.device("cpu")
# keep device as the cpu
# device = torch.device('cpu')
print(device)

# %%
hcp_full_res_data_dir = Path("/data/srv/data/pitn/hcp")
hcp_full_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")
hcp_low_res_data_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/vol")
hcp_low_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf")
fibercup_fodf_dir = Path("/data/srv/outputs/fibercup/fiberfox_replication/B1-3/fodf")

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()
assert hcp_low_res_data_dir.exists()
assert hcp_low_res_fodf_dir.exists()
assert fibercup_fodf_dir.exists()

# %% [markdown]
# ## Seed-Based Tractography Test

# %% [markdown]
# ### Data & Parameter Selection

# %%
# HCP Subject scan.
# sample_fod_f = (
#     hcp_full_res_fodf_dir / "162329" / "T1w" / "postproc_wm_msmt_csd_fod.nii.gz"
# )
# fod_coeff_im = nib.load(sample_fod_f)
# fod_coeff_im = nib.as_closest_canonical(fod_coeff_im)
# print("Original shape", fod_coeff_im.shape)
# print("Original affine", fod_coeff_im.affine)
# mask_f = sample_fod_f.parent / "postproc_nodif_brain_mask.nii.gz"
# mask_im = nib.load(mask_f)
# mask_im = nib.as_closest_canonical(mask_im)
# white_matter_mask_f = sample_fod_f.parent / "postproc_5tt_parcellation.nii.gz"
# wm_mask_im = nib.load(white_matter_mask_f)
# wm_mask_im = nib.as_closest_canonical(wm_mask_im)
# wm_mask_im = wm_mask_im.slicer[..., 2]

# # Pre-select voxels of interest in RAS+ space for this specific subject.
# lobe_vox_idx = dict(
# # CC forceps minor, strong L-R uni-modal lobe
# cc_lr_lobe_idx = (55, 98, 53),
# # Dual-polar approx. equal volume fiber crossing
# lr_and_ap_bipolar_lobe_idx = (70, 106, 54),
# # Vox. adjacent to CST, tri-polar
# tri_polar_lobe_idx = (60, 68, 43),
# )

# %%
# Fibercup phantom data.
sample_fod_f = fibercup_fodf_dir / "B1-3_bval-1500_wm_fod_coeffs.nii.gz"
fod_coeff_im = nib.load(sample_fod_f)
fod_coeff_im = nib.as_closest_canonical(fod_coeff_im)
print("Original shape", fod_coeff_im.shape)
print("Original affine", fod_coeff_im.affine)
mask_f = fibercup_fodf_dir.parent / "dwi" / "B1-3_mask.nii.gz"
mask_im = nib.load(mask_f)
mask_im = nib.as_closest_canonical(mask_im)
white_matter_mask_f = mask_f
wm_mask_im = nib.load(white_matter_mask_f)
wm_mask_im = nib.as_closest_canonical(wm_mask_im)

# Pre-select voxels of interest in RAS+ space.
lobe_vox_idx = dict(
    unipole_lr_c_lobe_idx=(51, 40, 1),
    unipole_left_A_lobe_idx=(41, 34, 1),
    bipole_lr_left_A_lobe_idx=(45, 24, 1),
    three_crossing_right_A_lobe_idx=(24, 39, 1),
)
# %%
# Re-orient volumes from RAS to SAR (xyz -> zyx)
nib_affine_vox2ras_mm = fod_coeff_im.affine
affine_ras_vox2ras_mm = torch.from_numpy(nib_affine_vox2ras_mm).to(device)
ornt_ras = nib.orientations.io_orientation(nib_affine_vox2ras_mm)
ornt_sar = nib.orientations.axcodes2ornt(("S", "A", "R"))
ornt_ras2sar = nib.orientations.ornt_transform(ornt_ras, ornt_sar)
# We also need an affine that maps from SAR -> RAS
affine_sar2ras = nib.orientations.inv_ornt_aff(
    ornt_ras2sar, tuple(fod_coeff_im.shape[:-1])
)
affine_sar2ras = torch.from_numpy(affine_sar2ras).to(affine_ras_vox2ras_mm)
affine_ras2sar = torch.linalg.inv(affine_sar2ras)

# This essentially just flips the translation vector in the affine matrix. It may be
# "RAS" relative to the object/volume itself, but it is "SAR" relative to the original
# ordering of the dimensions in the data.
affine_sar_vox2sar_mm = affine_ras2sar @ (affine_ras_vox2ras_mm @ affine_sar2ras)

# Swap spatial dimensions, assign a new vox->world affine space.
sar_fod = einops.rearrange(fod_coeff_im.get_fdata(), "x y z coeffs -> z y x coeffs")
fod_coeff_im = nib.Nifti1Image(
    sar_fod,
    affine=(affine_sar_vox2sar_mm).cpu().numpy(),
    header=fod_coeff_im.header,
)
sar_mask = einops.rearrange(mask_im.get_fdata().astype(bool), "x y z -> z y x")
mask_im = nib.Nifti1Image(
    sar_mask,
    affine=(affine_sar_vox2sar_mm).cpu().numpy(),
    header=mask_im.header,
)
sar_wm_mask = einops.rearrange(wm_mask_im.get_fdata().astype(bool), "x y z -> z y x")
wm_mask_im = nib.Nifti1Image(
    sar_wm_mask,
    affine=(affine_sar_vox2sar_mm).cpu().numpy(),
    header=wm_mask_im.header,
)

print(fod_coeff_im.affine)
print(fod_coeff_im.shape)
print(mask_im.affine)
print(mask_im.shape)

# Flip the pre-selected voxels.
sar_vox_idx = pitn.affine.coord_transform_3d(
    affine_ras2sar.new_tensor(list(lobe_vox_idx.values())),
    affine_ras2sar,
)
sar_vox_idx = sar_vox_idx.int().cpu().tolist()
for i, k in enumerate(lobe_vox_idx.keys()):
    lobe_vox_idx[k] = tuple(sar_vox_idx[i])
# cc_lr_lobe_idx, lr_and_ap_bipolar_lobe_idx, tri_polar_lobe_idx = tuple(
#     sar_vox_idx.int().cpu().tolist()
# )
# cc_lr_lobe_idx = tuple(cc_lr_lobe_idx)
# lr_and_ap_bipolar_lobe_idx = tuple(lr_and_ap_bipolar_lobe_idx)
# tri_polar_lobe_idx = tuple(tri_polar_lobe_idx)
# print(cc_lr_lobe_idx, lr_and_ap_bipolar_lobe_idx, tri_polar_lobe_idx)

# %%
coeffs = fod_coeff_im.get_fdata()
coeffs = torch.from_numpy(coeffs).to(device)
fod_coeff_im.uncache()
# Move to channels-first layout.
coeffs = einops.rearrange(coeffs, "z y x coeffs -> coeffs z y x")
brain_mask = mask_im.get_fdata().astype(bool)
brain_mask = torch.from_numpy(brain_mask).to(device)
mask_im.uncache()
brain_mask = einops.rearrange(brain_mask, "z y x -> 1 z y x")
wm_mask = torch.from_numpy(wm_mask_im.get_fdata().astype(bool)).to(device)
wm_mask = einops.rearrange(wm_mask, "z y x -> 1 z y x")
wm_mask_im.uncache()
seed_mask = torch.zeros_like(brain_mask).bool()

select_vox_idx = lobe_vox_idx["unipole_left_A_lobe_idx"]
# select_vox_idx = lobe_vox_idx["unipole_lr_c_lobe_idx"]
# select_vox_idx = cc_lr_lobe_idx
# select_vox_idx = lr_and_ap_bipolar_lobe_idx
# select_vox_idx = tri_polar_lobe_idx
seed_mask[0, select_vox_idx[0], select_vox_idx[1], select_vox_idx[2]] = True

print(coeffs.shape)
print(brain_mask.shape)
print(seed_mask.shape)

# %%
# sphere = dipy.data.HemiSphere.from_sphere(
#     dipy.data.get_sphere("repulsion724")
# ).subdivide(1)
sphere = dipy.data.HemiSphere.from_sphere(
    dipy.data.get_sphere("repulsion724")
).subdivide(1)

theta, phi = pitn.odf.get_torch_sample_sphere_coords(
    sphere, coeffs.device, coeffs.dtype
)

nearest_sphere_samples = pitn.odf.adjacent_sphere_points_idx(theta=theta, phi=phi)
nearest_sphere_samples_idx = nearest_sphere_samples[0]
nearest_sphere_samples_valid_mask = nearest_sphere_samples[1]

# %%
max_sh_order = 8

# Element-wise filtering of sphere samples.
min_sample_pdf_threshold = 0.0001

# Threshold parameter for FMLS segmentation.
lobe_merge_ratio = 0.8
# Post-segmentation label filtering.
min_lobe_pdf_peak_threshold = 1e-4
min_lobe_pdf_integral_threshold = 0.05

# Seed creation.
peaks_per_seed_vox = 1
seed_batch_size = 2
# Total seeds per voxel will be `seeds_per_vox_axis`^3
seeds_per_vox_axis = 5

# RK4 estimation
step_size = 0.4
alpha_exponential_moving_avg = 0.3

# Stopping & invalidation criteria.
min_streamline_len = 10
max_streamline_len = 100
gfa_min_threshold = 0.25
max_angular_thresh_rad = torch.pi / 6


# %% [markdown]
# ### Tractography Reconstruction Loop - Trilinear Interpolation

# %%
# temp is x,y,z tuple of scipy.sparse.lil_arrays
# full streamline list is x,y,z tuple of scipy.sparse.csr_arrays
# After every seed batch, the remaining temp tracts are row-wise stacked onto the full
# streamline list with scipy.sparse.vstack()

# %%
def _fn_linear_interp_zyx_tangent_t2theta_phi(
    target_coords_mm_zyx: torch.Tensor,
    init_direction_theta_phi: Optional[torch.Tensor],
    fodf_coeffs_brain_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    sphere_samples_theta: torch.Tensor,
    sphere_samples_phi: torch.Tensor,
    sh_order: int,
    fodf_pdf_thresh_min: float,
    fmls_lobe_merge_ratio: float,
    lobe_fodf_pdf_filter_kwargs: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initial interpolation of fodf coefficients at the target points.
    pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
        target_coords_mm_zyx,
        fodf_coeff_vol=fodf_coeffs_brain_vol,
        affine_vox2mm=affine_vox2mm,
    )

    # Transform to fodf spherical samples.
    target_sphere_samples = pitn.odf.sample_sphere_coords(
        pred_sample_fodf_coeffs,
        theta=sphere_samples_theta,
        phi=sphere_samples_phi,
        sh_order=sh_order,
    )

    # Threshold spherical function values.
    target_sphere_samples = pitn.odf.thresh_fodf_samples_by_pdf(
        target_sphere_samples, fodf_pdf_thresh_min
    )

    # Segment lobes on the fodf samples in each voxel.
    lobe_labels = pitn.tract.peak.fmls_fodf_seg(
        target_sphere_samples,
        lobe_merge_ratio=fmls_lobe_merge_ratio,
        theta=sphere_samples_theta,
        phi=sphere_samples_phi,
    )

    # Refine the segmentation.
    lobe_labels = pitn.tract.peak.remove_fodf_labels_by_pdf(
        lobe_labels, target_sphere_samples, **lobe_fodf_pdf_filter_kwargs
    )

    # Find the peaks from the lobe segmentation.
    peaks = pitn.tract.peak.peaks_from_segment(
        lobe_labels,
        target_sphere_samples,
        theta_coord=sphere_samples_theta,
        phi_coord=sphere_samples_phi,
    )

    # If no initial direction is given, or the initial direction vector is 0, then
    # just find the largest peak.
    if (init_direction_theta_phi is None) or (
        torch.as_tensor(init_direction_theta_phi) == 0
    ).all():
        largest_peak = pitn.tract.peak.topk_peaks(
            k=1,
            fodf_peaks=peaks.peaks,
            theta_peaks=peaks.theta,
            phi_peaks=peaks.phi,
            valid_peak_mask=peaks.valid_peak_mask,
        )
        result_direction_theta_phi = (largest_peak.theta, largest_peak.phi)
    # Otherwise if an initial direction vector is given, find the peak closest to that
    # incoming direction.
    else:
        fodf_peaks = peaks.peaks
        peak_coords_theta_phi = torch.stack([peaks.theta, peaks.phi], -1)
        valid_mask = peaks.valid_peak_mask
        # The tangent of the previous point cannot be directly translated onto the
        # new fodf spherical coordinate system. Rather, it must be flipped to the
        # opposite hemisphere. The `init_direction_theta_phi` is the previous point's
        # "outgoing" direction, but it must be changed to the current point's
        # "incoming" direction.
        outgoing_theta, outgoing_phi = (
            init_direction_theta_phi[..., 0],
            init_direction_theta_phi[..., 1],
        )
        incoming_direction_theta_phi = (
            pitn.tract.direction.project_sph_coord_opposite_hemisphere(
                outgoing_theta, outgoing_phi
            )
        )
        incoming_direction_theta_phi = torch.stack(incoming_direction_theta_phi, dim=-1)
        result_direction_theta_phi = pitn.tract.direction.closest_opposing_direction(
            incoming_direction_theta_phi,
            fodf_peaks=fodf_peaks,
            peak_coords_theta_phi=peak_coords_theta_phi,
            peaks_valid_mask=valid_mask,
        )

    return result_direction_theta_phi


fn_linear_interp_zyx_tangent_t2theta_phi = partial(
    _fn_linear_interp_zyx_tangent_t2theta_phi,
    fodf_coeffs_brain_vol=coeffs,
    affine_vox2mm=affine_sar_vox2sar_mm,
    sphere_samples_theta=theta,
    sphere_samples_phi=phi,
    sh_order=max_sh_order,
    fodf_pdf_thresh_min=min_sample_pdf_threshold,
    fmls_lobe_merge_ratio=lobe_merge_ratio,
    lobe_fodf_pdf_filter_kwargs={
        "pdf_peak_min": min_lobe_pdf_peak_threshold,
        "pdf_integral_min": min_lobe_pdf_integral_threshold,
    },
)


# %%
# Reduced version of the full interpolation function, to be called only when expanding
# the seed points at the start of streamline estimation.
def _peaks_only_fn_linear_interp_zyx(
    target_coords_mm_zyx: torch.Tensor,
    fodf_coeffs_brain_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    sphere_samples_theta: torch.Tensor,
    sphere_samples_phi: torch.Tensor,
    sh_order: int,
    fodf_pdf_thresh_min: float,
    fmls_lobe_merge_ratio: float,
    lobe_fodf_pdf_filter_kwargs: dict,
) -> pitn.tract.peak.PeaksContainer:
    # Initial interpolation of fodf coefficients at the target points.
    pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
        target_coords_mm_zyx,
        fodf_coeff_vol=fodf_coeffs_brain_vol,
        affine_vox2mm=affine_vox2mm,
    )

    # Transform to fodf spherical samples.
    target_sphere_samples = pitn.odf.sample_sphere_coords(
        pred_sample_fodf_coeffs,
        theta=sphere_samples_theta,
        phi=sphere_samples_phi,
        sh_order=sh_order,
    )

    # Threshold spherical function values.
    target_sphere_samples = pitn.odf.thresh_fodf_samples_by_pdf(
        target_sphere_samples, fodf_pdf_thresh_min
    )

    # Segment lobes on the fodf samples in each voxel.
    lobe_labels = pitn.tract.peak.fmls_fodf_seg(
        target_sphere_samples,
        lobe_merge_ratio=fmls_lobe_merge_ratio,
        theta=sphere_samples_theta,
        phi=sphere_samples_phi,
    )

    # Refine the segmentation.
    lobe_labels = pitn.tract.peak.remove_fodf_labels_by_pdf(
        lobe_labels, target_sphere_samples, **lobe_fodf_pdf_filter_kwargs
    )

    # Find the peaks from the lobe segmentation.
    peaks = pitn.tract.peak.peaks_from_segment(
        lobe_labels,
        target_sphere_samples,
        theta_coord=sphere_samples_theta,
        phi_coord=sphere_samples_phi,
    )

    return peaks


# Copy the static parameters from the full interplation function.
peaks_only_fn_linear_interp_zyx = partial(
    _peaks_only_fn_linear_interp_zyx,
    **fn_linear_interp_zyx_tangent_t2theta_phi.keywords,
)

# %%
# Create initial seeds and tangent/direction vectors.

seeds_t_neg1 = pitn.tract.seed.seeds_from_mask(
    seed_mask,
    seeds_per_vox_axis=seeds_per_vox_axis,
    affine_vox2mm=affine_sar_vox2sar_mm,
)
seed_peaks = peaks_only_fn_linear_interp_zyx(seeds_t_neg1)

(seeds_t_neg1_to_0, tangent_t0_zyx) = pitn.tract.seed.expand_seeds_from_topk_peaks_rk4(
    seeds_t_neg1,
    max_peaks_per_voxel=peaks_per_seed_vox,
    seed_peak_vals=seed_peaks.peaks,
    theta_peak=seed_peaks.theta,
    phi_peak=seed_peaks.phi,
    valid_peak_mask=seed_peaks.valid_peak_mask,
    step_size=step_size,
    fn_zyx_direction_t2theta_phi=fn_linear_interp_zyx_tangent_t2theta_phi,
)

# %%
# Handle stopping conditions.
with torch.no_grad():
    gfa_sampling_sphere = dipy.data.get_sphere("repulsion724")

    gfa_theta, gfa_phi = pitn.odf.get_torch_sample_sphere_coords(
        gfa_sampling_sphere, coeffs.device, coeffs.dtype
    )
    # Function applies non-negativity constraint.
    gfa_sphere_samples = pitn.odf.sample_sphere_coords(
        coeffs.cpu(),
        theta=gfa_theta.cpu(),
        phi=gfa_phi.cpu(),
        sh_order=8,
        sh_order_dim=0,
        mask=brain_mask.cpu(),
    )

    gfa = pitn.odf.gfa(gfa_sphere_samples, sphere_samples_idx=0).to(device)
    # Also, mask out only the white matter in the gfa! Otherwise, gfa can be high in
    # most places...
    gfa = gfa * wm_mask
    del gfa_sphere_samples, gfa_theta, gfa_phi, gfa_sampling_sphere

# %%
# #!DEBUG


def fn_only_right_zyx2theta_phi(
    target_coords_mm_zyx: torch.Tensor, init_direction_theta_phi: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    results_shape = tuple(target_coords_mm_zyx.shape[:-1])
    theta = target_coords_mm_zyx.new_ones(results_shape) * (torch.pi / 2)
    phi = torch.zeros_like(theta)

    return (theta, phi)


# %%
# Primary tracrography loop.
streamlines = list()
streamlines.append(seeds_t_neg1_to_0[0].clone())
streamlines.append(seeds_t_neg1_to_0[1].clone())

# t_max = 1e8
t_max = 300
t = 1

full_streamline_status = (
    torch.ones(
        seeds_t_neg1_to_0.shape[1], dtype=torch.int8, device=seeds_t_neg1_to_0.device
    )
    * pitn.tract.stopping.CONTINUE
)
# At least one step has been made.
full_streamline_len = torch.zeros_like(full_streamline_status).float() + step_size
full_points_t = seeds_t_neg1_to_0[1].clone()
full_tangent_t_theta_phi = torch.stack(
    pitn.tract.local.zyx2unit_sphere_theta_phi(tangent_t0_zyx), -1
)
full_tangent_t_zyx = tangent_t0_zyx
full_points_tp1 = torch.zeros_like(full_points_t) * torch.nan
# full_tangent_tp1_theta_phi = torch.zeros_like(full_tangent_t_theta_phi) * torch.nan
# full_tangent_tp1_zyx = torch.zeros_like(full_tangent_t_zyx) * torch.nan
while pitn.tract.stopping.to_continue_mask(full_streamline_status).any():

    to_continue = pitn.tract.stopping.to_continue_mask(full_streamline_status)

    points_t = full_points_t[to_continue]
    tangent_t_theta_phi = full_tangent_t_theta_phi[to_continue]
    tangent_t_zyx = full_tangent_t_zyx[to_continue]
    streamline_len = full_streamline_len[to_continue]
    status_t = full_streamline_status[to_continue]

    tangent_tp1_zyx = pitn.tract.local.gen_tract_step_rk4(
        points_t,
        init_direction_theta_phi=tangent_t_theta_phi,
        fn_zyx_direction_t2theta_phi=fn_linear_interp_zyx_tangent_t2theta_phi,
        # fn_zyx_direction_t2theta_phi=fn_only_right_zyx2theta_phi, #!DEBUG
        step_size=step_size,
    )
    ema_tangent_tp1_zyx = (
        alpha_exponential_moving_avg * tangent_tp1_zyx
        + (1 - alpha_exponential_moving_avg) * tangent_t_zyx
    )
    ema_tangent_tp1_zyx = (
        step_size
        * ema_tangent_tp1_zyx
        / torch.linalg.vector_norm(ema_tangent_tp1_zyx, ord=2, dim=-1, keepdim=True)
    )

    points_tp1 = points_t + ema_tangent_tp1_zyx
    tangent_tp1_zyx = ema_tangent_tp1_zyx
    tangent_tp1_theta_phi = torch.stack(
        pitn.tract.local.zyx2unit_sphere_theta_phi(tangent_tp1_zyx), -1
    )

    # Update state variables based upon new streamline statuses.
    tmp_len = streamline_len + step_size
    statuses_tp1 = list()
    statuses_tp1.append(
        pitn.tract.stopping.gfa_threshold(
            status_t,
            sample_coords_mm_zyx=points_tp1,
            gfa_min_threshold=gfa_min_threshold,
            gfa_vol=gfa,
            affine_vox2mm=affine_sar_vox2sar_mm,
        )
    )
    statuses_tp1.append(
        pitn.tract.stopping.angular_threshold(
            status_t, points_t, points_tp1, max_angular_thresh_rad
        )
    )
    statuses_tp1.append(
        pitn.tract.stopping.streamline_len_mm(
            status_t,
            tmp_len,
            min_len=min_streamline_len,
            max_len=max_streamline_len,
        )
    )
    status_tp1 = pitn.tract.stopping.merge_status(status_t, *statuses_tp1)
    full_streamline_status_tp1 = full_streamline_status.masked_scatter(
        to_continue, status_tp1
    )

    to_continue_tp1 = pitn.tract.stopping.to_continue_mask(full_streamline_status_tp1)
    full_points_tp1 = (full_points_tp1 * torch.nan).masked_scatter(
        to_continue_tp1[..., None], points_tp1
    )
    # full_points_tp1 = torch.where(to_continue_tp1[..., None], points_tp1, torch.nan)
    streamline_len_tp1 = tmp_len
    streamlines.append(full_points_tp1)

    # t <- t + 1
    print(t, end=" ")
    t += 1
    if t > t_max:
        break

    full_points_t = full_points_tp1
    full_tangent_t_theta_phi = (full_tangent_t_theta_phi * torch.nan).masked_scatter(
        to_continue_tp1[..., None], tangent_tp1_theta_phi
    )
    full_tangent_t_zyx = (full_tangent_t_zyx * torch.nan).masked_scatter(
        to_continue_tp1[..., None], tangent_tp1_zyx
    )
    full_streamline_len.masked_scatter_(
        to_continue_tp1,
        streamline_len_tp1,
    )
    full_streamline_status = full_streamline_status_tp1

# Shape `tract_seed x n_steps x 3`
streamlines = torch.stack(streamlines, 1)
print("", end="", flush=True)

# %%
full_tracts = np.split(streamlines.detach().cpu().numpy(), streamlines.shape[0], axis=0)
tracts = list()
for t, status in zip(full_tracts, full_streamline_status.cpu().numpy()):
    if status == pitn.tract.stopping.INVALID or np.isnan(t).all():
        continue
    tract_end_idx = np.argwhere(np.isnan(t.squeeze()))[:, 0].min()
    tract = t.squeeze()[:tract_end_idx]
    tracts.append(tract)

# tracts = [t.squeeze()[(~np.isnan(t.squeeze())).any(-1)] for t in tracts]
sar_tracts = dipy.io.dpy.Streamlines(tracts)
sar_tracto = dipy.io.streamline.Tractogram(
    sar_tracts, affine_to_rasmm=affine_sar2ras.cpu().numpy()
)
tracto = sar_tracto.to_world()
# Get the header from an "un-re-oriented" fod volume and give to the tractogram.

ref_header = nib.as_closest_canonical(nib.load(sample_fod_f)).header
tracto = dipy.io.streamline.StatefulTractogram(
    tracto.streamlines,
    space=dipy.io.stateful_tractogram.Space.RASMM,
    reference=ref_header,
)

# %%
dipy.io.streamline.save_tck(
    tracto, "/tmp/fibercup_unipole_bottom_C_repulsion_high-res_sphere_test_trax.tck"
)

# %%
# plt.plot(streamlines[3, :, 2].cpu().numpy(), label="x")
# plt.plot(streamlines[3, :, 1].cpu().numpy(), label="y")
# plt.plot(streamlines[3, :, 0].cpu().numpy(), label="z")

# plt.legend()

# %%
# im = nib.Nifti1Image(
#     gfa[0].cpu().swapdims(0, 2).numpy(), affine_ras_vox2ras_mm.cpu().numpy(), ref_header
# )

# nib.save(im, str(sample_fod_f.parent / "gfa.nii.gz"))
