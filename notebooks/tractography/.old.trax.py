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
#     display_name: pitn
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

MODEL_SELECTION = "inr"
# MODEL_SELECTION = "tri-linear"

import collections
import datetime
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
# Disable cuda blocking **for debugging only!**
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
from icecream import ic
from jax import lax

import pitn

# Disable flags for debugging.
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

jax.config.update("jax_enable_x64", True)
# if MODEL_SELECTION.casefold() != "inr":
#     jax.config.update("jax_default_device", jax.devices()[1])
jax.config.update("jax_default_device", jax.devices()[1])
# jax.config.update("jax_default_matmul_precision", 32)


plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})
plt.rcParams.update({"image.cmap": "gray"})
plt.rcParams.update({"image.interpolation": "antialiased"})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)
torch.set_printoptions(sci_mode=False, threshold=100, linewidth=88)


def save_streamlines_to_tck(
    streamlines: list,
    affine_to_rasmm: torch.Tensor,
    save_dir: Path,
    tck_fname: str,
    ref_header,
):

    # Collect all valid streamlines and cut them at the stopping point.
    streamlines = torch.stack(streamlines, 1).squeeze(2)
    remove_streamline_mask = torch.isnan(streamlines).all(dim=1).any(dim=1)
    keep_streamline_mask = ~remove_streamline_mask
    streams = streamlines[keep_streamline_mask].detach().cpu().numpy()
    # tract_end_idx = np.argwhere(np.isnan(streams).any(2))[:, 1]
    batch_stream_list = np.split(streams, streams.shape[1], axis=1)
    all_tracts = list()
    for s in batch_stream_list:
        s = s.squeeze()
        if np.isnan(s).any():
            end_idx = np.argwhere(np.isnan(s).any(-1)).min()
            all_tracts.append(s[:end_idx])
        else:
            all_tracts.append(s)

    tracts = all_tracts

    # Create tractogram and save.
    sar_tracts = dipy.io.dpy.Streamlines(tracts)
    sar_tracto = dipy.io.streamline.Tractogram(
        sar_tracts, affine_to_rasmm=affine_sar2ras.cpu().numpy()
    )
    tracto = sar_tracto.to_world()
    # Get the header from an "un-re-oriented" fod volume and give to the tractogram.

    tracto = dipy.io.streamline.StatefulTractogram(
        tracto.streamlines,
        space=dipy.io.stateful_tractogram.Space.RASMM,
        reference=ref_header,
    )

    save_dir.mkdir(exist_ok=True, parents=True)
    fiber_fname = str(save_dir / tck_fname)
    # fiber_fname = f"/tmp/fibercup_single_vox_seed_test_trax.tck"
    print("Saving tractogram", flush=True)
    dipy.io.streamline.save_tck(tracto, fiber_fname)


def plot_buffer(buffer: torch.Tensor):
    streamlines = buffer.swapdims(1, 0).detach().cpu().numpy()
    # remove_streamline_mask = np.isnan(streamlines).all(0).any(0)
    # keep_streamline_mask = ~remove_streamline_mask
    # streams = streamlines[keep_streamline_mask]
    streams = list(streamlines)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=120)

    t = list()
    for s in streams:
        s = s.squeeze()
        if np.isnan(s).all():
            continue
        elif np.isnan(s).any():
            end_idx = np.argwhere(np.isnan(s).any(-1)).min()
            t.append(s[:end_idx])
            if (~np.isnan(s)).sum() != len(t[-1]) * 3:
                print("Warning: non-nans after nans in streamlines ", end="")
                ic((~np.isnan(s)).sum(), len(t[-1]), s.shape)
        else:
            t.append(s)
    for ax in range(3):
        a = axs[ax]
        sts_on_ax = [s[..., ax] for s in t]

        for i, s in enumerate(sts_on_ax):
            a.plot(s, label=i, alpha=0.7)
            a.scatter(np.arange(len(s)), s, alpha=0.7)
        a.set_ylabel(("x", "y", "z")[ax])

    return fig


# %%
# torch setup
# allow for CUDA usage, if available
if torch.cuda.is_available():
    # Pick only one device for the default, may use multiple GPUs for training later.
    # dev_idx = 0
    # dev_idx = 0 if MODEL_SELECTION.casefold() == "inr" else 1
    dev_idx = 1
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
    torch.set_float32_matmul_precision("medium")

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
tmp_results_dir = Path("/tmp") / Path("/data/srv/outputs/pitn/results/tmp")

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()
assert hcp_low_res_data_dir.exists()
assert hcp_low_res_fodf_dir.exists()
assert fibercup_fodf_dir.exists()
assert tmp_results_dir.exists()

# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
tmp_res_dir = Path(tmp_results_dir) / f"{ts}__tractography_{MODEL_SELECTION}"

# %%
network_weights_f = (
    Path("/data/srv/outputs/pitn/results")
    / "tmp"
    / "2023-02-22T01_29_25"
    / "state_dict_epoch_249_step_50000.pt"
)
assert network_weights_f.exists()

# %%

# Parameters
max_sh_order = 8

# Seed creation.
peaks_per_seed_vox = 1
# Total seeds per voxel will be `seeds_per_vox_axis`^3
seeds_per_vox_axis = 3
seed_batch_size = 2500

# Threshold parameter for peak finding in the seed voxels.
# Element-wise filtering of sphere samples.
fodf_sample_min_val = 0.05
fodf_sample_min_quantile_thresh = 0.001
dipy_relative_peak_threshold = 0.5
dipy_min_separation_angle = 25

# RK4 estimation
step_size = 0.2
alpha_exponential_moving_avg = 0.9
tracking_fodf_sample_min_val = 0.05
# Seems that MRtrix's Newton optimization also has a tolerance value of 0.001, max
# iterations of 100.
# <https://github.com/MRtrix3/mrtrix3/blob/f633dfd7e9080f71877ea6a4619dabdde99a0fb6/src/dwi/tractography/SIFT2/coeff_optimiser.cpp#L369>
grad_descent_kwargs = dict(
    tol=1e-3,
    # stepsize=4.0,
    maxiter=100,
    acceleration=True,
    # acceleration=False,
    implicit_diff=True,
    # implicit_diff=False,
    jit=True,
    # unroll=True,
)

# Stopping & invalidation criteria.
min_streamline_len = 20
max_streamline_len = 300
fa_min_threshold = 0.1
max_angular_thresh_rad = torch.pi / 3

# %% [markdown]
# ## Seed-Based Tractography Test

# %% [markdown]
# ### Data & Parameter Selection

# %%
# HCP or fibercup
dataset_selection = "HCP"
SUBJECT_ID = "581450"
FIVETT_MASK_FNAME = "postproc_5tt_parcellation.nii.gz"
# SEED_MASK_FNAME = "postproc_5tt_parcellation.nii.gz"
# selected_seed_vox_name = SEED_MASK_FNAME.replace(".nii.gz", "")
SEED_GM_WM_FNAME = "postproc_gm-wm-interface.nii.gz"
selected_seed_vox_name = SEED_GM_WM_FNAME.replace(".nii.gz", "")

# %%
# HCP Subject scan.
sample_fod_f = (
    hcp_low_res_fodf_dir / SUBJECT_ID / "T1w" / "postproc_wm_msmt_csd_fod.nii.gz"
)
fod_coeff_im = nib.load(sample_fod_f)
fod_coeff_im = nib.as_closest_canonical(fod_coeff_im)

sample_dwi_f = hcp_low_res_data_dir / SUBJECT_ID / "T1w" / "Diffusion" / "data.nii.gz"
dwi_im = nib.load(sample_dwi_f)
dwi_im = nib.as_closest_canonical(dwi_im)

mask_f = sample_fod_f.parent / "postproc_nodif_brain_mask.nii.gz"
mask_im = nib.load(mask_f)
mask_im = nib.as_closest_canonical(mask_im)

fa_f = sample_fod_f.parent / "fa.nii.gz"
fa_im = nib.load(fa_f)
fa_im = nib.as_closest_canonical(fa_im)

# If selecting GM-WM interface seeding:
seed_roi_mask_f = sample_fod_f.parent / SEED_GM_WM_FNAME
seed_roi_mask_im = nib.load(seed_roi_mask_f)
seed_roi_mask_im = nib.as_closest_canonical(seed_roi_mask_im)
seed_roi_mask_data = seed_roi_mask_im.get_fdata() > 0
gen = np.random.default_rng(1563972931212)
random_select_mask = gen.uniform(0, 1, size=seed_roi_mask_data.shape)
random_select_mask = random_select_mask < 1.0
seed_roi_mask_data = seed_roi_mask_data * random_select_mask

print("Num voxels in seed mask: ", seed_roi_mask_data.sum())
seed_roi_mask_im = nib.Nifti1Image(
    seed_roi_mask_data,
    affine=seed_roi_mask_im.affine,
    header=seed_roi_mask_im.header,
)

##### *If selecting white matter seed masks:*
# seed_roi_mask_f = sample_fod_f.parent / SEED_MASK_FNAME
# seed_roi_mask_im = nib.load(seed_roi_mask_f)
# seed_roi_mask_im = nib.as_closest_canonical(seed_roi_mask_im)
# fa_roa_csf_mask = seed_roi_mask_im.slicer[..., 3]
# # Negate to *exclude* the mask, rather than *include*
# fa_roa_csf_mask_data = ~(fa_roa_csf_mask.get_fdata().astype(bool))
# seed_roi_mask_im = seed_roi_mask_im.slicer[..., 2]
# seed_roi_mask_data = seed_roi_mask_im.get_fdata().astype(bool)
# # Reduce white matter seed area with binary opening to focus seeding towards the
# # center of the white matter tracts.
# seed_roi_mask_data = (
#     skimage.morphology.binary_opening(seed_roi_mask_data, skimage.morphology.ball(1))
#     * seed_roi_mask_data
# )
# # Randomly sub-select seed voxels.
# gen = np.random.default_rng(4900255)
# random_select_mask = gen.uniform(0, 1, size=seed_roi_mask_data.shape)
# random_select_mask = random_select_mask < 1.0
# seed_roi_mask_data = seed_roi_mask_data * random_select_mask
# print("Num voxels in seed mask: ", seed_roi_mask_data.sum())
# seed_roi_mask_im = nib.Nifti1Image(
#     seed_roi_mask_data,
#     affine=seed_roi_mask_im.affine,
#     header=seed_roi_mask_im.header,
# )

# CSF zeroing
fa_roa_f = sample_fod_f.parent / FIVETT_MASK_FNAME
fa_roa_im = nib.load(fa_roa_f)
fa_roa_im = nib.as_closest_canonical(fa_roa_im)
fa_roa_csf_mask = fa_roa_im.slicer[..., 3]
# Negate to *exclude* the mask, rather than *include*
fa_roa_csf_mask_data = ~(fa_roa_csf_mask.get_fdata().astype(bool))

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
    affine=affine_sar_vox2sar_mm.cpu().numpy(),
    header=fod_coeff_im.header,
)
sar_dwi = einops.rearrange(dwi_im.get_fdata(), "x y z b_grads -> z y x b_grads")
dwi_im = nib.Nifti1Image(
    sar_dwi, affine=affine_sar_vox2sar_mm.cpu().numpy(), header=dwi_im.header
)

sar_mask = einops.rearrange(mask_im.get_fdata().astype(bool), "x y z -> z y x")
mask_im = nib.Nifti1Image(
    sar_mask,
    affine=affine_sar_vox2sar_mm.cpu().numpy(),
    header=mask_im.header,
)
sar_fa = einops.rearrange(fa_im.get_fdata(), "x y z -> z y x")
# The FA may have nans from MRtrix...
if np.isnan(sar_fa).any():
    if np.isnan(sar_fa).sum() > 100:
        raise RuntimeError("ERROR: Too many nans in FA!")
    sar_fa = np.nan_to_num(sar_fa, nan=0)
# Zero-out CSF regions in the FA map to help reduce poor tracking due to partial volume
# effects.
sar_csf_roa = einops.rearrange(fa_roa_csf_mask_data, "x y z -> z y x")
sar_fa = sar_fa * sar_csf_roa
fa_im = nib.Nifti1Image(
    sar_fa,
    affine=affine_sar_vox2sar_mm.cpu().numpy(),
    header=fa_im.header,
)

sar_roi_seed_mask = einops.rearrange(
    seed_roi_mask_im.get_fdata().astype(bool), "x y z -> z y x"
)
seed_roi_mask_im = nib.Nifti1Image(
    sar_roi_seed_mask,
    affine=affine_sar_vox2sar_mm.cpu().numpy(),
    header=seed_roi_mask_im.header,
)


# %%
coeffs = fod_coeff_im.get_fdata()
coeffs = torch.from_numpy(coeffs).to(device)
# Move to channels-first layout.
coeffs = einops.rearrange(coeffs, "z y x coeffs -> coeffs z y x")
fod_coeff_im.uncache()

brain_mask = mask_im.get_fdata().astype(bool)
brain_mask = torch.from_numpy(brain_mask).to(device)
brain_mask = einops.rearrange(brain_mask, "z y x -> 1 z y x")
mask_im.uncache()

dwi = dwi_im.get_fdata()
dwi = torch.from_numpy(dwi).to(device)
dwi = einops.rearrange(dwi, "z y x b_grads -> b_grads z y x")
dwi = dwi * brain_mask
dwi_im.uncache()


fa = torch.from_numpy(fa_im.get_fdata()).to(device)
fa = einops.rearrange(fa, "z y x -> 1 z y x")
fa = fa * brain_mask
fa_im.uncache()

seed_roi_mask = torch.from_numpy(seed_roi_mask_im.get_fdata().astype(bool)).to(device)
seed_roi_mask = einops.rearrange(seed_roi_mask, "z y x -> 1 z y x")
seed_roi_mask = seed_roi_mask * brain_mask
seed_roi_mask_im.uncache()

# %%
seed_mask = torch.zeros_like(brain_mask).bool()

# #!DEBUG
# seed_roi_mask = 0 * seed_roi_mask
# seed_roi_mask[:, 16:17, 38:39, 33:34] = True
# roi_shape = seed_roi_mask.shape
# s_roi = einops.rearrange(seed_roi_mask, "1 z y x -> (z y x)")
# mask_idx = torch.where(s_roi)
# s_roi = s_roi * 0
# s_roi[(mask_idx[0][140:142],)] = True
# seed_roi_mask = einops.rearrange(
#     s_roi,
#     "(z y x) -> 1 z y x",
#     z=seed_roi_mask.shape[1],
#     y=seed_roi_mask.shape[2],
# )
# #!
seed_mask = (1 + seed_mask) * seed_roi_mask


# %%
seed_sphere = dipy.data.get_sphere("repulsion724")

seed_theta, seed_phi = pitn.odf.get_torch_sample_sphere_coords(
    seed_sphere, coeffs.device, coeffs.dtype
)

# %% [markdown]
# ### Tractography Reconstruction Loop - Trilinear Interpolation

# %%
# temp is x,y,z tuple of scipy.sparse.lil_arrays
# full streamline list is x,y,z tuple of scipy.sparse.csr_arrays
# After every seed batch, the remaining temp tracts are row-wise stacked onto the full
# streamline list with scipy.sparse.vstack()

# %%
sh_degrees = torch.arange(0, max_sh_order + 1, step=2).to(device)
sh_orders = torch.concatenate(
    [torch.arange(-n_, n_ + 1).to(device) for n_ in sh_degrees]
).to(device)
sh_degrees = torch.concatenate(
    [(torch.arange(-n_, n_ + 1).to(device) * 0) + n_ for n_ in sh_degrees]
).to(device)

peak_finder_fn_theta_phi_c2theta_phi = pitn.tract.peak.get_grad_descent_peak_finder_fn(
    sh_orders=sh_orders,
    sh_degrees=sh_degrees,
    degree_max=max_sh_order,
    batch_size=seed_batch_size,
    min_sphere_val=tracking_fodf_sample_min_val,
    **grad_descent_kwargs,
)

# %%
# Tri-linear interpolation functions.
def _fn_linear_interp_zyx_tangent_t2theta_phi(
    target_coords_mm_zyx: torch.Tensor,
    init_direction_theta_phi: Optional[torch.Tensor],
    fodf_coeffs_brain_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    fn_peak_finder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initial interpolation of fodf coefficients at the target points.
    pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
        target_coords_mm_zyx,
        fodf_coeff_vol=fodf_coeffs_brain_vol,
        affine_vox2mm=affine_vox2mm,
    )

    # The previous outgoing direction is not really the true "incoming" direction in
    # the new voxel, but it is located on the opposite hemisphere in the new voxel.
    # However, the peak finding locates the peak nearest the given initialization
    # direction, so it would just be two consecutive mirrorings on the sphere, which
    # is obviously identity.
    outgoing_theta, outgoing_phi = (
        init_direction_theta_phi[..., 0],
        init_direction_theta_phi[..., 1],
    )
    init_direction_theta_phi = (outgoing_theta, outgoing_phi)
    result_direction_theta_phi = fn_peak_finder(
        pred_sample_fodf_coeffs, init_direction_theta_phi
    )

    return result_direction_theta_phi


fn_linear_interp_zyx_tangent_t2theta_phi = partial(
    _fn_linear_interp_zyx_tangent_t2theta_phi,
    fodf_coeffs_brain_vol=coeffs,
    affine_vox2mm=affine_sar_vox2sar_mm,
    fn_peak_finder=peak_finder_fn_theta_phi_c2theta_phi,
)


def _fn_linear_interp_spatial_fodf_sample(
    coords_zyx: torch.Tensor,
    directions_theta_phi: torch.Tensor,
    fodf_coeffs_brain_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initial interpolation of fodf coefficients at the target points.
    pred_sample_fodf_coeffs = pitn.odf.sample_odf_coeffs_lin_interp(
        coords_zyx,
        fodf_coeff_vol=fodf_coeffs_brain_vol,
        affine_vox2mm=affine_vox2mm,
    )
    theta = directions_theta_phi[..., 0]
    phi = directions_theta_phi[..., 1]
    Y_basis = pitn.tract.peak.sh_basis_mrtrix3(
        theta=theta, phi=phi, batch_size=batch_size
    )
    Y_basis = einops.rearrange(Y_basis, "b sh_idx -> b sh_idx 1")
    pred_sample_fodf_coeffs = einops.rearrange(
        pred_sample_fodf_coeffs, "b sh_idx -> b 1 sh_idx"
    ).to(Y_basis)
    samples = torch.bmm(pred_sample_fodf_coeffs, Y_basis)
    samples.squeeze_()

    return samples


fn_linear_interp_spatial_fodf_sample = partial(
    _fn_linear_interp_spatial_fodf_sample,
    fodf_coeffs_brain_vol=coeffs,
    affine_vox2mm=affine_sar_vox2sar_mm,
)

# %%
# INR Interpolator
# Encoding model
class INREncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_channels=in_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
        )

        self.in_channels = in_channels
        self.interior_channels = interior_channels
        self.out_channels = out_channels

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.in_channels,
                self.interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    pitn.nn.layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                pitn.nn.layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.post_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.interior_channels,
                self.interior_channels,
                kernel_size=5,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.interior_channels,
                self.out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0)),
            torch.nn.AvgPool3d(kernel_size=2, stride=1),
            torch.nn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
        )
        # self.post_conv = torch.nn.Conv3d(
        #     self.interior_channels,
        #     self.out_channels,
        #     kernel_size=3,
        #     padding="same",
        #     padding_mode="reflect",
        # )

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)

        return y


class ReducedDecoder(torch.nn.Module):
    def __init__(
        self,
        context_v_features: int,
        out_features: int,
        m_encode_num_freqs: int,
        sigma_encode_scale: float,
        in_features=None,
    ):
        super().__init__()
        self.init_kwargs = dict(
            context_v_features=context_v_features,
            out_features=out_features,
            m_encode_num_freqs=m_encode_num_freqs,
            sigma_encode_scale=sigma_encode_scale,
            in_features=in_features,
        )

        # Determine the number of input features needed for the MLP.
        # The order for concatenation is
        # 1) ctx feats over the low-res input space, unfolded over a 3x3x3 window
        # ~~2) target voxel shape~~
        # 3) absolute coords of this forward pass' prediction target
        # 4) absolute coords of the high-res target voxel
        # ~~5) relative coords between high-res target coords and this forward pass'
        #    prediction target, normalized by low-res voxel shape~~
        # 6) encoding of relative coords
        self.context_v_features = context_v_features
        self.ndim = 3
        self.m_encode_num_freqs = m_encode_num_freqs
        self.sigma_encode_scale = torch.as_tensor(sigma_encode_scale)
        self.n_encode_features = self.ndim * 2 * self.m_encode_num_freqs
        self.n_coord_features = 2 * self.ndim + self.n_encode_features
        self.internal_features = self.context_v_features + self.n_coord_features

        self.in_features = in_features
        self.out_features = out_features

        # "Swish" function, recommended in MeshFreeFlowNet
        activate_cls = torch.nn.SiLU
        self.activate_fn = activate_cls(inplace=True)
        # Optional resizing linear layer, if the input size should be different than
        # the hidden layer size.
        if self.in_features is not None:
            self.lin_pre = torch.nn.Linear(self.in_features, self.context_v_features)
            self.norm_pre = None
        else:
            self.lin_pre = None
            self.norm_pre = None
        self.norm_pre = None

        # Internal hidden layers are two res MLPs.
        self.internal_res_repr = torch.nn.ModuleList(
            [
                pitn.nn.inr.SkipMLPBlock(
                    n_context_features=self.context_v_features,
                    n_coord_features=self.n_coord_features,
                    n_dense_layers=3,
                    activate_fn=activate_cls,
                )
                for _ in range(2)
            ]
        )
        self.lin_post = torch.nn.Linear(self.context_v_features, self.out_features)

    def encode_relative_coord(self, coords):
        c = einops.rearrange(coords, "b d x y z -> (b x y z) d")
        sigma = self.sigma_encode_scale.expand_as(c).to(c)[..., None]
        encode_pos = pitn.nn.inr.fourier_position_encoding(
            c, sigma_scale=sigma, m_num_freqs=self.m_encode_num_freqs
        )

        encode_pos = einops.rearrange(
            encode_pos,
            "(b x y z) d -> b d x y z",
            x=coords.shape[2],
            y=coords.shape[3],
            z=coords.shape[4],
        )
        return encode_pos

    def sub_grid_forward(
        self,
        context_val,
        context_coord,
        query_coord,
        context_vox_size,
        # query_vox_size,
        return_rel_context_coord=False,
    ):
        # Take relative coordinate difference between the current context
        # coord and the query coord.
        rel_context_coord = query_coord - context_coord
        # Also normalize to [0, 1) by subtracting the lower bound of differences
        # (- voxel size) and dividing by 2xupper bound (2 x voxel size).
        rel_norm_context_coord = (rel_context_coord - -context_vox_size) / (
            2 * context_vox_size
        )
        rel_norm_context_coord.round_(decimals=5)
        assert (rel_norm_context_coord >= 0).all() and (
            rel_norm_context_coord <= 1.0
        ).all()
        encoded_rel_norm_context_coord = self.encode_relative_coord(
            rel_norm_context_coord
        )

        # Perform forward pass of the MLP.
        if self.norm_pre is not None:
            context_val = self.norm_pre(context_val)
        context_feats = einops.rearrange(context_val, "b c x y z -> (b x y z) c")

        # q_vox_size = query_vox_size.expand_as(rel_norm_context_coord)
        coord_feats = (
            # q_vox_size,
            context_coord,
            query_coord,
            # rel_norm_context_coord,
            encoded_rel_norm_context_coord,
        )
        coord_feats = torch.cat(coord_feats, dim=1)
        spatial_layout = {
            "b": coord_feats.shape[0],
            "x": coord_feats.shape[2],
            "y": coord_feats.shape[3],
            "z": coord_feats.shape[4],
        }

        coord_feats = einops.rearrange(coord_feats, "b c x y z -> (b x y z) c")
        x_coord = coord_feats
        sub_grid_pred = context_feats

        if self.lin_pre is not None:
            sub_grid_pred = self.lin_pre(sub_grid_pred)
            sub_grid_pred = self.activate_fn(sub_grid_pred)

        for l in self.internal_res_repr:
            sub_grid_pred, x_coord = l(sub_grid_pred, x_coord)
        sub_grid_pred = self.lin_post(sub_grid_pred)
        sub_grid_pred = einops.rearrange(
            sub_grid_pred, "(b x y z) c -> b c x y z", **spatial_layout
        )
        if return_rel_context_coord:
            ret = (sub_grid_pred, rel_context_coord)
        else:
            ret = sub_grid_pred
        return ret

    def forward(
        self,
        context_v,
        context_spatial_extent,
        affine_context_vox2mm,
        # query_vox_size,
        query_coord,
    ) -> torch.Tensor:
        # if query_vox_size.ndim == 2:
        #     query_vox_size = query_vox_size[:, :, None, None, None]
        context_vox_size = torch.abs(
            context_spatial_extent[..., 1, 1, 1] - context_spatial_extent[..., 0, 0, 0]
        )
        context_vox_size = context_vox_size[:, :, None, None, None]

        batch_size = query_coord.shape[0]

        query_coord_in_context_fov = query_coord - torch.amin(
            context_spatial_extent, (2, 3, 4), keepdim=True
        )
        query_bottom_back_left_corner_coord = (
            query_coord_in_context_fov - (query_coord_in_context_fov % context_vox_size)
        ) + torch.amin(context_spatial_extent, (2, 3, 4), keepdim=True)
        context_vox_bottom_back_left_corner = pitn.affine.coord_transform_3d(
            query_bottom_back_left_corner_coord.movedim(1, -1),
            torch.linalg.inv(affine_context_vox2mm),
        )
        context_vox_bottom_back_left_corner = (
            context_vox_bottom_back_left_corner.movedim(-1, 1)
        )
        batch_vox_idx = einops.repeat(
            torch.arange(
                batch_size,
                dtype=context_vox_bottom_back_left_corner.dtype,
                device=context_vox_bottom_back_left_corner.device,
            ),
            "idx_b -> idx_b 1 i j k",
            idx_b=batch_size,
            i=query_coord.shape[2],
            j=query_coord.shape[3],
            k=query_coord.shape[4],
        )
        #     (context_vox_bottom_back_left_corner.shape[0], 1)
        #     + tuple(context_vox_bottom_back_left_corner.shape[2:])
        # )
        context_vox_bottom_back_left_corner = torch.cat(
            [batch_vox_idx, context_vox_bottom_back_left_corner], dim=1
        )
        context_vox_bottom_back_left_corner = (
            context_vox_bottom_back_left_corner.floor().long()
        )
        # Slice with a range to keep the "1" dimension in place.
        batch_vox_idx = context_vox_bottom_back_left_corner[:, 0:1]

        y_weighted_accumulate = None
        # Build the low-res representation one sub-window voxel index at a time.
        # The indicators specify if the current voxel index that surrounds the
        # query coordinate should be "off the center voxel" or not. If not, then
        # the center voxel (read: no voxel offset from the center) is selected
        # (for that dimension).
        for (
            corner_offset_i,
            corner_offset_j,
            corner_offset_k,
        ) in itertools.product((0, 1), (0, 1), (0, 1)):
            # Rebuild indexing tuple for each element of the sub-window
            sub_window_offset_ijk = query_bottom_back_left_corner_coord.new_tensor(
                [corner_offset_i, corner_offset_j, corner_offset_k]
            ).reshape(1, -1, 1, 1, 1)
            corner_offset_mm = sub_window_offset_ijk * context_vox_size

            i_idx = context_vox_bottom_back_left_corner[:, 1:2] + corner_offset_i
            j_idx = context_vox_bottom_back_left_corner[:, 2:3] + corner_offset_j
            k_idx = context_vox_bottom_back_left_corner[:, 3:4] + corner_offset_k
            context_val = context_v[
                batch_vox_idx.flatten(),
                :,
                i_idx.flatten(),
                j_idx.flatten(),
                k_idx.flatten(),
            ]
            context_val = einops.rearrange(
                context_val,
                "(b x y z) c -> b c x y z",
                b=batch_size,
                x=query_coord.shape[2],
                y=query_coord.shape[3],
                z=query_coord.shape[4],
            )
            context_coord = query_bottom_back_left_corner_coord + corner_offset_mm

            sub_grid_pred_ijk = self.sub_grid_forward(
                context_val=context_val,
                context_coord=context_coord,
                query_coord=query_coord,
                context_vox_size=context_vox_size,
                # query_vox_size=query_vox_size,
                return_rel_context_coord=False,
            )
            # Initialize the accumulated prediction after finding the
            # output size; easier than trying to pre-compute it.
            if y_weighted_accumulate is None:
                y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

            sub_window_offset_ijk_compliment = torch.abs(1 - sub_window_offset_ijk)
            sub_window_context_coord_compliment = (
                query_bottom_back_left_corner_coord
                + (sub_window_offset_ijk_compliment * context_vox_size)
            )
            w_sub_window_cube = torch.abs(
                sub_window_context_coord_compliment - query_coord
            )
            w_sub_window = einops.reduce(
                w_sub_window_cube, "b side_len i j k -> b 1 i j k", reduction="prod"
            ) / einops.reduce(
                context_vox_size, "b size 1 1 1 -> b 1 1 1 1", reduction="prod"
            )

            # Weigh this cell's prediction by the inverse of the distance
            # from the cell physical coordinate to the true target
            # physical coordinate. Normalize the weight by the inverse
            # "sum of the inverse distances" found before.

            # Accumulate weighted cell predictions to eventually create
            # the final prediction.
            y_weighted_accumulate += w_sub_window * sub_grid_pred_ijk
            # del sub_grid_pred_ijk

        y = y_weighted_accumulate

        return y


class INR_Interpolator:
    def __init__(
        self,
        dwi_brain_vol,
        brain_mask_vol,
        encoder,
        decoder,
        affine_vox2mm,
        fn_peak_finder,
    ):

        self.affine_vox2mm = affine_vox2mm.to(torch.float32)
        # Network was trained with xyz orientations, but there are zyx orientations in
        # the tractography code. So, all coordinates and images need to be rearranged
        # and flipped, then again for the output
        self.affine_vox2mm[:3, 3] = torch.flip(self.affine_vox2mm[:3, 3], dims=(-1,))
        self.decoder = decoder
        self.fn_peak_finder = fn_peak_finder

        if dwi_brain_vol.ndim == 4:
            dwi_brain_vol = dwi_brain_vol[None]
        dwi_brain_vol = dwi_brain_vol.to(torch.float32)
        if brain_mask_vol.ndim == 4:
            brain_mask_vol = brain_mask_vol[None]
        dwi_brain_vol = einops.rearrange(dwi_brain_vol, "b c z y x -> b c x y z")
        brain_mask_vol = einops.rearrange(brain_mask_vol, "b c z y x -> b c x y z")
        with torch.no_grad():
            self.encoded_ctx = encoder(dwi_brain_vol)
            self.encoded_ctx = self.encoded_ctx * brain_mask_vol
            self.ctx_spatial_extent = pitn.data.datasets._get_extent_world(
                brain_mask_vol[:, 0], self.affine_vox2mm
            )
            self.ctx_spatial_extent = self.ctx_spatial_extent[None].to(torch.float32)

    def spatial_fodf_sample(
        self,
        coords_mm_zyx: torch.Tensor,
        directions_theta_phi: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        theta = directions_theta_phi[..., 0]
        phi = directions_theta_phi[..., 1]
        Y_basis = pitn.tract.peak.sh_basis_mrtrix3(
            theta=theta, phi=phi, batch_size=batch_size
        )

        # Interpolation of fodf coefficients at the target points.
        with torch.no_grad():
            volumetric_target_coords = einops.rearrange(
                coords_mm_zyx, "b c -> 1 c 1 b 1"
            ).to(torch.float32)
            volumetric_target_coords = torch.flip(volumetric_target_coords, (1,))
            pred_sample_fodf_coeffs = self.decoder(
                context_v=self.encoded_ctx,
                context_spatial_extent=self.ctx_spatial_extent,
                affine_context_vox2mm=self.affine_vox2mm,
                query_coord=volumetric_target_coords,
            )
        pred_sample_fodf_coeffs = einops.rearrange(
            pred_sample_fodf_coeffs, "1 coeff 1 b 1 -> b coeff"
        )

        Y_basis = einops.rearrange(Y_basis, "b sh_idx -> b sh_idx 1")
        pred_sample_fodf_coeffs = einops.rearrange(
            pred_sample_fodf_coeffs, "b sh_idx -> b 1 sh_idx"
        )
        common_t = torch.promote_types(Y_basis.dtype, pred_sample_fodf_coeffs.dtype)
        Y_basis = Y_basis.to(common_t)
        pred_sample_fodf_coeffs = pred_sample_fodf_coeffs.to(common_t)
        samples = torch.bmm(pred_sample_fodf_coeffs, Y_basis)
        samples.squeeze_()

        return samples

    def __call__(
        self,
        target_coords_mm_zyx: torch.Tensor,
        init_direction_theta_phi: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial interpolation of fodf coefficients at the target points.
        with torch.no_grad():
            volumetric_target_coords = einops.rearrange(
                target_coords_mm_zyx, "b c -> 1 c 1 b 1"
            ).to(torch.float32)
            volumetric_target_coords = torch.flip(volumetric_target_coords, (1,))
            pred_sample_fodf_coeffs = self.decoder(
                context_v=self.encoded_ctx,
                context_spatial_extent=self.ctx_spatial_extent,
                affine_context_vox2mm=self.affine_vox2mm,
                query_coord=volumetric_target_coords,
            )
        pred_sample_fodf_coeffs = einops.rearrange(
            pred_sample_fodf_coeffs, "1 coeff 1 b 1 -> b coeff"
        ).to(target_coords_mm_zyx)

        # The previous outgoing direction is not really the true "incoming" direction in
        # the new voxel, but it is located on the opposite hemisphere in the new voxel.
        # However, the peak finding locates the peak nearest the given initialization
        # direction, so it would just be two consecutive mirrorings on the sphere, which
        # is obviously identity.
        outgoing_theta, outgoing_phi = (
            init_direction_theta_phi[..., 0],
            init_direction_theta_phi[..., 1],
        )
        init_direction_theta_phi = (outgoing_theta, outgoing_phi)
        result_direction_theta_phi = self.fn_peak_finder(
            pred_sample_fodf_coeffs, init_direction_theta_phi
        )
        # #!DEBUG
        # bugged_idx = (35, 728, 4570)
        # result_direction_theta_phi = self.fn_peak_finder(
        #     pred_sample_fodf_coeffs[
        #         (torch.tensor(bugged_idx).to(pred_sample_fodf_coeffs).long(),)
        #     ],
        #     (
        #         init_direction_theta_phi[0][
        #             (torch.tensor(bugged_idx).to(pred_sample_fodf_coeffs).long(),)
        #         ],
        #         init_direction_theta_phi[1][
        #             (torch.tensor(bugged_idx).to(pred_sample_fodf_coeffs).long(),)
        #         ],
        #     ),
        # )
        #!
        return result_direction_theta_phi


#!
encoder_init_kwargs = dict(
    in_channels=189,
    interior_channels=80,
    out_channels=128,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="relu",
)
decoder_init_kwargs = dict(
    context_v_features=128,
    in_features=encoder_init_kwargs["out_channels"],
    out_features=45,
    m_encode_num_freqs=36,
    sigma_encode_scale=3.0,
)
inr_system_state_dict = torch.load(network_weights_f)

encoder = INREncoder(**encoder_init_kwargs)
encoder.load_state_dict(inr_system_state_dict["encoder"])
encoder = encoder.to(device).eval()

decoder = ReducedDecoder(**decoder_init_kwargs)
decoder.load_state_dict(inr_system_state_dict["decoder"])
decoder = decoder.to(device).eval()
del inr_system_state_dict
fn_inr_interp_zyx_tangent_t2theta_phi = INR_Interpolator(
    dwi_brain_vol=dwi,
    brain_mask_vol=brain_mask,
    encoder=encoder,
    decoder=decoder,
    affine_vox2mm=affine_sar_vox2sar_mm,
    fn_peak_finder=peak_finder_fn_theta_phi_c2theta_phi,
)
del encoder
#!

# %% [markdown]
# ### Seeding


# %% [markdown]
# ### Primary Tractography Loop

# %%
# #! debug
# Create initial seeds and tangent/direction vectors.
init_unique_seeds = pitn.tract.seed.seeds_from_mask(
    seed_mask,
    seeds_per_vox_axis=seeds_per_vox_axis,
    affine_vox2mm=affine_sar_vox2sar_mm,
)

fn_zyx2theta_phi_seed_expansion = (
    fn_inr_interp_zyx_tangent_t2theta_phi
    if MODEL_SELECTION.casefold() == "inr"
    else fn_linear_interp_zyx_tangent_t2theta_phi
)

print("*" * 20, "Generating seeds", "*" * 20, flush=True)
seed_sampler = pitn.tract.seed.BatchSeedSequenceSampler(
    max_batch_size=seed_batch_size,
    max_peaks_per_voxel=peaks_per_seed_vox,
    unique_seed_coords_zyx_mm=init_unique_seeds,
    tracking_step_size=step_size,
    fodf_coeffs_brain_vol=coeffs,
    affine_vox2mm=affine_sar_vox2sar_mm,
    fn_zyx_direction_t2theta_phi=fn_zyx2theta_phi_seed_expansion,
    pytorch_device=device,
    # dipy peak finder kwargs
    seed_sphere_theta=seed_theta,
    seed_sphere_phi=seed_phi,
    fodf_sample_min_val=fodf_sample_min_val,
    fodf_sample_min_quantile_thresh=fodf_sample_min_quantile_thresh,
    relative_peak_threshold=dipy_relative_peak_threshold,
    min_separation_angle=dipy_min_separation_angle,
)

seeds_t_neg1_to_0, tangent_t0_zyx = seed_sampler.sample_direction_seeds_sequential(
    0, seed_batch_size
)

# Prep objects & initialize all state objects to t=0.

fn_direction_estimate = (
    fn_inr_interp_zyx_tangent_t2theta_phi
    if MODEL_SELECTION.casefold() == "inr"
    else fn_linear_interp_zyx_tangent_t2theta_phi
)

fn_fod_ampl_estimate = (
    fn_inr_interp_zyx_tangent_t2theta_phi.spatial_fodf_sample
    if MODEL_SELECTION.casefold() == "inr"
    else fn_linear_interp_spatial_fodf_sample
)

all_tracts = list()
streamlines = list()

max_steps = math.ceil(max_streamline_len / step_size) + 1
batch_size = tangent_t0_zyx.shape[0]
streamline_buffer = (
    torch.ones(
        max_steps, batch_size, 3, device=seeds_t_neg1_to_0.device, dtype=torch.float32
    )
    * torch.nan
)

v_t = torch.zeros(batch_size, dtype=torch.long, device=tangent_t0_zyx.device)
streamline_buffer.index_put_((v_t,), seeds_t_neg1_to_0[0].to(streamline_buffer))
v_t += 1
streamline_buffer.index_put_((v_t,), seeds_t_neg1_to_0[1].to(streamline_buffer))
v_t_diag_batch_selection = torch.arange(batch_size).to(v_t)

# t_max = 1e8
t_max = 1e6

full_streamline_status = (
    torch.ones(batch_size, dtype=torch.int8, device=seeds_t_neg1_to_0.device)
    * pitn.tract.stopping.CONTINUE
)
# At least one step has been made.
full_streamline_len = torch.zeros_like(full_streamline_status).float() + step_size
full_points_t = streamline_buffer[1]
full_tangent_t_theta_phi = torch.stack(
    pitn.tract.local.zyx2unit_sphere_theta_phi(tangent_t0_zyx), dim=-1
)
full_tangent_t_zyx = tangent_t0_zyx
full_points_tp1 = torch.zeros_like(full_points_t) * torch.nan

sampler_empty = False
curr_sampler_idx = batch_size

seeds_completed = 0
counter = 0
total_seeds = seed_sampler.tangent_buffer.shape[0]
tracks_so_far = 0
prev_print_tracks = 0
streamline_save_counter = 0
print("*" * 20, "Starting tractography loop", "*" * 20, flush=True)
print("Total Seeds: ", total_seeds, flush=True)
while (not sampler_empty) or pitn.tract.stopping.to_continue_mask(
    full_streamline_status
).any():
    # print(counter, end="|", flush=True)
    counter += 1

    to_continue = pitn.tract.stopping.to_continue_mask(full_streamline_status)

    points_t = full_points_t[to_continue]
    tangent_t_theta_phi = full_tangent_t_theta_phi[to_continue]
    tangent_t_zyx = full_tangent_t_zyx[to_continue]
    streamline_len = full_streamline_len[to_continue]
    status_t = full_streamline_status[to_continue]

    tangent_tp1_zyx = pitn.tract.local.gen_tract_step_rk4(
        points_t,
        init_direction_theta_phi=tangent_t_theta_phi,
        fn_zyx_direction_t2theta_phi=fn_direction_estimate,
        step_size=step_size,
    )
    tangent_tp1_theta_phi = torch.stack(
        pitn.tract.local.zyx2unit_sphere_theta_phi(tangent_tp1_zyx), -1
    )

    fodf_sample_point_t_direction_tp1 = fn_fod_ampl_estimate(
        points_t,
        directions_theta_phi=tangent_tp1_theta_phi,
        batch_size=seed_batch_size,
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

    # Update state variables based upon new streamline statuses.
    tmp_len = streamline_len + step_size
    statuses_tp1 = list()
    statuses_tp1.append(
        pitn.tract.stopping.scalar_vol_threshold(
            status_t,
            sample_coords_mm_zyx=points_tp1,
            scalar_min_threshold=fa_min_threshold,
            vol=fa,
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
    statuses_tp1.append(
        pitn.tract.stopping.scalar_vec_threshold(
            status_t,
            fodf_sample_point_t_direction_tp1,
            scalar_min_threshold=tracking_fodf_sample_min_val,
        )
    )
    # ad-hoc NaN catcher.
    statuses_tp1.append(
        torch.where(
            tangent_tp1_zyx.isnan().any(-1)
            & (status_t == pitn.tract.stopping.CONTINUE),
            pitn.tract.stopping.STOP,
            status_t,
        )
    )
    status_tp1 = pitn.tract.stopping.merge_status(status_t, *statuses_tp1)
    status_tp1 = pitn.tract.stopping.merge_status(
        status_tp1,
        pitn.tract.stopping.streamline_len_mm(
            status_tp1,
            tmp_len,
            min_len=min_streamline_len,
            max_len=max_streamline_len,
        ),
    )

    full_streamline_status_tp1 = full_streamline_status.masked_scatter(
        to_continue, status_tp1
    )

    # Remove any stopped tracks from the tp1 variables. We want the number of
    # elements in the "x_tp1" variables to equal the number of "True" values in
    # the "to_continue_tp1" array! Otherwise, the masked_scatter() will not put
    # the masked values where we want them.
    tp1_props_filter_mask = pitn.tract.stopping.to_continue_mask(status_tp1)
    points_tp1 = points_tp1[tp1_props_filter_mask]
    tangent_tp1_theta_phi = tangent_tp1_theta_phi[tp1_props_filter_mask]
    tangent_tp1_zyx = tangent_tp1_zyx[tp1_props_filter_mask]
    streamline_len_tp1 = tmp_len
    streamline_len_tp1 = streamline_len_tp1[tp1_props_filter_mask]
    status_tp1 = status_tp1[tp1_props_filter_mask]

    to_continue_tp1 = pitn.tract.stopping.to_continue_mask(full_streamline_status_tp1)

    assert points_tp1.shape[0] == to_continue_tp1.sum()

    full_points_tp1 = (full_points_tp1 * torch.nan).masked_scatter(
        to_continue_tp1[..., None], points_tp1.to(full_points_tp1)
    )

    # t <- t + 1
    v_t[to_continue_tp1] += 1
    if v_t.max() > t_max:
        break

    # Only write into the streamline buffer those points that continue, at their
    # timestep indices.
    streamline_buffer.index_put_(
        (v_t, v_t_diag_batch_selection),
        torch.where(to_continue_tp1[..., None], full_points_tp1, full_points_t),
    )

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
    full_streamline_status_change = full_streamline_status != full_streamline_status_tp1
    full_streamline_status = full_streamline_status_tp1

    # If any streamlines have stopped when they previously were not stopped, then store
    # those streamlines and sample more seeds to take their place.
    # to_continue_tp1 indicates stopped streamlines, while the streamline status
    # change indicates whether those buffer slots have been empty for more than one
    # iteration.
    if (~to_continue_tp1 & full_streamline_status_change).any():
        to_free_mask = ~to_continue_tp1 & full_streamline_status_change
        n_free = to_free_mask.sum().cpu().int().item()
        seeds_completed += n_free

        # Only store the valid/stopped streamlines, but empty out all the `to_free`
        # streamlines.
        to_store_valid = to_free_mask & (
            full_streamline_status != pitn.tract.stopping.INVALID
        )
        n_to_store = to_store_valid.sum().cpu().int().item()

        if n_to_store > 0:
            tracks_so_far += n_to_store

            if ((tracks_so_far % 10000) == 0) or (
                (tracks_so_far - prev_print_tracks) > 10000
            ):
                prev_print_tracks = tracks_so_far
                tot_seeds = seed_sampler.tangent_buffer.shape[0]
                print(
                    f"Tracts {tracks_so_far}, Seeds {seeds_completed}/{tot_seeds}",
                    end="...",
                    flush=True,
                )

            stopped_streamlines = torch.tensor_split(
                streamline_buffer[:, to_store_valid].cpu(), n_to_store, dim=1
            )
            streamlines.extend(stopped_streamlines)

            if len(streamlines) > 1000000:
                streamline_save_counter += 1

                ref_header = nib.as_closest_canonical(nib.load(sample_fod_f)).header
                fiber_fname = (
                    f"{SUBJECT_ID}_{dataset_selection}_"
                    + f"{selected_seed_vox_name}_{MODEL_SELECTION}"
                    + f"_trax_{streamline_save_counter:03d}.tck"
                )
                save_streamlines_to_tck(
                    streamlines,
                    affine_sar2ras,
                    save_dir=tmp_res_dir,
                    tck_fname=fiber_fname,
                    ref_header=ref_header,
                )
                streamlines.clear()

        full_streamline_len[to_free_mask] = 0.0
        full_points_t.masked_fill_(to_free_mask[:, None], torch.nan)
        full_tangent_t_theta_phi[to_free_mask] = torch.nan
        full_tangent_t_zyx.masked_fill_(to_free_mask[:, None], torch.nan)
        full_streamline_status[to_free_mask] = pitn.tract.stopping.STOP
        v_t[to_free_mask] = 0
        streamline_buffer.masked_fill_(to_free_mask[None, ..., None], torch.nan)

        # Only initialize new seeds if there are any to be sampled.
        if not sampler_empty:
            try:
                (
                    new_seeds_tneg1_to_t0,
                    new_tangent_t0_zyx,
                ) = seed_sampler.sample_direction_seeds_sequential(
                    curr_sampler_idx, curr_sampler_idx + n_free
                )
            except IndexError:
                sampler_empty = True
            else:
                n_new_seeds = new_tangent_t0_zyx.shape[0]
                curr_sampler_idx += n_new_seeds
                # It may be the case that the number of new seeds is less than the
                # number of available slots.
                to_refill_mask = to_free_mask.clone()
                to_refill_mask[torch.argwhere(to_refill_mask)[n_new_seeds:]] = False
                streamline_buffer[0, to_refill_mask] = new_seeds_tneg1_to_t0[0].to(
                    streamline_buffer
                )
                streamline_buffer[1, to_refill_mask] = new_seeds_tneg1_to_t0[1].to(
                    streamline_buffer
                )
                full_streamline_status[to_refill_mask] = pitn.tract.stopping.CONTINUE

                full_points_t[to_refill_mask] = new_seeds_tneg1_to_t0[1].to(
                    full_points_t
                )
                full_tangent_t_zyx[to_refill_mask] = new_tangent_t0_zyx
                full_tangent_t_theta_phi[to_refill_mask] = torch.stack(
                    pitn.tract.local.zyx2unit_sphere_theta_phi(new_tangent_t0_zyx), -1
                )
                v_t[to_refill_mask] = 1
                full_streamline_len[to_refill_mask] = step_size

print()
print("*" * 20, "Finished tractography loop", "*" * 20, flush=True)

# Collect all valid streamlines and cut them at the stopping point.
streamlines = torch.stack(streamlines, 1).squeeze(2)
remove_streamline_mask = torch.isnan(streamlines).all(dim=1).any(dim=1)
keep_streamline_mask = ~remove_streamline_mask
streams = streamlines[keep_streamline_mask].detach().cpu().numpy()
# tract_end_idx = np.argwhere(np.isnan(streams).any(2))[:, 1]
batch_stream_list = np.split(streams, streams.shape[1], axis=1)
all_tracts = list()
for s in batch_stream_list:
    s = s.squeeze()
    if np.isnan(s).any():
        end_idx = np.argwhere(np.isnan(s).any(-1)).min()
        all_tracts.append(s[:end_idx])
    else:
        all_tracts.append(s)

print("", end="", flush=True)

tracts = all_tracts

# Create tractogram and save.
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

# %%
streamline_save_counter += 1

ref_header = nib.as_closest_canonical(nib.load(sample_fod_f)).header
fiber_fname = (
    f"{SUBJECT_ID}_{dataset_selection}_"
    + f"{selected_seed_vox_name}_{MODEL_SELECTION}"
    + f"_trax_{streamline_save_counter:03d}.tck"
)
tmp_res_dir.mkdir(parents=True, exist_ok=True)
fiber_fname = str(tmp_res_dir / fiber_fname)
# fiber_fname = f"/tmp/fibercup_single_vox_seed_test_trax.tck"
print("Saving tractogram", flush=True)
dipy.io.streamline.save_tck(tracto, fiber_fname)


# %%
# for ax in range(3):
#     sts_on_ax = [s[:, ax] for s in tracto.streamlines]
#     plt.figure(dpi=120)
#     for i, s in enumerate(sts_on_ax):
#         plt.plot(s, label=i, lw=0.4, alpha=0.7)
#     plt.ylabel(("x", "y", "z")[ax])
#     # plt.legend()
#     plt.show()

# %%

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