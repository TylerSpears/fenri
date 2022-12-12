# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.10.8 ('base')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tractgraphy Evaluation for Continuous fODF Interpolation
#
# Code by:
#
# Tyler Spears - tas6hh@virginia.edu
#
# Dr. Tom Fletcher

# %% [markdown]
# ## Imports & Setup

# %%
# Imports
# Automatically re-import project-specific modules.
# %load_ext autoreload
# %autoreload 2

# imports
import collections
import copy
import datetime
import functools
import inspect
import io
import itertools
import math
import os
import pathlib
import pdb
import random
import shutil
import subprocess
import sys
import tempfile
import time
import timeit
import typing
import warnings
import zipfile
from pathlib import Path
from pprint import pprint as ppr

import dipy
import dipy.data
import dipy.direction
import dipy.direction.pmf
import dipy.io
import dipy.io.stateful_tractogram
import dipy.io.streamline
import dipy.reconst
import dipy.reconst.csdeconv
import dipy.reconst.shm
import dipy.tracking
import dipy.tracking.local_tracking
import dipy.tracking.stopping_criterion
import dipy.tracking.streamline
import dotenv
import einops

# visualization libraries
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import monai

# Data management libraries.
import nibabel as nib
import nibabel.processing

# Computation & ML libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
import skimage
import torch
import torch.nn.functional as F
import transforms3d
from box import Box
from dipy.viz import has_fury
from icecream import ic
from natsort import natsorted
from pytorch_lightning.lite import LightningLite

if has_fury:
    from dipy.viz import window, actor, colormap

import pitn

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})
plt.rcParams.update({"image.cmap": "gray"})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)
torch.set_printoptions(sci_mode=False, threshold=100, linewidth=88)

# %%
# Update notebook's environment variables with direnv.
# This requires the python-dotenv package, and direnv be installed on the system
# This will not work on Windows.
# NOTE: This is kind of hacky, and not necessarily safe. Be careful...
# Libraries needed on the python side:
# - os
# - subprocess
# - io
# - dotenv

# Form command to be run in direnv's context. This command will print out
# all environment variables defined in the subprocess/sub-shell.
command = "direnv exec {} /usr/bin/env".format(os.getcwd())
# Run command in a new subprocess.
proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd())
# Store and format the subprocess' output.
proc_out = proc.communicate()[0].strip().decode("utf-8")
# Use python-dotenv to load the environment variables by using the output of
# 'direnv exec ...' as a 'dummy' .env file.
dotenv.load_dotenv(stream=io.StringIO(proc_out), override=True)

# %%
# %%capture --no-stderr cap
# Capture output and save to log. Needs to be at the *very first* line of the cell.
# Watermark
# %load_ext watermark
# %watermark --author "Tyler Spears" --updated --iso8601  --python --machine --iversions --githash
if torch.cuda.is_available():
    # GPU information
    try:
        gpu_info = pitn.utils.system.get_gpu_specs()
        print(gpu_info)
    except NameError:
        print("CUDA Version: ", torch.version.cuda)
else:
    print("CUDA not in use, falling back to CPU")

# %%
# cap is defined in an ipython magic command
try:
    print(cap)
except NameError:
    pass

# %% [markdown]
# ## Experiment & Parameters Setup

# %%
p = Box(default_box=True)
# Experiment defaults, can be overridden in a config file.

# General experiment-wide params
###############################################
# p.experiment_name = "interpolation_baseline"
p.override_experiment_name = False
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"

p.tvt_split_files = list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))

# If a config file exists, override the defaults with those values.
try:
    if "PITN_CONFIG" in os.environ.keys():
        config_fname = Path(os.environ["PITN_CONFIG"])
    else:
        config_fname = pitn.utils.system.get_file_glob_unique(Path("."), r"config.*")
    f_type = config_fname.suffix.casefold()
    if f_type in {".yaml", ".yml"}:
        f_params = Box.from_yaml(filename=config_fname)
    elif f_type == ".json":
        f_params = Box.from_json(filename=config_fname)
    elif f_type == ".toml":
        f_params = Box.from_toml(filename=config_fname)
    else:
        raise RuntimeError()

    p.merge_update(f_params)

except:
    print("WARNING: Config file not loaded")
    pass

# Remove the default_box behavior now that params have been fully read in.
_p = Box(default_box=False)
_p.merge_update(p)
p = _p

# %%
# subj_ids = set()
# for f in p.tvt_split_files:
#     split = pd.read_csv(f)
#     split_subjs = set(split.subj_id.tolist())
#     subj_ids = subj_ids | split_subjs
p.subj_ids = [
    # 123117,
    201818,
]

# %% [markdown]
# ## Data Loading

# %%
hcp_full_res_data_dir = Path("/data/srv/data/pitn/hcp")
hcp_full_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")
hcp_low_res_data_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/vol")
hcp_low_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf")

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()
assert hcp_low_res_data_dir.exists()
assert hcp_low_res_fodf_dir.exists()

# %% [markdown]
# ### Datasets & DataLoader Creation

# %%
with warnings.catch_warnings(record=True) as warn_list:

    # Test dataset.
    # The test dataset won't be cached, as each image should only be loaded once.
    test_paths_dataset = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        lr_dwi_root_dir=hcp_low_res_data_dir,
        lr_fodf_root_dir=hcp_low_res_fodf_dir,
        transform=pitn.data.datasets.HCPfODFINRDataset.default_pre_sample_tf(
            0, skip_sample_mask=True
        ),
    )
    test_dataset = pitn.data.datasets.HCPfODFINRWholeVolDataset(
        test_paths_dataset,
        transform=pitn.data.datasets.HCPfODFINRWholeVolDataset.default_tf(),
    )

print("=" * 10)
print("Warnings caught:")
ws = "\n".join(
    [
        warnings.formatwarning(
            w.message, w.category, w.filename, w.lineno, w.file, w.line
        )
        for w in warn_list
    ]
)
ws = "\n".join(filter(lambda s: bool(s.strip()), ws.splitlines()))
print(ws, flush=True)
print("=" * 10)

# %%
test_dataloader = monai.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    num_workers=0,
    # num_workers=3,
    # prefetch_factor=3,
    # persistent_workers=True,
)

# %% [markdown]
# ## Tractography Tests

# %%
result_table = {"subj_id": list(), "mse_mean": list(), "mse_var": list()}
sampling_sphere = dipy.data.HemiSphere.from_sphere(
    dipy.data.get_sphere("repulsion724").subdivide(1)
)
gfa_stopping_threshold = 0.25
pmf_threshold = 0.1
max_angle = 30.0
step_size = 0.2
max_cross = 4
maxlen = 700
seed_density = 1


sh_transform_mat, _, _ = dipy.reconst.csdeconv.real_sh_descoteaux(
    sh_order=8,
    theta=sampling_sphere.theta,
    phi=sampling_sphere.phi,
    full_basis=False,
    legacy=False,
)


# %% [markdown]
# ### Linear Interpolation

# # %%
class LinearInterpPmfGen(dipy.direction.pmf.SHCoeffPmfGen):
    def __init__(self, subj_odf_coeffs, sphere, affine_vox2world, itk_im):
        self._subj_odf_coeffs = subj_odf_coeffs.copy().astype(np.double)
        dipy.direction.pmf.SHCoeffPmfGen.__init__(
            self, self._subj_odf_coeffs, sphere, basis_type=None
        )

        self._sphere = sphere
        self._theta = sphere.theta
        self._phi = sphere.phi
        self.im = itk_im
        self.affine_vox2world = affine_vox2world

        self.sh_transform_mat, _, _ = dipy.reconst.csdeconv.real_sh_descoteaux(
            sh_order=8, theta=self._theta, phi=self._phi, full_basis=False, legacy=False
        )

    def batched_sphere_sample(self, odf_coeffs):
        orig_spatial_shape = tuple(odf_coeffs.shape[:-1])
        coeffs = odf_coeffs.reshape(-1, odf_coeffs.shape[-1])
        samples = np.matmul(coeffs, self.sh_transform_mat.T[None])
        samples = samples.reshape(*orig_spatial_shape, -1)
        samples = np.clip(samples, 0, np.inf)
        return samples


#     def get_pmf(self, point):
#         # The `point` var is given in voxel space! Need to change to physical space
#         # first!
#         point = np.asarray(point)
#         phys_point = (self.affine_vox2world[:3, :3] @ point[:, None]) + self.affine_vox2world[
#             :3, 3:4
#         ]
#         phys_point = phys_point.flatten().astype(np.double)
#         interp_coeffs = self.im.EvaluateAtPhysicalPoint(phys_point, sitk.sitkLinear)
#         interp_coeffs = np.array(interp_coeffs)[None]
#         pmf = interp_coeffs @ self.sh_transform_mat.T
#         pmf = np.clip(pmf, 0, np.inf)
#         pmf = pmf.flatten()
#         return pmf


# # %%
# SHOW_WARNINGS = False

# RUN_NAME = "linear_interpolation_baseline"
# ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# # Break ISO format because many programs don't like having colons ':' in a filename.
# ts = ts.replace(":", "_")
# # tmp_res_dir = Path(p.tmp_results_dir) / "_".join([ts, RUN_NAME])
# # tmp_res_dir.mkdir(parents=True)

# lin_results = copy.deepcopy(result_table)
with warnings.catch_warnings(record=True) as warn_list:
    for subj_dict in test_dataloader:
        print("Loaded subject")
        subj_id = subj_dict["subj_id"][0]
        x = subj_dict["lr_fodf"][0].cpu()
        mask = subj_dict["mask"][0].bool().cpu()
        x_affine = subj_dict["affine_lrvox2acpc"][0].cpu().numpy().astype(np.double)

        x_fs_seg = subj_dict["lr_freesurfer_seg"].detach().cpu().numpy()
        x_fs_seg = x_fs_seg[0, 0]

        roi_dir = Path(
            "/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf/201818/T1w/rois/cst"
        )
        intern_capsule = nib.load(roi_dir / "cst_border_internal_capsule.nii.gz")
        seed_mask = intern_capsule.get_fdata().astype(bool)

        # seed_mask = (x_fs_seg >= 251) & (x_fs_seg <= 255)

        seeds = dipy.tracking.utils.seeds_from_mask(
            seed_mask,
            affine=intern_capsule.affine,
            density=seed_density,
        )
        print("Created seeds ", seeds.shape)

        x_transl, x_rot, x_zoom, x_shear = transforms3d.affines.decompose(x_affine)
        x_np = einops.rearrange(x, "c z y x -> z y x c").numpy()
        x = sitk.GetImageFromArray(x_np)
        x.SetSpacing(tuple(x_zoom))
        x.SetOrigin(tuple(x_transl))
        x.SetDirection(tuple(x_rot.flatten()))

        pmf_gen = LinearInterpPmfGen(
            x_np.astype(np.double),
            sphere=sampling_sphere,
            affine_vox2world=x_affine,
            itk_im=x,
        )
        odf_samples_vol = pmf_gen.batched_sphere_sample(x_np)
        their_pmf_gen = dipy.direction.pmf.SHCoeffPmfGen(
            x_np.astype(np.double), sampling_sphere, None
        )
        print("Sampled odf coefficients in data.")
        gfa_x = dipy.reconst.odf.gfa(odf_samples_vol)
        print("Created gen. fractional anisotropy image", gfa_x.shape)
        stopping_criterion = (
            dipy.tracking.stopping_criterion.ThresholdStoppingCriterion(
                gfa_x, gfa_stopping_threshold
            )
        )
        direction_getter = dipy.direction.ClosestPeakDirectionGetter(
            their_pmf_gen,
            max_angle=max_angle,
            sphere=sampling_sphere,
            pmf_threshold=pmf_threshold,
        )
        streamline_gen = dipy.tracking.local_tracking.LocalTracking(
            direction_getter,
            stopping_criterion=stopping_criterion,
            seeds=seeds,
            affine=x_affine,
            step_size=step_size,
            maxlen=maxlen,
            max_cross=max_cross,
        )
        print("Starting tractography")
        t_0 = timeit.default_timer()

        streamlines = dipy.tracking.streamline.Streamlines(streamline_gen)
        tractogram = dipy.io.stateful_tractogram.StatefulTractogram(
            streamlines,
            nib.Nifti1Image(x_np, affine=x_affine),
            dipy.io.stateful_tractogram.Space.RASMM,
        )
        print("Finished tracrography, saving tracks")
        t_1 = timeit.default_timer()
        print("Duration", time.strftime("%M:%S", time.localtime(t_1 - t_0)))

        # dipy.io.streamline.save_tck(tractogram, f"test_streamline_subj-{subj_id}.tck")
        dipy.io.streamline.save_tck(
            tractogram, f"linear_interp_streamline_cst_subj-{subj_id}.tck"
        )

# %%

# %% [markdown]
# ### INR

# %%
model_weight_f = Path(
    "/data/srv/outputs/pitn/results/runs/2022-12-06T21_40_10__fixed_ensemble_split_03/state_dict_epoch_174_step_35000.pt"
)


# %%
# Model definitions


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

    TARGET_COORD_EPSILON = 1e-6

    def __init__(
        self,
        context_v_features: int,
        out_features: int,
        m_encode_num_freqs: int,
        sigma_encode_scale: float,
        in_features=None,
    ):
        super().__init__()

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
        query_vox_size,
        return_rel_context_coord=False,
    ):
        # Take relative coordinate difference between the current context
        # coord and the query coord.
        # rel_context_coord = context_coord - query_coord + self.TARGET_COORD_EPSILON
        rel_context_coord = torch.clamp_min(
            context_coord - query_coord,
            (-context_vox_size / 2) + self.TARGET_COORD_EPSILON,
        )
        # Also normalize to [0, 1)
        # Coordinates are located in the center of the voxel. By the way
        # the context vector is being constructed surrounding the query
        # coord, the query coord is always within 1.5 x vox_size of the
        # context (low-res space) coordinate. So, subtract the
        # batch-and-channel-wise minimum, and divide by the known upper
        # bound.
        rel_norm_context_coord = (
            rel_context_coord
            - torch.amin(rel_context_coord, dim=(2, 3, 4), keepdim=True)
        ) / (1.5 * context_vox_size)
        assert (rel_norm_context_coord >= 0).all() and (
            rel_norm_context_coord < 1.0
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

    def equal_space_forward(self, context_v, context_spatial_extent, context_vox_size):
        return self.sub_grid_forward(
            context_val=context_v,
            context_coord=context_spatial_extent,
            query_coord=context_spatial_extent,
            context_vox_size=context_vox_size,
            query_vox_size=context_vox_size,
        )

    def forward(
        self,
        context_v,
        context_spatial_extent,
        query_vox_size,
        query_coord,
    ) -> torch.Tensor:
        if query_vox_size.ndim == 2:
            query_vox_size = query_vox_size[:, :, None, None, None]
        context_vox_size = torch.abs(
            context_spatial_extent[..., 1, 1, 1] - context_spatial_extent[..., 0, 0, 0]
        )
        context_vox_size = context_vox_size[:, :, None, None, None]

        # If the context space and the query coordinates are equal, then we are actually
        # just mapping within the same physical space to the same coordinates. So,
        # linear interpolation would just zero-out all surrounding predicted voxels,
        # and would be a massive waste of computation.
        if (
            (context_spatial_extent.shape == query_coord.shape)
            and torch.isclose(context_spatial_extent, query_coord).all()
            and torch.isclose(query_vox_size, context_vox_size).all()
        ):
            y = self.equal_space_forward(
                context_v=context_v,
                context_spatial_extent=context_spatial_extent,
                context_vox_size=context_vox_size,
            )
        # More commonly, the input space will not equal the output space, and the
        # prediction will need to be interpolated.
        else:
            batch_size = query_coord.shape[0]
            # Construct a grid of nearest indices in context space by sampling a grid of
            # *indices* given the coordinates in mm.
            # The channel dim is just repeated for every
            # channel, so that doesn't need to be in the idx grid.
            nearest_coord_idx = torch.stack(
                torch.meshgrid(
                    *[
                        torch.arange(0, context_spatial_extent.shape[i])
                        for i in (0, 2, 3, 4)
                    ],
                    indexing="ij",
                ),
                dim=1,
            ).to(context_spatial_extent)

            # Find the nearest grid point, where the batch+spatial dims are the
            # "channels."
            nearest_coord_idx = pitn.nn.inr.weighted_ctx_v(
                encoded_feat_vol=nearest_coord_idx,
                input_space_extent=context_spatial_extent,
                target_space_extent=query_coord,
                reindex_spatial_extents=True,
                sample_mode="nearest",
            ).to(torch.long)
            # Expand along channel dimension for raw indexing.
            nearest_coord_idx = einops.rearrange(
                nearest_coord_idx, "b dim i j k -> dim (b i j k)"
            )
            batch_idx = nearest_coord_idx[0]

            # Use the world coordinates to determine the necessary voxel coordinate
            # offsets such that the offsets enclose the query point.
            # World coordinate in the low-res input grid that is closest to the
            # query coordinate.
            phys_coords_0 = context_spatial_extent[
                batch_idx,
                :,
                nearest_coord_idx[1],
                nearest_coord_idx[2],
                nearest_coord_idx[3],
            ]

            phys_coords_0 = einops.rearrange(
                phys_coords_0,
                "(b x y z) c -> b c x y z",
                b=batch_size,
                c=query_coord.shape[1],
                x=query_coord.shape[2],
                y=query_coord.shape[3],
                z=query_coord.shape[4],
            )
            # Determine the quadrants that the query point lies in relative to the
            # context grid. We only care about the spatial/non-batch coordinates.
            surround_query_point_quadrants = (
                query_coord - self.TARGET_COORD_EPSILON - phys_coords_0
            )
            # 3 x batch_and_spatial_size
            # The signs of the "query coordinate - grid coordinate" should match the
            # direction the indexing should go for the nearest voxels to the query.
            surround_offsets_vox = einops.rearrange(
                surround_query_point_quadrants.sign(), "b dim i j k -> dim (b i j k)"
            ).to(torch.int8)
            del surround_query_point_quadrants

            # Now, find sum of distances to normalize the distance-weighted weight vector
            # for in-place 'linear interpolation.'
            inv_dist_total = torch.zeros_like(phys_coords_0)
            inv_dist_total = (inv_dist_total[:, 0])[:, None]
            surround_offsets_vox_volume_order = einops.rearrange(
                surround_offsets_vox,
                "dim (b i j k) -> b dim i j k",
                b=batch_size,
                i=query_coord.shape[2],
                j=query_coord.shape[3],
                k=query_coord.shape[4],
            )
            for (
                offcenter_indicate_i,
                offcenter_indicate_j,
                offcenter_indicate_k,
            ) in itertools.product((0, 1), (0, 1), (0, 1)):
                phys_coords_offset = torch.ones_like(phys_coords_0)
                phys_coords_offset[:, 0] *= (
                    offcenter_indicate_i * surround_offsets_vox_volume_order[:, 0]
                ) * context_vox_size[:, 0]
                phys_coords_offset[:, 1] *= (
                    offcenter_indicate_j * surround_offsets_vox_volume_order[:, 1]
                ) * context_vox_size[:, 1]
                phys_coords_offset[:, 2] *= (
                    offcenter_indicate_k * surround_offsets_vox_volume_order[:, 2]
                ) * context_vox_size[:, 2]
                # phys_coords_offset = context_vox_size * phys_coords_offset
                phys_coords = phys_coords_0 + phys_coords_offset
                inv_dist_total += 1 / torch.linalg.vector_norm(
                    query_coord - phys_coords, ord=2, dim=1, keepdim=True
                )
            # Potentially free some memory here.
            del phys_coords
            del phys_coords_0
            del phys_coords_offset
            del surround_offsets_vox_volume_order

            y_weighted_accumulate = None
            # Build the low-res representation one sub-window voxel index at a time.
            # The indicators specify if the current voxel index that surrounds the
            # query coordinate should be "off the center voxel" or not. If not, then
            # the center voxel (read: no voxel offset from the center) is selected
            # (for that dimension).
            for (
                offcenter_indicate_i,
                offcenter_indicate_j,
                offcenter_indicate_k,
            ) in itertools.product((0, 1), (0, 1), (0, 1)):
                # Rebuild indexing tuple for each element of the sub-window
                i_idx = nearest_coord_idx[1] + (
                    offcenter_indicate_i * surround_offsets_vox[0]
                )
                j_idx = nearest_coord_idx[2] + (
                    offcenter_indicate_j * surround_offsets_vox[1]
                )
                k_idx = nearest_coord_idx[3] + (
                    offcenter_indicate_k * surround_offsets_vox[2]
                )
                context_val = context_v[batch_idx, :, i_idx, j_idx, k_idx]
                context_val = einops.rearrange(
                    context_val,
                    "(b x y z) c -> b c x y z",
                    b=batch_size,
                    x=query_coord.shape[2],
                    y=query_coord.shape[3],
                    z=query_coord.shape[4],
                )
                context_coord = context_spatial_extent[
                    batch_idx, :, i_idx, j_idx, k_idx
                ]
                context_coord = einops.rearrange(
                    context_coord,
                    "(b x y z) c -> b c x y z",
                    b=batch_size,
                    x=query_coord.shape[2],
                    y=query_coord.shape[3],
                    z=query_coord.shape[4],
                )

                sub_grid_pred_ijk = self.sub_grid_forward(
                    context_val=context_val,
                    context_coord=context_coord,
                    query_coord=query_coord,
                    context_vox_size=context_vox_size,
                    query_vox_size=query_vox_size,
                    return_rel_context_coord=False,
                )
                # Initialize the accumulated prediction after finding the
                # output size; easier than trying to pre-compute it.
                if y_weighted_accumulate is None:
                    y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

                # Weigh this cell's prediction by the inverse of the distance
                # from the cell physical coordinate to the true target
                # physical coordinate. Normalize the weight by the inverse
                # "sum of the inverse distances" found before.
                w = (
                    1
                    / torch.linalg.vector_norm(
                        query_coord - context_coord, ord=2, dim=1, keepdim=True
                    )
                ) / inv_dist_total

                # Accumulate weighted cell predictions to eventually create
                # the final prediction.
                y_weighted_accumulate += w * sub_grid_pred_ijk
                del sub_grid_pred_ijk

            y = y_weighted_accumulate

        return y


class INRSystemLoader(LightningLite):
    def run(self, checkpoint_state_dict_f):

        encoder = INREncoder(
            in_channels=189,
            interior_channels=80,
            out_channels=128,
            n_res_units=3,
            n_dense_units=3,
            activate_fn="relu",
        )
        encoder = self.setup(encoder)
        decoder = ReducedDecoder(
            context_v_features=128,
            in_features=128,
            out_features=45,
            m_encode_num_freqs=36,
            sigma_encode_scale=3.0,
        )
        decoder = self.setup(decoder)
        checkpoint = self.load(checkpoint_state_dict_f)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder = self.to_device(encoder)
        decoder = self.to_device(decoder)

        return encoder, decoder


# %%
class INRPredictionPmfGen(dipy.direction.pmf.SHCoeffPmfGen):
    def __init__(
        self,
        encoder_model,
        decoder_model,
        subj_dwi_data,
        subj_fodf_coeff_data,
        subj_dwi_coord_grid,
        sphere,
        affine_lrvox2world: np.ndarray,
        query_vox_size,
        seed_ref,
    ):
        self.device = encoder_model.device
        encoder_model.eval()
        self.decoder = decoder_model.to(self.device)
        self.decoder.eval()
        self.query_vox_size = (
            torch.as_tensor(query_vox_size).to(self.device).reshape(1, -1, 1, 1, 1)
        )
        self.subj_dwi_coord_grid = torch.as_tensor(subj_dwi_coord_grid).to(self.device)

        self.sphere = sphere
        self.theta, self.phi = pitn.odf.get_torch_sample_sphere_coords(
            self.sphere, self.device, self.subj_dwi_coord_grid.dtype
        )
        self.theta = self.theta.to(self.device)
        self.phi = self.phi.to(self.device)
        self.affine_lrvox2world = affine_lrvox2world
        self.sh_order = 8

        self.encoded_dwi = encoder_model(subj_dwi_data)
        self.subj_fodf_coeff_data = torch.as_tensor(subj_fodf_coeff_data).to(
            self.encoded_dwi
        )
        self.subj_fodf_coeff_data = einops.rearrange(
            self.subj_fodf_coeff_data, "1 coeff i j k -> 1 i j k coeff"
        )
        self.tck_counter = 0
        self.seed_ref = seed_ref

    def batched_sphere_sample(self, odf_coeffs):
        coeffs = torch.as_tensor(odf_coeffs).to(self.device)
        samples = pitn.odf.sample_sphere_coords(
            coeffs, theta=self.theta, phi=self.phi, sh_order=self.sh_order
        )
        return samples

    def get_pmf(self, point):
        # The `point` var is given in voxel space! Need to change to physical space
        # first!
        point = np.asarray(point)
        phys_point = (
            self.affine_lrvox2world[:3, :3] @ point[:, None]
        ) + self.affine_lrvox2world[:3, 3:4]
        # If the point is exactly at the center of a voxel, then just return the LR
        # odf coefficient without any prediction.
        if (point.astype(int) == point).all():
            pred_odf_coeff = self.subj_fodf_coeff_data[0][
                tuple(point.astype(int).flatten())
            ]
            pred_odf_coeff = pred_odf_coeff[None, :, None, None, None]
            self.tck_counter += 1
            seed_idx = np.where(np.prod(phys_point.T == self.seed_ref, axis=1))[
                0
            ].tolist()
            print(
                f"{seed_idx[0]}/{self.seed_ref.shape[0]}",
                end="...",
                flush=(self.tck_counter % 20) == 0,
            )

        else:
            phys_point = torch.from_numpy(phys_point.flatten().astype(np.double))
            query_coord = phys_point.reshape(1, -1, 1, 1, 1).to(self.device)

            pred_odf_coeff = self.decoder(
                self.encoded_dwi,
                context_spatial_extent=self.subj_dwi_coord_grid,
                query_vox_size=self.query_vox_size,
                query_coord=query_coord,
            )

        pmf = pitn.odf.sample_sphere_coords(
            pred_odf_coeff, theta=self.theta, phi=self.phi, sh_order=self.sh_order
        )
        pmf = pmf.flatten().detach().cpu().numpy().astype(np.double)

        return pmf


# %%
SHOW_WARNINGS = False

ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
# tmp_res_dir = Path(p.tmp_results_dir) / "_".join([ts, RUN_NAME])
# tmp_res_dir.mkdir(parents=True)

loader_system = INRSystemLoader(accelerator="gpu", devices=1, precision=32)
encoder, decoder = loader_system.run(model_weight_f)
print(encoder.device, decoder.device)
dev = encoder.device
# The target vox size isn't used right now, so just make a dummy vox size to satisfy the
# function args.
dummy_vox_size = torch.as_tensor([1.25, 1.25, 1.25]).reshape(1, -1).to(dev)

with warnings.catch_warnings(record=True) as warn_list:
    with torch.no_grad():
        for subj_dict in test_dataloader:
            subj_id = subj_dict["subj_id"][0]
            print("Loaded subject", subj_id)
            x = subj_dict["lr_dwi"].to(dev)
            x_np = subj_dict["lr_dwi"].cpu().numpy()[0]
            x_np = einops.rearrange(x_np, "c i j k -> i j k c")
            lr_mask = subj_dict["lr_mask"].bool().to(dev)
            lr_fodf = subj_dict["lr_fodf"].to(dev)
            x = x * lr_mask
            x_affine = subj_dict["affine_lrvox2acpc"][0].cpu().numpy().astype(np.double)
            x_coords = subj_dict["lr_extent_acpc"].to(dev)
            x_fs_seg = subj_dict["lr_freesurfer_seg"].detach().cpu().numpy()
            x_fs_seg = x_fs_seg[0, 0]
            # seed_mask = (x_fs_seg >= 251) & (x_fs_seg <= 255)

            roi_dir = Path(
                "/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf/201818/T1w/rois/cst"
            )
            intern_capsule = nib.load(roi_dir / "cst_border_internal_capsule.nii.gz")
            seed_mask = intern_capsule.get_fdata().astype(bool)

            # seed_mask = (x_fs_seg >= 251) & (x_fs_seg <= 255)

            seeds = dipy.tracking.utils.seeds_from_mask(
                seed_mask,
                affine=intern_capsule.affine,
                density=seed_density,
            )
            print("Created seeds ", seeds.shape)

            pmf_gen = INRPredictionPmfGen(
                encoder,
                decoder,
                subj_dwi_data=x,
                subj_fodf_coeff_data=lr_fodf,
                subj_dwi_coord_grid=x_coords,
                sphere=sampling_sphere,
                affine_lrvox2world=x_affine,
                query_vox_size=dummy_vox_size,
                seed_ref=seeds,
            )
            odf_samples_vol = pmf_gen.batched_sphere_sample(lr_fodf)
            odf_samples_vol = odf_samples_vol[0].cpu().numpy()
            odf_samples_vol = np.moveaxis(odf_samples_vol, 0, -1)
            print("Sampled odf coefficients in data.")
            gfa_x = dipy.reconst.odf.gfa(odf_samples_vol)
            gfa_x[np.isnan(gfa_x)] = 0.0
            print("Created gen. fractional anisotropy image.")
            stopping_criterion = (
                dipy.tracking.stopping_criterion.ThresholdStoppingCriterion(
                    gfa_x, gfa_stopping_threshold
                )
            )
            direction_getter = dipy.direction.ClosestPeakDirectionGetter(
                pmf_gen,
                max_angle=max_angle,
                sphere=sampling_sphere,
                pmf_threshold=pmf_threshold,
            )
            streamline_gen = dipy.tracking.local_tracking.LocalTracking(
                direction_getter,
                stopping_criterion=stopping_criterion,
                seeds=seeds,
                affine=x_affine,
                step_size=step_size,
                maxlen=maxlen,
                max_cross=max_cross,
            )
            print("Starting tractography")
            t_0 = timeit.default_timer()
            streamlines = dipy.tracking.streamline.Streamlines(streamline_gen)
            tractogram = dipy.io.stateful_tractogram.StatefulTractogram(
                streamlines,
                nib.Nifti1Image(x_np, affine=x_affine),
                dipy.io.stateful_tractogram.Space.RASMM,
            )
            print("Finished tracrography, saving tracks")
            t_1 = timeit.default_timer()
            print("Duration", time.strftime("%M:%S", time.localtime(t_1 - t_0)))

            # dipy.io.streamline.save_tck(tractogram, f"test_streamline_subj-{subj_id}.tck")
            dipy.io.streamline.save_tck(
                tractogram, f"test_streamline_cst_trained-INR_subj-{subj_id}.tck"
            )

# pd.DataFrame.from_dict(lin_results).to_csv(tmp_res_dir / f"run_results_{RUN_NAME}.csv")
# shutil.copytree(tmp_res_dir, Path(p.results_dir) / tmp_res_dir.name)

if SHOW_WARNINGS:
    print("=" * 10)
    print("Warnings caught:")
    ws = "\n".join(
        [
            warnings.formatwarning(
                w.message, w.category, w.filename, w.lineno, w.file, w.line
            )
            for w in warn_list
        ]
    )
    ws = "\n".join(filter(lambda s: bool(s.strip()), ws.splitlines()))
    print(ws, flush=True)
    print("=" * 10)


# %%

# %%
# df = pd.DataFrame.from_dict(lin_results)
# df.mse_mean.median()
