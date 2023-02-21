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
#     display_name: pitn
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Continuous-Space Super-Resolution of fODFs in Diffusion MRI
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
import typing
import warnings
import zipfile
from pathlib import Path
from pprint import pprint as ppr

import aim
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
import skimage
import torch
import torch.nn.functional as F
import torchinfo
from box import Box
from icecream import ic
from lightning_fabric.fabric import Fabric
from natsort import natsorted

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
# torch setup
# allow for CUDA usage, if available
if torch.cuda.is_available():
    # Pick only one device for the default, may use multiple GPUs for training later.
    if "CUDA_PYTORCH_DEVICE_IDX" in os.environ.keys():
        dev_idx = int(os.environ["CUDA_PYTORCH_DEVICE_IDX"])
    else:
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
p.experiment_name = "test_second-attempt-trilin-weight_long-run"
p.override_experiment_name = False
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
p.train_val_test_split_file = random.choice(
    list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
)
p.aim_logger = dict(
    repo="aim://dali.cpe.virginia.edu:53800",
    experiment="PITN_INR",
    meta_params=dict(run_name=p.experiment_name),
    tags=("PITN", "INR", "HCP", "super-res", "dMRI"),
)
###############################################
p.train = dict(
    in_patch_size=(24, 24, 24),
    batch_size=4,
    samples_per_subj_per_epoch=40,
    max_epochs=200,
    dwi_recon_epoch_proportion=0.05,
)
# Optimizer kwargs for training.
p.train.optim.encoder.lr = 2e-4
p.train.optim.decoder.lr = 1e-4
p.train.optim.recon_decoder.lr = 5e-4
# Train dataloader kwargs.
p.train.dataloader = dict(num_workers=17, persistent_workers=True, prefetch_factor=3)

# Network/model parameters.
p.encoder = dict(
    interior_channels=80,
    out_channels=128,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="relu",
)
p.decoder = dict(
    context_v_features=128,
    in_features=p.encoder.out_channels,
    out_features=45,
    m_encode_num_freqs=36,
    sigma_encode_scale=3.0,
)


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
tvt_split = pd.read_csv(p.train_val_test_split_file)
p.train.subj_ids = natsorted(tvt_split[tvt_split.split == "train"].subj_id.tolist())
p.val = dict()
p.val.subj_ids = natsorted(tvt_split[tvt_split.split == "val"].subj_id.tolist())
p.test = dict()
p.test.subj_ids = natsorted(tvt_split[tvt_split.split == "test"].subj_id.tolist())

# Ensure that no test subj ids are in either the training or validation sets.
# However, we can have overlap between training and validation.
assert len(set(p.train.subj_ids) & set(p.test.subj_ids)) == 0
assert len(set(p.val.subj_ids) & set(p.test.subj_ids)) == 0

# %%
ppr(p.to_dict())

# %%
# Select which parameters to store in the aim meta-params.
p.aim_logger.meta_params.hparams = dict(
    batch_size=p.train.batch_size,
    patch_size=p.train.in_patch_size,
    samples_per_subj_per_epoch=p.train.samples_per_subj_per_epoch,
    max_epochs=p.train.max_epochs,
)
p.aim_logger.meta_params.data = dict(
    train_subj_ids=p.train.subj_ids,
    val_subj_ids=p.val.subj_ids,
    test_subj_ids=p.test.subj_ids,
)

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
# ### Create Patch-Based Training Dataset

# %%
DEBUG_TRAIN_DATA_SUBJS = 10
with warnings.catch_warnings(record=True) as warn_list:
    # pre_sample_ds = pitn.data.datasets.HCPfODFINRDataset(
    #     subj_ids=p.train.subj_ids,
    #     dwi_root_dir=hcp_full_res_data_dir,
    #     fodf_root_dir=hcp_full_res_fodf_dir,
    #     lr_dwi_root_dir=hcp_low_res_data_dir,
    #     lr_fodf_root_dir=hcp_low_res_fodf_dir,
    #     transform=None,
    # )

    # #! DEBUG
    print("DEBUG Train subject numbers")
    pre_sample_ds = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.train.subj_ids,  # [:DEBUG_TRAIN_DATA_SUBJS],
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        lr_dwi_root_dir=hcp_low_res_data_dir,
        lr_fodf_root_dir=hcp_low_res_fodf_dir,
        transform=None,
    )
    #!

    pre_sample_train_dataset = monai.data.CacheDataset(
        pre_sample_ds,
        transform=pre_sample_ds.default_pre_sample_tf(
            # Dilate by half the radius of one patch size.
            mask_dilate_radius=max(p.train.in_patch_size)
            // 4
        ),
        copy_cache=False,
        num_workers=4,
    )

train_dataset = pitn.data.datasets.HCPfODFINRPatchDataset(
    pre_sample_train_dataset,
    patch_func=pitn.data.datasets.HCPfODFINRPatchDataset.default_patch_func(
        spatial_size=p.train.in_patch_size,
        num_samples=p.train.samples_per_subj_per_epoch,
    ),
    samples_per_image=p.train.samples_per_subj_per_epoch,
    transform=pitn.data.datasets.HCPfODFINRPatchDataset.default_feature_tf(
        p.train.in_patch_size
    ),
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

# %% [markdown]
# ### Validation & Test Datasets

# %%
with warnings.catch_warnings(record=True) as warn_list:

    # #!DEBUG
    DEBUG_VAL_SUBJS = 3
    # Validation dataset.
    val_paths_dataset = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.val.subj_ids[:DEBUG_VAL_SUBJS],
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        lr_dwi_root_dir=hcp_low_res_data_dir,
        lr_fodf_root_dir=hcp_low_res_fodf_dir,
    )
    cached_val_dataset = monai.data.CacheDataset(
        val_paths_dataset,
        transform=val_paths_dataset.default_pre_sample_tf(0, skip_sample_mask=True),
        copy_cache=False,
        num_workers=4,
    )
    val_dataset = pitn.data.datasets.HCPfODFINRWholeVolDataset(
        cached_val_dataset,
        transform=pitn.data.datasets.HCPfODFINRWholeVolDataset.default_tf(),
    )

    # Test dataset.
    # The test dataset won't be cached, as each image should only be loaded once.
    test_paths_dataset = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.test.subj_ids,
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

# %% [markdown]
# ## Models

# %%
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
        # rel_context_coord = context_coord - query_coord + self.TARGET_COORD_EPSILON
        rel_context_coord = torch.clamp_min(
            context_coord - query_coord,
            (-context_vox_size / 2),
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


# %% [markdown]
# ## Training


# %%
def setup_logger_run(run_kwargs: dict, logger_meta_params: dict, logger_tags: list):
    aim_run = aim.Run(
        system_tracking_interval=None,
        log_system_params=True,
        capture_terminal_logs=True,
        **run_kwargs,
    )
    for k, v in logger_meta_params.items():
        aim_run[k] = v
    for v in logger_tags:
        aim_run.add_tag(v)

    return aim_run


# %%
def calc_grad_norm(model, norm_type=2):
    # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/5
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def batchwise_masked_mse(y_pred, y, mask):
    masked_y_pred = y_pred.clone()
    masked_y = y.clone()
    masked_y_pred[mask] = torch.nan
    masked_y[mask] = torch.nan
    se = F.mse_loss(masked_y_pred, masked_y, reduction="none")
    se = se.reshape(se.shape[0], -1)
    mse = torch.nanmean(se, dim=1)
    return mse


def validate_stage(
    fabric,
    encoder,
    decoder,
    val_dataloader,
    step: int,
    epoch: int,
    aim_run,
    val_viz_subj_id,
):
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Set up validation metrics to track for this validation run.
        val_metrics = {"mse": list()}
        for batch_dict in val_dataloader:
            subj_id = batch_dict["subj_id"]
            if len(subj_id) == 1:
                subj_id = subj_id[0]
            if val_viz_subj_id is None:
                val_viz_subj_id = subj_id
            x = batch_dict["lr_dwi"]
            x_coords = batch_dict["lr_extent_acpc"]
            x_affine_vox2mm = batch_dict["affine_lrvox2acpc"]
            # lr_fodf = batch_dict["lr_fodf"]
            y = batch_dict["fodf"]
            y_mask = batch_dict["mask"].to(torch.bool)
            y_coords = batch_dict["extent_acpc"]
            # y_vox_size = torch.atleast_2d(batch_dict["vox_size"])

            ctx_v = encoder(x)
            # pred_fodf = decoder(
            #     context_v=ctx_v,
            #     context_spatial_extent=x_coords,
            #     query_vox_size=y_vox_size,
            #     query_coord=y_coords,
            # )
            # Whole-volume inference is memory-prohibitive, so use a sliding
            # window inference method on the encoded volume.
            pred_fodf = monai.inferers.sliding_window_inference(
                y_coords,
                roi_size=(32, 32, 32),
                sw_batch_size=y_coords.shape[0],
                predictor=lambda q: decoder(
                    query_coord=q,
                    context_v=ctx_v,
                    context_spatial_extent=x_coords,
                    affine_context_vox2mm=x_affine_vox2mm,
                    # query_vox_size=y_vox_size,
                ),
                overlap=0,
                padding_mode="replicate",
            )

            y_mask_broad = torch.broadcast_to(y_mask, y.shape)
            # Calculate performance metrics
            mse_loss = batchwise_masked_mse(y, pred_fodf, mask=y_mask_broad)
            val_metrics["mse"].append(mse_loss.detach().cpu().flatten())

            # If visualization subj_id is in this batch, create the visual and log it.
            if subj_id == val_viz_subj_id:
                with mpl.rc_context({"font.size": 6.0}):
                    figsize = (8, 5)
                    width_pixels = 3 * (
                        pred_fodf.shape[2] + pred_fodf.shape[3] + pred_fodf.shape[4]
                    )
                    height_pixels = 5 * min(
                        pred_fodf.shape[2], pred_fodf.shape[3], pred_fodf.shape[4]
                    )
                    target_dpi = int(
                        np.ceil(
                            np.sqrt(
                                (width_pixels * height_pixels)
                                / ((0.95 * figsize[0]) * (0.9 * figsize[1]))
                            )
                        )
                    )
                    fig = plt.figure(dpi=target_dpi, figsize=figsize)
                    fig, _ = pitn.viz.plot_fodf_coeff_slices(
                        pred_fodf,
                        y,
                        pred_fodf - y,
                        fig=fig,
                        fodf_vol_labels=("Predicted", "Target", "Pred - GT"),
                        imshow_kwargs={
                            "interpolation": "nearest",
                            "cmap": "gray",
                        },
                    )
                    aim_run.track(
                        aim.Image(
                            fig,
                            caption=f"Val Subj {subj_id}, "
                            + f"MSE = {val_metrics['mse'][-1].item()}",
                            optimize=True,
                            quality=100,
                            format="png",
                        ),
                        name="sh_whole_volume",
                        context={"subset": "val"},
                        epoch=epoch,
                        step=step,
                    )
                    plt.close(fig)

                # Plot MSE as distributed over the SH orders.
                sh_coeff_labels = {
                    "idx": list(range(0, 45)),
                    "l": np.concatenate(
                        list(
                            map(
                                lambda x: np.array([x] * (2 * x + 1)),
                                range(0, 9, 2),
                            )
                        ),
                        dtype=int,
                    ).flatten(),
                }
                error_fodf = F.mse_loss(pred_fodf, y, reduction="none")
                error_fodf = einops.rearrange(
                    error_fodf, "b sh_idx x y z -> b x y z sh_idx"
                )
                error_fodf = error_fodf[
                    y_mask[:, 0, ..., None].broadcast_to(error_fodf.shape)
                ]
                error_fodf = einops.rearrange(
                    error_fodf, "(elem sh_idx) -> elem sh_idx", sh_idx=45
                )
                error_fodf = error_fodf.flatten().detach().cpu().numpy()
                error_df = pd.DataFrame.from_dict(
                    {
                        "MSE": error_fodf,
                        "SH_idx": np.tile(
                            sh_coeff_labels["idx"], error_fodf.shape[0] // 45
                        ),
                        "L Order": np.tile(
                            sh_coeff_labels["l"], error_fodf.shape[0] // 45
                        ),
                    }
                )
                with mpl.rc_context({"font.size": 6.0}):
                    fig = plt.figure(dpi=140, figsize=(6, 2))
                    sns.boxplot(
                        data=error_df,
                        x="SH_idx",
                        y="MSE",
                        hue="L Order",
                        linewidth=0.8,
                        showfliers=False,
                        width=0.85,
                        dodge=False,
                    )
                    aim_run.track(
                        aim.Image(fig, caption="MSE over SH orders", optimize=True),
                        name="mse_over_sh_orders",
                        epoch=epoch,
                        step=step,
                    )
                    plt.close(fig)
            fabric.print(f"MSE {val_metrics['mse'][-1].item()}")
            fabric.print("Finished validation subj ", subj_id)
            del pred_fodf

    val_metrics["mse"] = torch.cat(val_metrics["mse"])
    # Log metrics
    aim_run.track(
        {"mse": val_metrics["mse"].mean().numpy()},
        context={"subset": "val"},
        step=step,
        epoch=epoch,
    )

    encoder.train(mode=encoder_was_training)
    decoder.train(mode=decoder_was_training)
    return aim_run, val_viz_subj_id


# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
tmp_res_dir = Path(p.tmp_results_dir) / ts
tmp_res_dir.mkdir(parents=True)

# %%
fabric = Fabric(accelerator="gpu", devices=1, precision=32)
fabric.launch()
device = fabric.device

aim_run = setup_logger_run(
    run_kwargs={
        k: p.aim_logger[k] for k in set(p.aim_logger.keys()) - {"meta_params", "tags"}
    },
    logger_meta_params=p.aim_logger.meta_params.to_dict(),
    logger_tags=p.aim_logger.tags,
)
if "in_channels" not in p.encoder:
    in_channels = int(train_dataset[0]["lr_dwi"].shape[0])
else:
    in_channels = p.encoder.in_channels

# Wrap the entire training & validation loop in a try...except statement.
try:
    encoder = INREncoder(**{**p.encoder.to_dict(), **{"in_channels": in_channels}})
    # decoder = ContRepDecoder(**decoder_kwargs)
    decoder = ReducedDecoder(**p.decoder.to_dict())
    recon_decoder = INREncoder(
        in_channels=encoder.out_channels,
        interior_channels=48,
        out_channels=6,
        n_res_units=2,
        n_dense_units=2,
        activate_fn=p.encoder.activate_fn,
    )

    fabric.print(encoder)
    fabric.print(decoder)
    fabric.print(recon_decoder)

    optim_encoder = torch.optim.AdamW(
        encoder.parameters(), **p.train.optim.encoder.to_dict()
    )
    encoder, optim_encoder = fabric.setup(encoder, optim_encoder)
    optim_decoder = torch.optim.AdamW(
        decoder.parameters(), **p.train.optim.decoder.to_dict()
    )
    decoder, optim_decoder = fabric.setup(decoder, optim_decoder)
    optim_recon_decoder = torch.optim.AdamW(
        recon_decoder.parameters(), **p.train.optim.recon_decoder.to_dict()
    )
    recon_decoder, optim_recon_decoder = fabric.setup(
        recon_decoder, optim_recon_decoder
    )
    loss_fn = torch.nn.MSELoss(reduction="mean")
    recon_loss_fn = torch.nn.MSELoss(reduction="mean")

    train_dataloader = monai.data.DataLoader(
        train_dataset,
        batch_size=p.train.batch_size,
        shuffle=True,
        pin_memory=True,
        **p.train.dataloader.to_dict(),
    )
    val_dataloader = monai.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    val_viz_subj_id = None

    encoder.train()
    decoder.train()
    recon_decoder.train()
    # output_dir = Path(output_dir)
    losses = dict(
        loss=list(),
        epoch=list(),
        step=list(),
        encoder_grad_norm=list(),
        decoder_grad_norm=list(),
        recon_decoder_grad_norm=list(),
    )
    step = 0
    train_dwi_recon_epoch_proportion = p.train.dwi_recon_epoch_proportion
    train_recon = False

    epochs = p.train.max_epochs
    for epoch in range(epochs):
        fabric.print(f"\nEpoch {epoch}\n", "=" * 10)
        if epoch <= math.floor(epochs * train_dwi_recon_epoch_proportion):
            if not train_recon:
                train_recon = True
        else:
            train_recon = False

        for batch_dict in train_dataloader:

            x = batch_dict["lr_dwi"]
            x_coords = batch_dict["lr_patch_extent_acpc"]
            x_vox_size = torch.atleast_2d(batch_dict["lr_vox_size"])
            x_mask = batch_dict["lr_mask"].to(torch.bool)
            x_affine_vox2mm = batch_dict["affine_lr_patchvox2acpc"]

            y = batch_dict["fodf"]
            y_mask = batch_dict["mask"].to(torch.bool)
            y_coords = batch_dict["fr_patch_extent_acpc"]
            # y_vox_size = torch.atleast_2d(batch_dict["vox_size"])

            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            optim_recon_decoder.zero_grad()

            ctx_v = encoder(x)

            if not train_recon:
                y_mask_broad = torch.broadcast_to(y_mask, y.shape)
                pred_fodf = decoder(
                    context_v=ctx_v,
                    context_spatial_extent=x_coords,
                    affine_context_vox2mm=x_affine_vox2mm,
                    # query_vox_size=y_vox_size,
                    query_coord=y_coords,
                )
                loss_fodf = loss_fn(pred_fodf[y_mask_broad], y[y_mask_broad])
                loss_recon = y.new_zeros(1)
                recon_pred = None
            else:
                recon_pred = recon_decoder(ctx_v)
                # Index bvals to be 2 b=0s, 2 b=1000s, and 2 b=3000s.
                recon_y = x[:, (0, 1, 2, 21, 22, 23)]
                x_mask_broad = torch.broadcast_to(x_mask, recon_y.shape)
                loss_recon = recon_loss_fn(
                    recon_pred[x_mask_broad], recon_y[x_mask_broad]
                )
                loss_fodf = recon_y.new_zeros(1)
                pred_fodf = None

            loss = loss_fodf + loss_recon

            fabric.backward(loss)
            for model in (encoder, decoder, recon_decoder):
                if train_recon and model is decoder:
                    continue
                elif not train_recon and model is recon_decoder:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                    error_if_nonfinite=True,
                )
            optim_encoder.step()
            optim_decoder.step()
            optim_recon_decoder.step()

            encoder_grad_norm = calc_grad_norm(encoder)
            recon_decoder_grad_norm = (
                calc_grad_norm(recon_decoder) if train_recon else 0
            )
            decoder_grad_norm = calc_grad_norm(decoder) if not train_recon else 0
            to_track = {
                "loss": loss.detach().cpu().item(),
                "grad_norm_encoder": encoder_grad_norm,
            }
            # Depending on whether or not the reconstruction decoder is training,
            # select which metrics to track at this time.
            if train_recon:
                to_track = {
                    **to_track,
                    **{
                        "loss_recon": loss_recon.detach().cpu().item(),
                        "grad_norm_recon_decoder": recon_decoder_grad_norm,
                    },
                }
            else:
                to_track = {
                    **to_track,
                    **{
                        "loss_pred_fodf": loss_fodf.detach().cpu().item(),
                        "grad_norm_decoder": decoder_grad_norm,
                    },
                }
            aim_run.track(
                to_track,
                context={
                    "subset": "train",
                },
                step=step,
                epoch=epoch,
            )
            fabric.print(
                f"| {loss.detach().cpu().item()}",
                end=" ",
                flush=(step % 10) == 0,
            )
            losses["loss"].append(loss.detach().cpu().item())
            losses["epoch"].append(epoch)
            losses["step"].append(step)
            losses["encoder_grad_norm"].append(encoder_grad_norm)
            losses["recon_decoder_grad_norm"].append(recon_decoder_grad_norm)
            losses["decoder_grad_norm"].append(decoder_grad_norm)

            step += 1

        optim_encoder.zero_grad(set_to_none=True)
        optim_decoder.zero_grad(set_to_none=True)
        optim_recon_decoder.zero_grad(set_to_none=True)
        # Delete some training inputs to relax memory constraints in whole-
        # volume inference inside validation step.
        del x, x_coords, y, y_coords, pred_fodf, recon_pred

        fabric.print("\n==Validation==", flush=True)
        aim_run, val_viz_subj_id = validate_stage(
            fabric,
            encoder,
            decoder,
            val_dataloader=val_dataloader,
            step=step,
            epoch=epoch,
            aim_run=aim_run,
            val_viz_subj_id=val_viz_subj_id,
        )

except KeyboardInterrupt as e:
    aim_run.add_tag("STOPPED")
    (tmp_res_dir / "STOPPED").touch()
    raise e
except Exception as e:
    aim_run.add_tag("FAILED")
    (tmp_res_dir / "FAILED").touch()
    raise e
finally:
    aim_run.close()

# Sync all pytorch-lightning processes.
fabric.barrier()
if fabric.is_global_zero:
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "recon_decoder": recon_decoder.state_dict(),
            "epoch": epoch,
            "step": step,
            "aim_run_hash": aim_run.hash,
            "optim_encoder": optim_encoder.state_dict(),
            "optim_decoder": optim_decoder.state_dict(),
            "optim_recon_decoder": optim_recon_decoder.state_dict(),
        },
        Path(tmp_res_dir) / f"state_dict_epoch_{epoch}_step_{step}.pt",
    )
    fabric.print("=" * 40)
    losses = pd.DataFrame.from_dict(losses)
    losses.to_csv(Path(tmp_res_dir) / "train_losses.csv")

# %%
