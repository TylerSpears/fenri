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

# %% [markdown]
# ## Experiment & Parameters Setup

# %%
p = Box(default_box=True)
# Experiment defaults, can be overridden in a config file.

p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
p.test.subj_ids = ["299154"]
p.model_weight_f = str(
    Path(p.tmp_results_dir) / "2023-02-21T16_46_25/state_dict_epoch_199_step_40000.pt"
)
# p.model_weight_f = str(
#     Path(p.tmp_results_dir) / "2023-02-09T21_09_47/state_dict_epoch_174_step_35000.pt"
# )
p.target_vox_size = 0.374
###############################################
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

# %% [markdown]
# ### Validation & Test Datasets

# %%
with warnings.catch_warnings(record=True) as warn_list:

    # Validation dataset.
    test_paths_dataset = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.test.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        lr_dwi_root_dir=hcp_low_res_data_dir,
        lr_fodf_root_dir=hcp_low_res_fodf_dir,
    )
    cached_test_dataset = monai.data.CacheDataset(
        test_paths_dataset,
        transform=test_paths_dataset.default_pre_sample_tf(0, skip_sample_mask=True),
        copy_cache=False,
        num_workers=2,
    )
    test_dataset = pitn.data.datasets.HCPfODFINRWholeVolDataset(
        cached_test_dataset,
        transform=pitn.data.datasets.HCPfODFINRWholeVolDataset.default_tf(),
    )

    # # Test dataset.
    # # The test dataset won't be cached, as each image should only be loaded once.
    # test_paths_dataset = pitn.data.datasets.HCPfODFINRDataset(
    #     subj_ids=p.test.subj_ids,
    #     dwi_root_dir=hcp_full_res_data_dir,
    #     fodf_root_dir=hcp_full_res_fodf_dir,
    #     lr_dwi_root_dir=hcp_low_res_data_dir,
    #     lr_fodf_root_dir=hcp_low_res_fodf_dir,
    #     transform=pitn.data.datasets.HCPfODFINRDataset.default_pre_sample_tf(
    #         0, skip_sample_mask=True
    #     ),
    # )
    # test_dataset = pitn.data.datasets.HCPfODFINRWholeVolDataset(
    #     test_paths_dataset,
    #     transform=pitn.data.datasets.HCPfODFINRWholeVolDataset.default_tf(),
    # )

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
# ## Evaluation

# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
tmp_res_dir = Path(p.tmp_results_dir) / "_".join([ts, "super_res_odf_test"])
tmp_res_dir.mkdir(parents=True)

# %%
test_dataloader = monai.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)


# %% [markdown]
# ### INR

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
        rel_context_coord = query_coord - context_coord
        # Also normalize to [0, 1) by subtracting the lower bound of differences
        # (- voxel size) and dividing by 2xupper bound (2 x voxel size).
        rel_norm_context_coord = (rel_context_coord - -context_vox_size) / (
            2 * context_vox_size
        )
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


# %%
# Test all given subjects.
system_state_dict = torch.load(p.model_weight_f)
encoder_state_dict = system_state_dict["encoder"]

decoder_state_dict = system_state_dict["decoder"]

if "in_channels" not in p.encoder:
    in_channels = int(test_dataset[0]["lr_dwi"].shape[0])
else:
    in_channels = p.encoder.in_channels

encoder = INREncoder(**{**p.encoder.to_dict(), **{"in_channels": in_channels}})
encoder.load_state_dict(encoder_state_dict)
encoder.to(device)

decoder = ReducedDecoder(**p.decoder.to_dict())
decoder.load_state_dict(decoder_state_dict)
decoder.to(device)
del (
    system_state_dict,
    encoder_state_dict,
    decoder_state_dict,
)

encoder.eval()
decoder.eval()

for batch_dict in test_dataloader:

    subj_id = batch_dict["subj_id"]
    if len(subj_id) == 1:
        subj_id = subj_id[0]
    x = batch_dict["lr_dwi"].to(device)
    x_coords = batch_dict["lr_extent_acpc"].to(device)
    x_affine_vox2mm = batch_dict["affine_lrvox2acpc"].to(device)
    x_vox_size = torch.atleast_2d(batch_dict["lr_vox_size"]).to(device)
    x_mask = batch_dict["lr_mask"].to(torch.bool).to(device)

    lower_lim = torch.stack(
        [
            x_coords[0, 0][0].unique()[0],
            x_coords[0, 1][:, 0].unique()[0],
            x_coords[0, 2][:, :, 0].unique()[0],
        ]
    )
    upper_lim = torch.stack(
        [
            x_coords[0, 0][-1].unique()[0],
            x_coords[0, 1][:, -1].unique()[0],
            x_coords[0, 2][:, :, -1].unique()[0],
        ]
    )
    super_z = torch.arange(lower_lim[0], upper_lim[0], step=p.target_vox_size).to(
        x_coords
    )
    super_y = torch.arange(lower_lim[1], upper_lim[1], step=p.target_vox_size).to(
        x_coords
    )
    super_x = torch.arange(lower_lim[2], upper_lim[2], step=p.target_vox_size).to(
        x_coords
    )

    super_zzz, super_yyy, super_xxx = torch.meshgrid(
        [super_z, super_y, super_x], indexing="ij"
    )
    super_coords = torch.stack([super_zzz, super_yyy, super_xxx], dim=0)[None]
    super_vol_shape = tuple(super_coords.shape[2:])

    super_vox_size = torch.ones_like(x_vox_size) * p.target_vox_size

    vox2acpc = batch_dict["affine_lrvox2acpc"][0].cpu()
    scale = (p.target_vox_size / x_vox_size)[0].cpu()
    scale = torch.cat([scale, scale.new_ones(1)]).cpu()
    scale = torch.diag_embed(scale).to(vox2acpc).cpu()
    new_aff = vox2acpc @ scale
    new_aff = new_aff.numpy()

    with torch.no_grad():
        ic("Starting net inference.")
        ctx_v = encoder(x)

        # Whole-volume inference is memory-prohibitive, so use a sliding
        # window inference method on the encoded volume.
        pred_super_fodf = monai.inferers.sliding_window_inference(
            super_coords.cpu(),
            # roi_size=(48, 48, 48),
            roi_size=(64, 64, 64),
            sw_batch_size=super_coords.shape[0],
            predictor=lambda q: decoder(
                query_coord=q.to(device),
                context_v=ctx_v,
                context_spatial_extent=x_coords,
                affine_context_vox2mm=x_affine_vox2mm,
            ).cpu(),
            overlap=0,
            padding_mode="replicate",
        )
    ic("Finished network inference.")
    mask_coords = einops.rearrange(super_coords, "b coord z y x -> b (z y x) coord")
    super_mask = pitn.affine.sample_3d(
        x_mask.cpu(), mask_coords.cpu(), vox2acpc, mode="nearest", align_corners=True
    )
    super_mask = (
        einops.rearrange(
            super_mask,
            "b (z y x) c -> b z y x c",
            z=super_vol_shape[0],
            y=super_vol_shape[1],
            x=super_vol_shape[2],
        )
        .squeeze()
        .cpu()
        .to(torch.int8)
        .numpy()
    )
    superres_pred = pred_super_fodf.cpu().numpy()
    superres_pred = superres_pred * super_mask
    odf_coeffs = np.moveaxis(superres_pred, 1, -1).squeeze()
    ic("Saving super-res fodf coeffs.")
    nib.save(
        nib.Nifti1Image(odf_coeffs, affine=new_aff),
        tmp_res_dir / f"{subj_id}_odf-coeff_inr-super-res_{p.target_vox_size}mm.nii.gz",
    )
    ic("Saving mask.")
    nib.save(
        nib.Nifti1Image(super_mask, affine=new_aff),
        tmp_res_dir / f"{subj_id}_mask-super-res_{p.target_vox_size}mm.nii.gz",
    )

# %% [markdown]
# ### Tri-Linear Interp

# %%
for batch_dict in test_dataloader:

    subj_id = batch_dict["subj_id"]
    if len(subj_id) == 1:
        subj_id = subj_id[0]
    x = batch_dict["lr_fodf"].to(device)
    x_coords = batch_dict["lr_extent_acpc"].to(device)
    x_vox_size = torch.atleast_2d(batch_dict["lr_vox_size"]).to(device)
    x_mask = batch_dict["lr_mask"].to(torch.bool).to(device)

    lower_lim = torch.stack(
        [
            x_coords[0, 0][0].unique()[0],
            x_coords[0, 1][:, 0].unique()[0],
            x_coords[0, 2][:, :, 0].unique()[0],
        ]
    )
    upper_lim = torch.stack(
        [
            x_coords[0, 0][-1].unique()[0],
            x_coords[0, 1][:, -1].unique()[0],
            x_coords[0, 2][:, :, -1].unique()[0],
        ]
    )
    super_z = torch.arange(lower_lim[0], upper_lim[0], step=p.target_vox_size).to(
        x_coords
    )
    super_y = torch.arange(lower_lim[1], upper_lim[1], step=p.target_vox_size).to(
        x_coords
    )
    super_x = torch.arange(lower_lim[2], upper_lim[2], step=p.target_vox_size).to(
        x_coords
    )

    super_zzz, super_yyy, super_xxx = torch.meshgrid(
        [super_z, super_y, super_x], indexing="ij"
    )
    super_coords = torch.stack([super_zzz, super_yyy, super_xxx], dim=-1)[None]
    super_vol_shape = tuple(super_coords.shape[:-1])
    super_coords = einops.rearrange(super_coords, "b z y x coord -> b (z y x) coord")

    super_vox_size = torch.ones_like(x_vox_size) * p.target_vox_size

    vox2acpc = batch_dict["affine_lrvox2acpc"][0].cpu()
    scale = (p.target_vox_size / x_vox_size)[0].cpu()
    scale = torch.cat([scale, scale.new_ones(1)]).cpu()
    scale = torch.diag_embed(scale).to(vox2acpc).cpu()
    new_aff = vox2acpc @ scale
    new_aff = new_aff.numpy()
    print("Resample fodf coeffs")
    pred_super_fodf = pitn.affine.sample_3d(
        x.cpu(), super_coords.cpu(), vox2acpc, mode="bilinear", align_corners=True
    )
    super_mask = pitn.affine.sample_3d(
        x_mask.cpu(), super_coords.cpu(), vox2acpc, mode="nearest", align_corners=True
    )
    pred_super_fodf = pred_super_fodf * super_mask.bool()
    superres_pred = pred_super_fodf.detach().cpu()
    superres_pred = einops.rearrange(
        superres_pred,
        "b (z y x) c -> b z y x c",
        z=super_vol_shape[1],
        y=super_vol_shape[2],
        x=super_vol_shape[3],
    )
    superres_pred = superres_pred.numpy().astype(np.float32).squeeze()

    super_mask = (
        einops.rearrange(
            super_mask,
            "b (z y x) c -> b z y x c",
            z=super_vol_shape[1],
            y=super_vol_shape[2],
            x=super_vol_shape[3],
        )
        .squeeze()
        .cpu()
        .to(torch.int8)
        .numpy()
    )
    nib.save(
        nib.Nifti1Image(superres_pred, affine=new_aff),
        tmp_res_dir
        / f"{subj_id}_odf-coeff_tri-linear-super-res_{p.target_vox_size}mm.nii.gz",
    )
    # nib.save(
    #     nib.Nifti1Image(super_mask, affine=new_aff),
    #     tmp_res_dir / f"{subj_id}_mask-super-res_{p.target_vox_size}mm.nii.gz",
    # )

# %%
