# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.10.5 ('base')
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
import torchio
from box import Box
from natsort import natsorted
from pytorch_lightning.lite import LightningLite

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
    # Activate cudnn benchmarking to optimize convolution algorithm speed.
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        print("CuDNN convolution optimization enabled.")
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
p.experiment_name = "sr_debug"
p.override_experiment_name = False
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
p.train_val_test_split_file = random.choice(
    list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
)
p.aim_uri = "aim://dali.cpe.virginia.edu:53800"
###############################################
p.train = dict(
    in_patch_size=(32, 32, 32),
    batch_size=1,
    samples_per_subj_per_epoch=10,
    max_epochs=30,
    loss="mse",
)

# Network/model parameters.
p.encoder = dict(
    interior_channels=75,
    # (number of SH orders (l) + 1) * X that is as close to 100 as possible.
    out_channels=16 * 6,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="elu",
)
p.decoder = dict(
    context_v_features=p.encoder.out_channels,
    out_features=45,
    m_encode_num_freqs=12,
    sigma_encode_scale=4.0,
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
p.train.subj_ids = tvt_split[tvt_split.split == "train"].subj_id.tolist()
p.val = dict()
p.val.subj_ids = tvt_split[tvt_split.split == "val"].subj_id.tolist()
p.test = dict()
p.test.subj_ids = tvt_split[tvt_split.split == "test"].subj_id.tolist()

# Ensure that no test subj ids are in either the training or validation sets.
# However, we can have overlap between training and validation.
assert len(set(p.train.subj_ids) & set(p.test.subj_ids)) == 0
assert len(set(p.val.subj_ids) & set(p.test.subj_ids)) == 0

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
with warnings.catch_warnings(record=True) as warn_list:
    # pre_sample_ds = pitn.data.datasets.HCPfODFINRDataset(
    #     subj_ids=p.train.subj_ids,
    #     dwi_root_dir=hcp_full_res_data_dir,
    #     fodf_root_dir=hcp_full_res_fodf_dir,
    #     lr_dwi_root_dir=hcp_low_res_data_dir,
    #     lr_fodf_root_dir=hcp_low_res_fodf_dir,
    #     transform=None,
    # )

    # #!DEBUG
    pre_sample_ds = pitn.data.datasets.HCPfODFINRDataset(
        subj_ids=p.train.subj_ids[:4],
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
    )

train_dataset = pitn.data.datasets.HCPINRfODFPatchDataset(
    pre_sample_train_dataset,
    patch_func=pitn.data.datasets.HCPINRfODFPatchDataset.default_patch_func(
        spatial_size=p.train.in_patch_size,
        num_samples=p.train.samples_per_subj_per_epoch,
    ),
    samples_per_image=p.train.samples_per_subj_per_epoch,
    transform=pitn.data.datasets.HCPINRfODFPatchDataset.default_feature_tf(
        p.train.in_patch_size
    ),
)
# train_dataset = monai.data.PatchDataset(
#     cache_dataset,
#     patch_func=tf_patch_sampler,
#     samples_per_image=p.train.samples_per_subj_per_epoch,
#     transform=per_patch_transforms,
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
# ### Validation & Test Datasets

# %%

# %% [markdown]
# ## Model

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

        self.in_channels = in_channels
        self.interior_channels = interior_channels
        self.out_channels = out_channels

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Conv3d(
            self.in_channels,
            self.interior_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
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
        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.post_conv = torch.nn.Conv3d(
            self.interior_channels,
            self.out_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)

        return y


# %%
# INR/Decoder model
class ContRepDecoder(torch.nn.Module):

    TARGET_COORD_EPSILON = 1e-7

    def __init__(
        self,
        context_v_features: int,
        out_features: int,
        m_encode_num_freqs: int,
        sigma_encode_scale: float,
    ):
        super().__init__()

        # Determine the number of input features needed for the MLP.
        # The order for concatenation is
        # 1) ctx feats over the low-res input space
        # 2) target voxel shape
        # 3) absolute coords of this forward pass' prediction target
        # 4) absolute coords of the high-res target voxel
        # 5) relative coords between high-res target coords and this forward pass'
        #    prediction target, normalized by low-res voxel shape
        # 6) encoding of relative coords
        self.context_v_features = context_v_features
        self.ndim = 3
        self.m_encode_num_freqs = m_encode_num_freqs
        self.sigma_encode_scale = torch.as_tensor(sigma_encode_scale)
        self.n_encode_features = self.ndim * 2 * self.m_encode_num_freqs
        self.n_coord_features = 4 * self.ndim + self.n_encode_features
        self.internal_features = self.context_v_features + self.n_coord_features

        self.out_features = out_features

        # "Swish" function, recommended in MeshFreeFlowNet
        activate_cls = torch.nn.SiLU
        self.activate_fn = activate_cls(inplace=True)
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

    def forward(
        self,
        context_v,
        context_spatial_extent,
        query_vox_size,
        query_coord,
    ) -> torch.Tensor:
        context_vox_size = torch.abs(
            context_spatial_extent[..., 1, 1, 1] - context_spatial_extent[..., 0, 0, 0]
        )
        context_vox_size = context_vox_size[:, :, None, None, None]

        # Construct a grid of nearest indices in context space by sampling a grid of
        # *indices* given the coordinates in mm.
        # The channel dim is just repeated for every
        # channel, so that doesn't need to be in the idx grid.
        idx_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(0, context_spatial_extent.shape[i])
                    for i in (0, 2, 3, 4)
                ],
                indexing="ij",
            ),
            dim=1,
        ).to(context_spatial_extent)
        # Find the nearest grid point, where the batch+spatial dims are the "channels."
        nearest_coord_idx = pitn.nn.inr.weighted_ctx_v(
            idx_grid,
            # context_spatial_extent,
            input_space_extent=context_spatial_extent,
            target_space_extent=query_coord,
            reindex_spatial_extents=True,
            sample_mode="nearest",
        ).to(torch.long)
        # Expand along channel dimension for raw indexing.
        # nearest_coord_idx = einops.repeat(
        #     nearest_coord_idx,
        #     "b dim x y z -> dim b repeat_c x y z",
        #     repeat_c=self.context_v_features,
        # )
        nearest_coord_idx = einops.rearrange(
            nearest_coord_idx, "b dim x y z -> dim (b x y z)"
        )
        # nearest_coord_idx = tuple(torch.swapdims(nearest_coord_idx, 0, 1)).view(4, batch_size, -1)
        batch_idx = nearest_coord_idx[0]
        rel_norm_sub_window_grid_coord: torch.Tensor
        sub_window_query_sample_grid = list()
        # Build the low-res representation one sub-window voxel index at a time.
        for i in (0, 1):
            # Rebuild indexing tuple for each element of the sub-window
            x_idx = nearest_coord_idx[1] + i
            for j in (0, 1):
                y_idx = nearest_coord_idx[2] + j
                for k in (0, 1):
                    z_idx = nearest_coord_idx[3] + k
                    context_val = context_v[batch_idx, :, x_idx, y_idx, z_idx]
                    context_val = einops.rearrange(
                        context_val,
                        "(b x y z) c -> b c x y z",
                        x=query_coord.shape[2],
                        y=query_coord.shape[3],
                        z=query_coord.shape[4],
                    )
                    context_coord = context_spatial_extent[
                        batch_idx, :, x_idx, y_idx, z_idx
                    ]
                    context_coord = einops.rearrange(
                        context_coord,
                        "(b x y z) c -> b c x y z",
                        x=query_coord.shape[2],
                        y=query_coord.shape[3],
                        z=query_coord.shape[4],
                    )
                    # Take relative coordinate difference between the current context
                    # coord and the query coord.
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
                    q_vox_size = query_vox_size[..., None, None, None].expand_as(
                        rel_norm_context_coord
                    )

                    # Perform forward pass of the MLP.
                    context_feats = einops.rearrange(
                        context_val, "b c x y z -> (b x y z) c"
                    )
                    coord_feats = (
                        q_vox_size,
                        context_coord,
                        query_coord,
                        rel_norm_context_coord,
                        encoded_rel_norm_context_coord,
                    )
                    coord_feats = torch.cat(coord_feats, dim=1)
                    spatial_layout = {
                        "b": coord_feats.shape[0],
                        "x": coord_feats.shape[2],
                        "y": coord_feats.shape[3],
                        "z": coord_feats.shape[4],
                    }

                    coord_feats = einops.rearrange(
                        coord_feats, "b c x y z -> (b x y z) c"
                    )
                    x_coord = coord_feats
                    sub_grid_pred_ijk = context_feats
                    for l in self.internal_res_repr:
                        sub_grid_pred_ijk, x_coord = l(sub_grid_pred_ijk, x_coord)
                    sub_grid_pred_ijk = self.lin_post(sub_grid_pred_ijk)
                    sub_grid_pred_ijk = einops.rearrange(
                        sub_grid_pred_ijk, "(b x y z) c -> b c x y z", **spatial_layout
                    )
                    sub_window_query_sample_grid.append(sub_grid_pred_ijk)

                    if i == j == k == 0:
                        # Find the relative coordinate of the query within the
                        # sub-window.
                        rel_norm_sub_window_grid_coord = torch.clamp(
                            (rel_norm_context_coord - 0.5) * 2,
                            # ((context_coord + context_vox_size / 2) - query_coord)
                            # / context_vox_size,
                            -1 + self.TARGET_COORD_EPSILON,
                            1 - self.TARGET_COORD_EPSILON,
                        )
        sub_window_query_sample_grid = torch.stack(sub_window_query_sample_grid, dim=0)
        spatial_layout = {
            "b": sub_window_query_sample_grid.shape[1],
            "x": sub_window_query_sample_grid.shape[3],
            "y": sub_window_query_sample_grid.shape[4],
            "z": sub_window_query_sample_grid.shape[5],
        }
        sub_window = einops.rearrange(
            sub_window_query_sample_grid,
            "(x_sub y_sub z_sub) b c x y z -> (b x y z) c x_sub y_sub z_sub",
            x_sub=2,
            y_sub=2,
            z_sub=2,
        )
        sub_window_grid = einops.rearrange(
            rel_norm_sub_window_grid_coord, "b dim x y z -> (b x y z) 1 1 1 dim "
        )

        y = F.grid_sample(
            sub_window,
            sub_window_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="reflection",
        )
        y = einops.rearrange(y, "(b x y z) c 1 1 1 -> b c x y z", **spatial_layout)

        return y


# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
tmp_res_dir = Path(".") / "tmp_res" / ts
tmp_res_dir.mkdir(parents=True)


# %%
class INRSystem(LightningLite):
    def run(
        self,
        epochs: int,
        batch_size: int,
        in_channels: int,
        pred_channels: int,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        train_dataset,
        optim_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(),
        stage="train",
        logger=None,
    ):
        encoder = INREncoder(**{**encoder_kwargs, **{"in_channels": in_channels}})
        decoder = ContRepDecoder(**decoder_kwargs)

        optim = torch.optim.AdamW(
            itertools.chain(encoder.parameters(), decoder.parameters()), **optim_kwargs
        )
        encoder = self.setup(encoder)
        decoder, optim = self.setup(decoder, optim)

        loss_fn = torch.nn.MSELoss(reduction="mean")

        train_dataloader = monai.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            **dataloader_kwargs,
        )
        train_dataloader = self.setup_dataloaders(train_dataloader)

        encoder.train()
        decoder.train()
        out_dir = tmp_res_dir

        losses = dict(
            loss=list(),
            epoch=list(),
            step=list(),
            encoder_grad_norm=list(),
            decoder_grad_norm=list(),
        )
        step = 0
        train_lr = False
        for epoch in range(epochs):
            print(f"Epoch {epoch}\n", "=" * 10)
            # if epoch < (epochs // 10):
            if False:
                train_dataloader.dataset.set_select_tf_keys(
                    add_keys=["lr_fodf"],
                    remove_keys=["fodf", "mask", "fr_patch_extent_acpc"],
                )
                train_lr = True
            # elif epoch == (epochs // 10):
            elif False:
                train_dataloader.dataset.set_select_tf_keys(
                    add_keys=["fodf", "mask", "fr_patch_extent_acpc"],
                    remove_keys=["lr_fodf"],
                )
                train_lr = False

            for batch_dict in train_dataloader:
                x = batch_dict["lr_dwi"]
                # b_size = x.shape[0]
                # spatial_size = torch.prod(torch.as_tensor(x.shape[2:]))
                if not train_lr:
                    y = batch_dict["fodf"]
                else:
                    y = batch_dict["lr_fodf"]
                y_mask = batch_dict["mask"].to(torch.bool)
                x_coords = batch_dict["lr_patch_extent_acpc"]
                x_vox_size = torch.atleast_2d(batch_dict["lr_vox_size"])
                y_coords = batch_dict["fr_patch_extent_acpc"]
                y_vox_size = torch.atleast_2d(batch_dict["vox_size"])
                # print(y.shape)
                # print(y_mask.shape)
                # print(x.shape)
                # print(x_coords.shape)
                # print(y_coords.shape)

                optim.zero_grad()
                ctx_v = encoder(x)

                pred_fodf = decoder(
                    context_v=ctx_v,
                    context_spatial_extent=x_coords,
                    query_vox_size=y_vox_size,
                    query_coord=y_coords,
                )
                # pred_fodf_patch = einops.rearrange(
                #     pred_fodf,
                #     "(b x y z) c -> b c x y z",
                #     b=y.shape[0],
                #     c=y.shape[1],
                #     x=y.shape[2],
                #     y=y.shape[3],
                #     z=y.shape[4],
                # )
                y_mask_broad = torch.broadcast_to(y_mask, y.shape)
                loss = loss_fn(pred_fodf[y_mask_broad], y[y_mask_broad])
                self.backward(loss)
                optim.step()

                print(f"| {loss.detach().cpu().item()}", end=" ")
                losses["loss"].append(loss.detach().cpu().item())
                losses["epoch"].append(epoch)
                losses["step"].append(step)
                losses["encoder_grad_norm"].append(self._calc_grad_norm(encoder))
                losses["decoder_grad_norm"].append(self._calc_grad_norm(decoder))

                if False:
                    print("Overfitting to batch")
                    # plt.imshow(x[0, 7, :, 0].detach().cpu().numpy(), cmap="gray")
                    # plt.colorbar()
                    # plt.show()

                    # plt.imshow(y[0, 0, :, 0].detach().cpu().numpy(), cmap="gray")
                    # plt.colorbar()
                    # plt.show()
                    # plt.imshow(y_mask[0, 0, :, 0].detach().cpu().numpy(), cmap="gray")
                    # plt.colorbar()
                    # plt.show()

                    # fig = plt.figure(dpi=170, figsize=(5, 8))
                    # pitn.viz.plot_vol_slices(
                    #     x_coords[0].detach(),
                    #     y_coords[0].detach(),
                    #     slice_idx=(0.4, 0.5, 0.5),
                    #     title=f"Epoch {epoch} Step {step}",
                    #     vol_labels=["Source Coord", "Target Coord"],
                    #     channel_labels=["X", "Y", "Z"],
                    #     colorbars="each",
                    #     fig=fig,
                    #     cmap="gray",
                    # )
                    # plt.show()

                    encoder, decoder, optim = self._overfit_batch(
                        repeats=10,
                        encoder=encoder,
                        decoder=decoder,
                        optim=optim,
                        loss_fn=loss_fn,
                        x=x,
                        y=y,
                        x_coords=x_coords,
                        y_coords=y_coords,
                        y_mask=y_mask,
                        y_vox_size=y_vox_size,
                    )
                    fig = plt.figure(dpi=170, figsize=(5, 8))
                    pitn.viz.plot_vol_slices(
                        x[0, 7].detach(),
                        pred_fodf_patch[0, 0].detach(),
                        y[0, 0].detach(),
                        y_mask[0, 0].detach(),
                        slice_idx=(0.4, 0.5, 0.5),
                        title=f"epoch {epoch} step {step}",
                        vol_labels=["input", "pred", "target", "target mask"],
                        colorbars="each",
                        fig=fig,
                        cmap="gray",
                    )
                    plt.savefig(Path(out_dir) / f"overfit_epoch_{epoch}.png")
                    # #!DEBUG
                    # return
                    #!
                step += 1
            # Save some example predictions after each epoch
            fig = plt.figure(dpi=150, figsize=(4, 6))
            pitn.viz.plot_vol_slices(
                x[0, 7].detach(),
                pred_fodf[0, 0].detach(),
                y[0, 0].detach(),
                slice_idx=(0.4, 0.5, 0.5),
                title=f"Epoch {epoch} Step {step}",
                vol_labels=["Input", "Pred", "Target"],
                colorbars="each",
                fig=fig,
                cmap="gray",
            )
            plt.savefig(Path(out_dir) / f"epoch_{epoch}_sample.png")

        print("=" * 10)
        losses = pd.DataFrame.from_dict(losses)
        losses.to_csv(Path(out_dir) / "train_losses.csv")
        losses.plot()

    def _overfit_batch(
        self,
        repeats: int,
        encoder,
        decoder,
        optim,
        loss_fn,
        x,
        y,
        x_coords,
        y_coords,
        y_mask,
        y_vox_size,
    ):
        optim.zero_grad()
        vectorized_y_coords = einops.rearrange(y_coords, "b c x y z -> (b x y z) c")
        vectorized_y_vox_size = einops.rearrange(
            y_vox_size.expand(*y.shape[2:], -1, -1),
            "x y z b c -> (b x y z) c",
        )
        y_mask_broad = torch.broadcast_to(y_mask, y.shape)
        for i in range(repeats):
            optim.zero_grad()
            ctx_v = encoder(x)
            ctx_v = pitn.nn.inr.linear_weighted_ctx_v(
                ctx_v,
                input_space_extent=x_coords,
                target_space_extent=y_coords,
                reindex_spatial_extents=True,
            )
            ctx_v = einops.rearrange(ctx_v, "b c x y z -> (b x y z) c")

            pred_fodf = decoder(
                query_coord=vectorized_y_coords,
                context_v=ctx_v,
                vox_size=vectorized_y_vox_size,
            )
            pred_fodf_patch = einops.rearrange(
                pred_fodf,
                "(b x y z) c -> b c x y z",
                b=y.shape[0],
                c=y.shape[1],
                x=y.shape[2],
                y=y.shape[3],
                z=y.shape[4],
            )

            loss = loss_fn(pred_fodf_patch[y_mask_broad], y[y_mask_broad])
            self.backward(loss)
            optim.step()
            if i % (repeats // 10) == 0:
                print(
                    f"Overfit step {i} out of {repeats} loss {loss.detach().cpu().item()}",
                    end=" ",
                )
        if repeats > 0:
            optim.zero_grad()
        return encoder, decoder, optim

    @staticmethod
    def _calc_grad_norm(model, norm_type=2):
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

    # def validate(self, model, val_dataset):
    #     pass

    # def test(self, model, test_dataset):
    #     pass


# %% [markdown]
# ## Training

# %%
# Instantiate the system, may be re-used between training and testing.
model_system = INRSystem(accelerator="gpu", devices=1, precision=32)

if "in_channels" not in p.encoder:
    in_channels = int(train_dataset[0]["lr_dwi"].shape[0])
else:
    in_channels = p.encoder.in_channels

model_system.run(
    p.train.max_epochs,
    p.train.batch_size,
    in_channels=in_channels,
    pred_channels=p.decoder.out_features,
    encoder_kwargs=p.encoder.to_dict(),
    decoder_kwargs=p.decoder.to_dict(),
    train_dataset=train_dataset,
)

# %%
losses = pd.read_csv(tmp_res_dir / "train_losses.csv")

plt.figure(dpi=100)
plt.plot(losses.step[50:], losses.loss[50:], label="loss")
plt.legend()
plt.show()

plt.figure(dpi=100)
plt.plot(losses.step[50:], losses.encoder_grad_norm[50:], label="encoder grad norm")
plt.legend()
plt.show()

plt.figure(dpi=100)
plt.plot(losses.step[50:], losses.decoder_grad_norm[50:], label="decoder grad norm")
plt.legend()
plt.show()

# %%
plt.figure(dpi=100)
plt.plot(losses.step[500:], losses.loss[500:], label="loss")
plt.legend()
plt.show()

plt.figure(dpi=100)
plt.plot(losses.step[500:], losses.encoder_grad_norm[500:], label="encoder grad norm")
plt.legend()
plt.show()

plt.figure(dpi=100)
plt.plot(losses.step[500:], losses.decoder_grad_norm[500:], label="decoder grad norm")
plt.legend()
plt.show()
# %%


# %% [markdown]
# ## Testing & Visualization

# %%

# %%
