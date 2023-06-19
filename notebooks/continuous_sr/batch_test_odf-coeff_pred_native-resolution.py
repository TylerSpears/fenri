# -*- coding: utf-8 -*-

# %% [markdown]
# # Batch Prediction in Subject Native Resolution for fODF INR
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
from functools import partial
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
from box import Box
from icecream import ic
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
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
# p.train_val_test_split_file = random.choice(
#     list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
# )
p.model_weight_f = str(
    Path(p.tmp_results_dir)
    / "2023-06-18T20_34_58"
    / "state_dict_epoch_99_step_25000.pt"
)
###############################################
# kwargs for the sub-selection function to go from full DWI -> low-res DWI.
# See `sub_select_dwi_from_bval` function in `pitn`.
p.bval_sub_sample_fn_kwargs = dict(
    shells_to_remove=[2000],
    within_shell_idx_to_keep={
        0: range(0, 9),
        1000: range(0, 45),
        3000: range(0, 45),
    },
)
# 1.25mm -> 2.0mm
p.baseline_lr_spacing_scale = 1.6
p.scale_prefilter_kwargs = dict(
    sigma_scale_coeff=2.5,
    sigma_truncate=4.0,
)
p.test.subj_ids = list(
    map(
        str,
        [
            581450,
        ],
    )
)

# Network/model parameters.
p.encoder = dict(
    interior_channels=80,
    out_channels=128,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="relu",
    input_coord_channels=True,
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
# tvt_split = pd.read_csv(p.train_val_test_split_file)
# p.test.subj_ids = natsorted(tvt_split[tvt_split.split == "test"].subj_id.tolist())

# %%
ic(p.to_dict())

# %% [markdown]
# ## Data Loading

# %%
hcp_full_res_data_dir = Path("/data/srv/data/pitn/hcp")
hcp_full_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()

# %%
# Define hte bval/bvec sub-sample scheme according to the parameter dict kwargs.
bval_sub_sample_fn = partial(
    pitn.data.datasets2.sub_select_dwi_from_bval,
    **p.bval_sub_sample_fn_kwargs.to_dict(),
)

# %% [markdown]
# ### Test Dataset

# %%
with warnings.catch_warnings(record=True) as warn_list:

    test_dataset = pitn.data.datasets2.HCPfODFINRDataset(
        subj_ids=p.test.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        transform=pitn.data.datasets2.HCPfODFINRDataset.default_pre_sample_tf(
            sample_mask_key="wm_mask",
            bval_sub_sample_fn=bval_sub_sample_fn,
        ),
    )

    test_dataset = pitn.data.datasets2.HCPfODFINRWholeBrainDataset(
        test_dataset,
        transform=pitn.data.datasets2.HCPfODFINRWholeBrainDataset.default_vol_tf(
            baseline_iso_scale_factor_lr_spacing_mm_low_high=p.baseline_lr_spacing_scale,
            scale_prefilter_kwargs=p.scale_prefilter_kwargs,
        ),
    )
    test_dataset = monai.data.CacheDataset(
        test_dataset,
        cache_num=3,
        transform=None,
        progress=True,
        copy_cache=False,
        num_workers=4,
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
        input_coord_channels: bool,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_channels=in_channels,
            input_coord_channels=input_coord_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
        )

        self.in_channels = in_channels
        # This is just a convenience flag, for now.
        self.input_coord_channels = input_coord_channels
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

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)

        return y


class Decoder(torch.nn.Module):
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
        c = einops.rearrange(coords, "b n coord -> (b n) coord")
        sigma = self.sigma_encode_scale.expand_as(c).to(c)[..., None]
        encode_pos = pitn.nn.inr.fourier_position_encoding(
            c, sigma_scale=sigma, m_num_freqs=self.m_encode_num_freqs
        )
        encode_pos = einops.rearrange(
            encode_pos, "(b n) freqs -> b n freqs", n=coords.shape[1]
        )
        return encode_pos

    def sub_grid_forward(
        self,
        context_v,
        context_world_coord,
        query_world_coord,
        query_world_coord_mask,
        query_sub_grid_coord,
        context_sub_grid_coord,
    ):
        # Take relative coordinate difference between the current context
        # coord and the query coord, given in normalized ([0, 1]) sub-grid coordinates.
        batch_size = context_v.shape[0]
        rel_q2ctx_norm_sub_grid_coord = (
            query_sub_grid_coord - context_sub_grid_coord + 1
        ) / 2

        # assert (rel_q2ctx_norm_sub_grid_coord >= 0).all() and (
        #     rel_q2ctx_norm_sub_grid_coord <= 1.0
        # ).all()
        encoded_rel_q2ctx_coord = self.encode_relative_coord(
            rel_q2ctx_norm_sub_grid_coord
        )
        # b n 1 -> b 1 n
        context_v = context_v * query_world_coord_mask[:, None, :, 0]
        # Perform forward pass of the MLP.
        if self.norm_pre is not None:
            context_v = self.norm_pre(context_v)
        # Group batches and queries-per-batch into just batches, keep context channels
        # as the feature vector.
        context_feats = einops.rearrange(context_v, "b channels n -> (b n) channels")
        feat_mask = einops.rearrange(query_world_coord_mask, "b n 1 -> (b n) 1")
        coord_feats = (
            context_world_coord,
            query_world_coord,
            encoded_rel_q2ctx_coord,
        )
        coord_feats = torch.cat(coord_feats, dim=-1)

        coord_feats = einops.rearrange(coord_feats, "b n coord -> (b n) coord")
        x_coord = coord_feats * feat_mask
        y_sub_grid_pred = context_feats * feat_mask

        if self.lin_pre is not None:
            y_sub_grid_pred = self.lin_pre(y_sub_grid_pred)
            y_sub_grid_pred = self.activate_fn(y_sub_grid_pred)

        for l in self.internal_res_repr:
            y_sub_grid_pred, x_coord = l(y_sub_grid_pred, x_coord)
        # The SkipMLPBlock contains the residual addition, so no need to add here.
        y_sub_grid_pred = self.lin_post(y_sub_grid_pred)
        y_sub_grid_pred = einops.rearrange(
            y_sub_grid_pred, "(b n) channels -> b channels n", b=batch_size
        )

        return y_sub_grid_pred

    def forward(
        self,
        context_v: torch.Tensor,
        context_world_coord_grid: torch.Tensor,
        query_world_coord: torch.Tensor,
        query_world_coord_mask: torch.Tensor,
        affine_context_vox2world: torch.Tensor,
        affine_query_vox2world: torch.Tensor,
        context_vox_size_world: torch.Tensor,
        query_vox_size_world: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_v.shape[0]
        query_orig_shape = tuple(query_world_coord.shape)
        # All coords and coord grids must be *coordinate-last* format (similar to
        # channel-last).
        q_world = einops.rearrange(
            query_world_coord, "b ... c -> b (...) c", b=batch_size, c=3
        )
        n_q_per_batch = q_world.shape[1]
        # Query mask should be b x y z 1
        q_mask = einops.rearrange(query_world_coord_mask, "b ... 1 -> b (...) 1")
        # Replace each unmasked (i.e., invalid) query point by a dummy point, in this
        # case a point from roughly the middle of each batch of query points, which
        # should be as safe as possible from being out of bounds wrt the context vox
        # indices.
        dummy_q_world = q_world[:, (n_q_per_batch // 2)]
        dummy_q_world.unsqueeze_(1)
        # Replace all unmasked q world coords with the dummy coord.
        q_world = torch.where(q_mask, q_world, dummy_q_world)

        affine_world2ctx_vox = torch.linalg.inv(affine_context_vox2world)
        q_ctx_vox = pitn.affine.transform_coords(q_world, affine_world2ctx_vox)
        q_ctx_vox_bottom = q_ctx_vox.floor().long()
        # The vox coordinates are not broadcast over every batch (like they are over
        # every channel), so we need a batch idx to associate each sub-grid voxel with
        # the appropriate batch index.
        batch_vox_idx = einops.repeat(
            torch.arange(
                batch_size,
                dtype=q_ctx_vox_bottom.dtype,
                device=q_ctx_vox_bottom.device,
            ),
            "batch -> batch n",
            n=n_q_per_batch,
        )
        q_sub_grid_coord = q_ctx_vox - q_ctx_vox_bottom.to(q_ctx_vox)
        # q_bottom_in_world_coord = pitn.affine.transform_coords(
        #     q_ctx_vox_bottom.to(affine_context_vox2world), affine_context_vox2world
        # )

        y_weighted_accumulate = None
        # Build the low-res representation one sub-grid voxel index at a time.
        # Each sub-grid is a [0, 1] voxel coordinate system local to the query point,
        # where the origin is the context voxel that is "lower" in all dimensions
        # than the query coordinate.
        # The indicators specify if the current voxel index that surrounds the
        # query coordinate should be "off  or not. If not, then
        # the center voxel (read: no voxel offset from the center) is selected
        # (for that dimension).
        sub_grid_offset_ijk = q_ctx_vox_bottom.new_zeros(1, 1, 3)
        for (
            corner_offset_i,
            corner_offset_j,
            corner_offset_k,
        ) in itertools.product((0, 1), (0, 1), (0, 1)):
            # Rebuild indexing tuple for each element of the sub-window
            sub_grid_offset_ijk[..., 0] = corner_offset_i
            sub_grid_offset_ijk[..., 1] = corner_offset_j
            sub_grid_offset_ijk[..., 2] = corner_offset_k
            sub_grid_index_ijk = q_ctx_vox_bottom + sub_grid_offset_ijk

            sub_grid_context_v = context_v[
                batch_vox_idx.flatten(),
                :,
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
            ]
            sub_grid_context_v = einops.rearrange(
                sub_grid_context_v,
                "(b n) channels -> b channels n",
                b=batch_size,
                n=n_q_per_batch,
            )
            sub_grid_context_world_coord = context_world_coord_grid[
                batch_vox_idx.flatten(),
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
                :,
            ]
            sub_grid_context_world_coord = einops.rearrange(
                sub_grid_context_world_coord,
                "(b n) coords -> b n coords",
                b=batch_size,
                n=n_q_per_batch,
            )

            sub_grid_pred_ijk = self.sub_grid_forward(
                context_v=sub_grid_context_v,
                context_world_coord=sub_grid_context_world_coord,
                query_world_coord=q_world,
                query_sub_grid_coord=q_sub_grid_coord,
                query_world_coord_mask=q_mask,
                context_sub_grid_coord=sub_grid_offset_ijk,
            )
            # Initialize the accumulated prediction after finding the
            # output size; easier than trying to pre-compute the shape.
            if y_weighted_accumulate is None:
                y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

            sub_grid_offset_ijk_compliment = torch.abs(1 - sub_grid_offset_ijk)
            sub_grid_context_vox_coord_compliment = (
                q_ctx_vox_bottom + sub_grid_offset_ijk_compliment
            )
            w_sub_grid_cube_ijk = torch.abs(
                sub_grid_context_vox_coord_compliment - q_ctx_vox
            )
            # Each coordinate difference is a side of the cube, so find the volume.
            w_ijk = einops.reduce(
                w_sub_grid_cube_ijk, "b n coord -> b 1 n", reduction="prod"
            )

            # Accumulate weighted cell predictions to eventually create
            # the final prediction.
            y_weighted_accumulate += w_ijk * sub_grid_pred_ijk

        y = y_weighted_accumulate

        out_channels = y.shape[1]
        # Reshape prediction to match the input query coordinates spatial shape.
        q_in_within_batch_samples_shape = query_orig_shape[1:-1]
        y = y.reshape(*((batch_size, out_channels) + q_in_within_batch_samples_shape))

        return y


# %% [markdown]
# ## Testing

# %%
def batchwise_masked_mse(y_pred, y, mask):
    masked_y_pred = y_pred.clone()
    masked_y = y.clone()
    masked_y_pred[~mask] = torch.nan
    masked_y[~mask] = torch.nan
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
            batch_size = x.shape[0]
            x_mask = batch_dict["lr_brain_mask"].to(torch.bool)
            x_affine_vox2world = batch_dict["affine_lr_vox2world"]
            x_vox_size = batch_dict["lr_vox_size"]
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"]
            y_mask = batch_dict["brain_mask"].to(torch.bool)
            y_affine_vox2world = batch_dict["affine_vox2world"]
            y_vox_size = batch_dict["vox_size"]
            y_coords = pitn.affine.affine_coordinate_grid(
                y_affine_vox2world, tuple(y.shape[2:])
            )
            # Fix an edge case in the affine_coordinate_grid function.
            if batch_size == 1:
                if x_coords.shape[0] != 1:
                    x_coords.unsqueeze_(0)
                if y_coords.shape[0] != 1:
                    y_coords.unsqueeze_(0)

            # Concatenate the input world coordinates as input features into the
            # encoder. Mask out the x coordinates that are not to be considered.
            x_coord_mask = einops.rearrange(x_mask, "b 1 x y z -> b x y z 1")
            x_coords_encoder = einops.rearrange(
                x_coords * x_coord_mask, "b x y z coord -> b coord x y z"
            )
            x = torch.cat([x, x_coords_encoder], dim=1)
            ctx_v = encoder(x)

            # Whole-volume inference is memory-prohibitive, so use a sliding
            # window inference method on the encoded volume.
            # Transform y_coords into a coordinates-first shape, for the interface, and
            # attach the mask for compatibility with the sliding inference function.
            y_slide_window = torch.cat(
                [
                    einops.rearrange(y_coords, "b x y z coord -> b coord x y z"),
                    y_mask.to(y_coords),
                ],
                dim=1,
            )
            fn_coordify = lambda x: einops.rearrange(
                x, "b coord x y z -> b x y z coord"
            )
            pred_fodf = monai.inferers.sliding_window_inference(
                y_slide_window,
                roi_size=(48, 48, 48),
                sw_batch_size=y_coords.shape[0],
                predictor=lambda q: decoder(
                    # Rearrange back into coord-last format.
                    query_world_coord=fn_coordify(q[:, :-1]),
                    query_world_coord_mask=fn_coordify(q[:, -1:].bool()),
                    context_v=ctx_v,
                    context_world_coord_grid=x_coords,
                    affine_context_vox2world=x_affine_vox2world,
                    affine_query_vox2world=y_affine_vox2world,
                    context_vox_size_world=x_vox_size,
                    query_vox_size_world=y_vox_size,
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
                    fig = plt.figure(dpi=175, figsize=(10, 6))
                    fig = pitn.viz.plot_fodf_coeff_slices(
                        pred_fodf,
                        y,
                        torch.abs(pred_fodf - y) * y_mask_broad,
                        col_headers=("Predicted",) * 3
                        + ("Target",) * 3
                        + ("|Pred - GT|",) * 3,
                        row_headers=[f"z-harm deg {i}" for i in range(0, 9, 2)],
                        colorbars="rows",
                        fig=fig,
                        interpolation="nearest",
                        cmap="gray",
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
experiment_name = f"{ts}_inr-pred-test_native-res"
tmp_res_dir = Path(p.tmp_results_dir) / experiment_name
tmp_res_dir.mkdir(parents=True)

# %%
model = "INR"
model_pred_res_dir = tmp_res_dir / model
model_pred_res_dir.mkdir(exist_ok=True)
with open(model_pred_res_dir / "model_description.txt", "x") as f:
    f.write(f"model weights file: {str(p.model_weight_f)}\n")
    f.write(f"encoder parameters: \n{str(p.encoder.to_dict())}\n")
    f.write(f"decoder parameters: \n{str(p.decoder.to_dict())}\n")

# Wrap the entire loop in a try...except statement to save out a failure indicator file.
try:
    system_state_dict = torch.load(p.model_weight_f)
    encoder_state_dict = system_state_dict["encoder"]

    decoder_state_dict = system_state_dict["decoder"]

    if "in_channels" not in p.encoder:
        in_channels = int(test_dataset[0]["lr_dwi"].shape[0]) + 3
    else:
        in_channels = p.encoder.in_channels

    encoder = INREncoder(**{**p.encoder.to_dict(), **{"in_channels": in_channels}})
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    decoder = Decoder(**p.decoder.to_dict())
    decoder.load_state_dict(decoder_state_dict)
    decoder.to(device)
    del (
        system_state_dict,
        encoder_state_dict,
        decoder_state_dict,
    )
    with open(model_pred_res_dir / "model_description.txt", "a") as f:
        f.write(f"encoder layers: \n{str(encoder)}\n")
        f.write(f"decoder layers: \n{str(decoder)}\n")

    test_dataloader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        print("Starting inference", flush=True)
        for batch_dict in test_dataloader:
            subj_id = batch_dict["subj_id"]
            if len(subj_id) == 1:
                subj_id = subj_id[0]
            print(f"Starting {subj_id}", flush=True)

            x = batch_dict["lr_dwi"].to(device)
            batch_size = x.shape[0]
            x_mask = batch_dict["lr_brain_mask"].to(torch.bool).to(device)
            x_affine_vox2world = batch_dict["affine_lr_vox2world"].to(device)
            x_vox_size = batch_dict["lr_vox_size"].to(device)
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"].to(device)
            y_mask = batch_dict["brain_mask"].to(torch.bool).to(device)
            y_affine_vox2world = batch_dict["affine_vox2world"].to(device)
            y_vox_size = batch_dict["vox_size"].to(device)
            y_coords = pitn.affine.affine_coordinate_grid(
                y_affine_vox2world, tuple(y.shape[2:])
            )
            # Fix an edge case in the affine_coordinate_grid function.
            if batch_size == 1:
                if x_coords.shape[0] != 1:
                    x_coords.unsqueeze_(0)
                if y_coords.shape[0] != 1:
                    y_coords.unsqueeze_(0)

            # Concatenate the input world coordinates as input features into the
            # encoder. Mask out the x coordinates that are not to be considered.
            x_coord_mask = einops.rearrange(x_mask, "b 1 x y z -> b x y z 1")
            x_coords_encoder = einops.rearrange(
                x_coords * x_coord_mask, "b x y z coord -> b coord x y z"
            )
            x = torch.cat([x, x_coords_encoder], dim=1)
            ctx_v = encoder(x)

            # Whole-volume inference is memory-prohibitive, so use a sliding
            # window inference method on the encoded volume.
            # Transform y_coords into a coordinates-first shape, for the interface, and
            # attach the mask for compatibility with the sliding inference function.
            y_slide_window = torch.cat(
                [
                    einops.rearrange(y_coords, "b x y z coord -> b coord x y z"),
                    y_mask.to(y_coords),
                ],
                dim=1,
            )
            fn_coordify = lambda x: einops.rearrange(
                x, "b coord x y z -> b x y z coord"
            )
            # Keep the whole volume on the CPU, and only transfer the sliding windows
            # to the GPU.
            pred_fodf = monai.inferers.sliding_window_inference(
                y_slide_window.cpu(),
                roi_size=(96, 96, 96),
                sw_batch_size=batch_size,
                predictor=lambda q: decoder(
                    # Rearrange back into coord-last format.
                    query_world_coord=fn_coordify(q[:, :-1]).to(device),
                    query_world_coord_mask=fn_coordify(q[:, -1:].bool()).to(device),
                    context_v=ctx_v,
                    context_world_coord_grid=x_coords,
                    affine_context_vox2world=x_affine_vox2world,
                    affine_query_vox2world=y_affine_vox2world,
                    context_vox_size_world=x_vox_size,
                    query_vox_size_world=y_vox_size,
                ).cpu(),
                overlap=0,
                padding_mode="replicate",
            )

            # Write out prediction to a .nii.gz file.
            input_vox_size = x_vox_size.flatten().cpu().numpy()[0]
            native_vox_size = y_vox_size.flatten().cpu().numpy()[0]
            pred_f = (
                model_pred_res_dir
                / f"{subj_id}_{model}_prediction_{input_vox_size}mm-to-{native_vox_size}mm.nii.gz"
            )
            pred_affine = y_affine_vox2world[0].cpu().numpy()
            pred_fodf_vol = einops.rearrange(
                pred_fodf.detach().cpu().numpy(), "1 c x y z -> x y z c"
            ).astype(np.float32)
            pred_im = nib.Nifti1Image(pred_fodf_vol, affine=pred_affine)
            # Crop/pad prediction to align with the fodf image created directly from
            # mrtrix. This should not change any of the prediction values, only align
            # the images for easier comparison.
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_pred_dir = Path(tmpdirname)
                tmp_pred_f = str((tmp_pred_dir / "tmp_pred_im.nii.gz").resolve())
                nib.save(pred_im, tmp_pred_f)

                # Do the resampling with mrtrix directly.
                subj_source_files = (
                    pitn.data.datasets2.HCPfODFINRDataset.get_fodf_subj_dict(
                        subj_id, root_dir=hcp_full_res_fodf_dir
                    )
                )
                subj_source_fodf_f = str(subj_source_files["fodf"].resolve())
                resample_pred_f = str(
                    (tmp_pred_dir / "tmp_pred_im_aligned.nii.gz").resolve()
                )
                subprocess.run(
                    [
                        "mrgrid",
                        tmp_pred_f,
                        "regrid",
                        "-template",
                        subj_source_fodf_f,
                        "-interp",
                        "nearest",
                        "-scale",
                        "1,1,1",
                        "-datatype",
                        "float32",
                        resample_pred_f,
                        "-quiet",
                    ],
                    # env=os.environ,
                    timeout=60,
                    check=True,
                )

                shutil.move(resample_pred_f, pred_f)

            print(f"Finished {subj_id}", flush=True)

except KeyboardInterrupt as e:
    (tmp_res_dir / "STOPPED").touch()
    raise e
except Exception as e:
    (tmp_res_dir / "FAILED").touch()
    raise e
