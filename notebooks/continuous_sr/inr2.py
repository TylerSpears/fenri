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
from functools import partial
from pathlib import Path
from pprint import pprint as ppr

import aim
import dotenv
import einops
import lightning

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
from inr_networks import Decoder, INREncoder, SimplifiedDecoder

# from lightning_fabric.fabric import Fabric
from natsort import natsorted

import pitn

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})
plt.rcParams.update({"image.cmap": "gray"})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)
torch.set_printoptions(sci_mode=False, threshold=100, linewidth=88)

# %%
# MAIN
# if __name__ == "__main__":

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
p.experiment_name = "test_inr_bvec-resample_fixed-mlp_subj_split-01.1"
p.override_experiment_name = False
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
p.train_val_test_split_file = (
    Path("./data_splits") / "HCP_train-val-test_split_01.1.csv"
)
# p.train_val_test_split_file = random.choice(
#     list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
# )
p.aim_logger = dict(
    repo="aim://dali.cpe.virginia.edu:53800",
    experiment="PITN_INR",
    meta_params=dict(run_name=p.experiment_name),
    tags=("PITN", "INR", "HCP", "super-res", "dMRI"),
)
p.checkpoint_epoch_ratios = (0.2, 0.33, 0.66)
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
p.train = dict(
    patch_spatial_size=(36, 36, 36),
    batch_size=6,
    samples_per_subj_per_epoch=110,
    # samples_per_subj_per_epoch=20, #!DEBUG
    max_epochs=50,
    dwi_recon_epoch_proportion=0.001,
    # dwi_recon_epoch_proportion=0.01,  #!DEBUG
    sample_mask_key="wm_mask",
)
p.train.augment = dict(
    augmentation_prob=0.3,
    baseline_iso_scale_factor_lr_spacing_mm_low_high=p.baseline_lr_spacing_scale,
    scale_prefilter_kwargs=p.scale_prefilter_kwargs,
    augment_iso_scale_factor_lr_spacing_mm_low_high=(1.61, 2.0),
    augment_rand_rician_noise_kwargs={"prob": 0.0},
    augment_rand_rotate_90_kwargs={"prob": 0.5},
    augment_rand_flip_kwargs={"prob": 0.5},
    keep_metatensor_output=True,
)
# Optimizer kwargs for training.
p.train.optim.encoder.lr = 5e-4
p.train.optim.decoder.lr = 5e-4
p.train.optim.recon_decoder.lr = 1e-3
# Train dataloader kwargs.
p.train.dataloader = dict(num_workers=17, persistent_workers=True, prefetch_factor=3)

# Network/model parameters.
p.encoder = dict(
    interior_channels=80,
    out_channels=96,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="relu",
    input_coord_channels=True,
    post_batch_norm=True,
)
p.decoder = dict(
    context_v_features=96,
    out_features=45,
    m_encode_num_freqs=36,
    sigma_encode_scale=3.0,
    n_internal_features=256,
    n_internal_layers=3,
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
ic(p.to_dict())

# %%
# Select which parameters to store in the aim meta-params.
p.aim_logger.meta_params.hparams = dict(
    batch_size=p.train.batch_size,
    patch_spatial_size=p.train.patch_spatial_size,
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

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()

# %%
# Define hte bval/bvec sub-sample scheme according to the parameter dict kwargs.
bval_sub_sample_fn = partial(
    pitn.data.datasets2.sub_select_dwi_from_bval,
    **p.bval_sub_sample_fn_kwargs.to_dict(),
)

# %% [markdown]
# ### Create Patch-Based Training Dataset

# %%
DEBUG_TRAIN_DATA_SUBJS = 2
with warnings.catch_warnings(record=True) as warn_list:

    # print("DEBUG Train subject numbers")
    pre_sample_ds = pitn.data.datasets2.HCPfODFINRDataset(
        # subj_ids=p.train.subj_ids[:DEBUG_TRAIN_DATA_SUBJS],  #!DEBUG
        subj_ids=p.train.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        transform=None,
    )

    pre_sample_train_dataset = monai.data.CacheDataset(
        pre_sample_ds,
        transform=pre_sample_ds.default_pre_sample_tf(
            sample_mask_key=p.train.sample_mask_key,
            bval_sub_sample_fn=bval_sub_sample_fn,
        ),
        copy_cache=False,
        num_workers=8,
    )

train_dataset = pitn.data.datasets2.HCPfODFINRPatchDataset(
    pre_sample_train_dataset,
    patch_func=pitn.data.datasets2.HCPfODFINRPatchDataset.default_patch_func(
        spatial_size=p.train.patch_spatial_size,
        num_samples=p.train.samples_per_subj_per_epoch,
    ),
    samples_per_image=p.train.samples_per_subj_per_epoch,
    transform=pitn.data.datasets2.HCPfODFINRPatchDataset.default_feature_tf(
        **p.train.augment.to_dict()
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
# #!DEBUG
DEBUG_VAL_SUBJS = 2
with warnings.catch_warnings(record=True) as warn_list:

    # print("DEBUG Val subject numbers")
    val_ds = pitn.data.datasets2.HCPfODFINRDataset(
        # subj_ids=p.val.subj_ids[:DEBUG_VAL_SUBJS],  #!DEBUG
        subj_ids=p.val.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        transform=pitn.data.datasets2.HCPfODFINRDataset.default_pre_sample_tf(
            sample_mask_key=p.train.sample_mask_key,
            bval_sub_sample_fn=bval_sub_sample_fn,
        ),
    )

    val_dataset = pitn.data.datasets2.HCPfODFINRWholeBrainDataset(
        val_ds,
        transform=pitn.data.datasets2.HCPfODFINRWholeBrainDataset.default_vol_tf(
            baseline_iso_scale_factor_lr_spacing_mm_low_high=p.baseline_lr_spacing_scale,
            scale_prefilter_kwargs=p.scale_prefilter_kwargs,
            keep_metatensor_output=True,
        ),
    )
    # Cache the transformations on the validation data.
    val_dataset = monai.data.CacheDataset(
        val_dataset,
        transform=None,
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
            x_affine_vox2world = batch_dict["affine_lr_vox2world"].to(x.dtype)
            x_vox_size = batch_dict["lr_vox_size"]
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"]
            y_mask = batch_dict["brain_mask"].to(torch.bool)
            y_affine_vox2world = batch_dict["affine_vox2world"].to(y.dtype)
            y_vox_size = batch_dict["vox_size"]
            y_coords = pitn.affine.affine_coordinate_grid(
                y_affine_vox2world, tuple(y.shape[2:])
            )
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
            mse_loss = batchwise_masked_mse(pred_fodf, y, mask=y_mask_broad)
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
                    fig = plt.figure(dpi=100, figsize=(6, 2))
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
    return aim_run, val_viz_subj_id, val_metrics["mse"]


# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
tmp_res_dir = Path(p.tmp_results_dir) / ts
tmp_res_dir.mkdir(parents=True)

# %%
fabric = lightning.Fabric(accelerator="gpu", devices=1, precision=32)
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
    in_channels = int(train_dataset[0]["lr_dwi"].shape[0]) + 3
else:
    in_channels = p.encoder.in_channels

# Wrap the entire training & validation loop in a try...except statement.
try:

    LOSS_ODF_COEFF_MEANS = torch.from_numpy(
        np.array([0.17] + [0.002] * 5 + [0.002] * 9 + [0.0] * 13 + [0.0] * 17)
    )
    LOSS_ODF_COEFF_STDS = torch.from_numpy(
        np.array([0.05] + [0.1] * 5 + [0.06] * 9 + [0.03] * 13 + [0.01] * 17)
    )
    LOSS_ODF_COEFF_MEANS = LOSS_ODF_COEFF_MEANS[None, :, None, None, None].to(device)
    LOSS_ODF_COEFF_STDS = LOSS_ODF_COEFF_STDS[None, :, None, None, None].to(device)

    encoder = INREncoder(**{**p.encoder.to_dict(), **{"in_channels": in_channels}})
    # Initialize weight shape for the encoder.
    encoder(torch.randn(1, in_channels, 20, 20, 20))
    decoder = SimplifiedDecoder(**p.decoder.to_dict())

    decoder = SimplifiedDecoder(**p.decoder.to_dict())
    recon_decoder = INREncoder(
        in_channels=encoder.out_channels,
        interior_channels=48,
        out_channels=9,
        n_res_units=2,
        n_dense_units=2,
        activate_fn=p.encoder.activate_fn,
        input_coord_channels=False,
    )
    # Initialize weight shape for the recon decoder.
    recon_decoder(torch.randn(1, recon_decoder.in_channels, 20, 20, 20))
    fabric.print(p.to_dict())
    fabric.print(encoder)
    fabric.print(decoder)
    fabric.print(recon_decoder)
    fabric.print("Encoder num params:", sum([p.numel() for p in encoder.parameters()]))
    fabric.print("Decoder num params:", sum([p.numel() for p in decoder.parameters()]))
    fabric.print(
        "Recon decoder num params:",
        sum([p.numel() for p in recon_decoder.parameters()]),
    )

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

    # The datasets will usually produce volumes of different shapes due to the possible
    # random re-sampling, so the batch must be padded, and the padded masks must be
    # used to calculate the loss.
    collate_fn = partial(
        pitn.data.datasets2.pad_list_data_collate_update_affines_to_tensor,
        meta_tensor_key_write_aff_key_pairs=[
            ("lr_dwi", "affine_lr_vox2world"),
            ("fodf", "affine_vox2world"),
        ],
        method="symmetric",
        mode="constant",
    )
    train_dataloader = monai.data.DataLoader(
        train_dataset,
        batch_size=p.train.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        **p.train.dataloader.to_dict(),
    )
    val_dataloader = monai.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    val_viz_subj_id = None

    encoder.train()
    decoder.train()
    recon_decoder.train()
    losses = dict(
        loss=list(),
        epoch=list(),
        step=list(),
    )
    step = 1
    train_dwi_recon_epoch_proportion = p.train.dwi_recon_epoch_proportion
    train_recon = False

    epochs = p.train.max_epochs
    curr_best_val_score = 1e8
    checkpoint_epochs = np.floor(np.array(p.checkpoint_epoch_ratios) * epochs)
    checkpoint_epochs = set(checkpoint_epochs.astype(int).tolist())
    curr_checkpoint = 0
    for epoch in range(epochs):
        fabric.print(f"\nEpoch {epoch}\n", "=" * 10)
        if epoch <= math.floor(epochs * train_dwi_recon_epoch_proportion):
            if not train_recon:
                train_recon = True
        else:
            if train_recon:
                train_recon = False
                fabric.barrier()
                if fabric.is_global_zero:
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step,
                            "aim_run_hash": aim_run.hash,
                            "recon_decoder": recon_decoder.state_dict(),
                            "optim_recon_decoder": optim_recon_decoder.state_dict(),
                        },
                        Path(tmp_res_dir)
                        / f"recon_decoder_state_dict_epoch_{epoch}_step_{step}.pt",
                    )
                    # Replace the recon network and optimization model with dummies, to
                    # release gpu memory.
                    del recon_decoder
                    del optim_recon_decoder
                    recon_decoder = torch.nn.Linear(1, 1, bias=False)
                    optim_recon_decoder = torch.optim.SGD(
                        recon_decoder.parameters(), 1e-3
                    )
                    recon_decoder, optim_recon_decoder = fabric.setup(
                        recon_decoder, optim_recon_decoder
                    )
                    fabric.barrier()

        for batch_dict in train_dataloader:

            x = batch_dict["lr_dwi"]
            x_mask = batch_dict["lr_brain_mask"].to(torch.bool)
            x_affine_vox2world = batch_dict["affine_lr_vox2world"].to(x.dtype)
            x_vox_size = batch_dict["lr_vox_size"]
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"]
            y_mask = batch_dict["brain_mask"].to(torch.bool)
            y_affine_vox2world = batch_dict["affine_vox2world"].to(y.dtype)
            y_vox_size = batch_dict["vox_size"]
            y_coords = pitn.affine.affine_coordinate_grid(
                y_affine_vox2world, tuple(y.shape[2:])
            )

            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            optim_recon_decoder.zero_grad()

            # Concatenate the input world coordinates as input features into the
            # encoder. Mask out the x coordinates that are not to be considered.
            x_coord_mask = einops.rearrange(x_mask, "b 1 x y z -> b x y z 1")
            x_coords_encoder = einops.rearrange(
                x_coords * x_coord_mask, "b x y z coord -> b coord x y z"
            )
            x = torch.cat([x, x_coords_encoder], dim=1)
            ctx_v = encoder(x)

            if not train_recon:
                y_mask_broad = torch.broadcast_to(y_mask, y.shape)
                y_coord_mask = einops.rearrange(y_mask, "b 1 x y z -> b x y z 1")
                pred_fodf = decoder(
                    context_v=ctx_v,
                    context_world_coord_grid=x_coords,
                    query_world_coord=y_coords,
                    query_world_coord_mask=y_coord_mask,
                    affine_context_vox2world=x_affine_vox2world,
                    affine_query_vox2world=y_affine_vox2world,
                    context_vox_size_world=x_vox_size,
                    query_vox_size_world=y_vox_size,
                )

                # Scale prediction and target to weigh the loss.
                # Transform coefficients to a standard normal distribution.
                # 45 x n_voxels
                pred_fodf_standardized = (
                    pred_fodf - LOSS_ODF_COEFF_MEANS
                ) / LOSS_ODF_COEFF_STDS
                y_standardized = (y - LOSS_ODF_COEFF_MEANS) / LOSS_ODF_COEFF_STDS
                # Tissue weights
                tissue_weight_mask = y_mask.to(pred_fodf)
                tissue_weight_mask[batch_dict["gm_mask"]] = 0.3
                tissue_weight_mask[batch_dict["csf_mask"]] = 0.1
                tissue_weight_mask[batch_dict["wm_mask"]] = 1.0
                pred_fodf_standardized *= tissue_weight_mask
                y_standardized *= tissue_weight_mask
                # Calculate loss over weighted prediction and target.
                loss_fodf = loss_fn(
                    pred_fodf_standardized[y_mask_broad], y_standardized[y_mask_broad]
                )

                # loss_fodf = loss_fn(pred_fodf[y_mask_broad], y[y_mask_broad])
                loss_recon = y.new_zeros(1)
                recon_pred = None
            else:
                recon_pred = recon_decoder(ctx_v)
                # Index bvals to be 2 b=0s, 2 b=1000s, and 2 b=3000s.
                recon_y = torch.cat(
                    [x[:, (0, 1, 2, 11, 12, 13)], x_coords.movedim(-1, 1)], dim=1
                )
                x_mask_broad = torch.broadcast_to(x_mask, recon_y.shape)
                loss_recon = recon_loss_fn(
                    recon_pred[x_mask_broad], recon_y[x_mask_broad]
                )
                loss_fodf = recon_y.new_zeros(1)
                pred_fodf = None

            loss = loss_fodf + loss_recon

            fabric.backward(loss)
            for model, model_optim in zip(
                (encoder, decoder, recon_decoder),
                (optim_encoder, optim_decoder, optim_recon_decoder),
            ):
                if train_recon and model is decoder:
                    continue
                elif not train_recon and model is recon_decoder:
                    continue
                fabric.clip_gradients(
                    model,
                    model_optim,
                    max_norm=5.0,
                    norm_type=2,
                    error_if_nonfinite=True,
                )
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     5.0,
                #     error_if_nonfinite=True,
                # )
            optim_encoder.step()
            optim_decoder.step()
            optim_recon_decoder.step()

            to_track = {
                "loss": loss.detach().cpu().item(),
            }
            # Depending on whether or not the reconstruction decoder is training,
            # select which metrics to track at this time.
            if train_recon:
                to_track = {
                    **to_track,
                    **{
                        "loss_recon": loss_recon.detach().cpu().item(),
                    },
                }
            else:
                to_track = {
                    **to_track,
                    **{
                        "loss_pred_fodf": loss_fodf.detach().cpu().item(),
                    },
                }
            if fabric.is_global_zero:
                aim_run.track(
                    to_track,
                    context={
                        "subset": "train",
                    },
                    step=step,
                    epoch=epoch,
                )
                losses["loss"].append(loss.detach().cpu().item())
                losses["epoch"].append(epoch)
                losses["step"].append(step)

            fabric.print(
                f"| {loss.detach().cpu().item()}",
                end=" ",
                flush=(step % 10) == 0,
            )

            step += 1

        optim_encoder.zero_grad(set_to_none=True)
        optim_decoder.zero_grad(set_to_none=True)
        optim_recon_decoder.zero_grad(set_to_none=True)
        # Delete some training inputs to relax memory constraints in whole-
        # volume inference inside validation step.
        del x, x_coords, y, y_coords, pred_fodf, recon_pred

        fabric.print("\n==Validation==", flush=True)
        fabric.barrier()
        if fabric.is_global_zero:
            aim_run, val_viz_subj_id, val_scores = validate_stage(
                fabric,
                encoder,
                decoder,
                val_dataloader=val_dataloader,
                step=step,
                epoch=epoch,
                aim_run=aim_run,
                val_viz_subj_id=val_viz_subj_id,
            )
            curr_val_score = val_scores.detach().cpu().mean().item()
        fabric.barrier()

        # Start saving best performing models if the previous best val score was
        # surpassed, and the current number of epcohs is >= half the total training
        # amount.
        if (curr_val_score < (curr_best_val_score - (0.05 * curr_best_val_score))) and (
            epoch >= epochs / 3
        ):
            fabric.print("Saving new best validation score")
            fabric.barrier()
            if fabric.is_global_zero:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "aim_run_hash": aim_run.hash,
                        "optim_encoder": optim_encoder.state_dict(),
                        "optim_decoder": optim_decoder.state_dict(),
                        "recon_decoder": recon_decoder.state_dict(),
                        "optim_recon_decoder": optim_recon_decoder.state_dict(),
                    },
                    Path(tmp_res_dir)
                    / f"best_val_score_state_dict_epoch_{epoch}_step_{step}.pt",
                )
                curr_best_val_score = curr_val_score
            fabric.barrier()

        if epoch in checkpoint_epochs:
            fabric.print(f"Saving checkpoint {curr_checkpoint}")
            fabric.barrier()
            if fabric.is_global_zero:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "aim_run_hash": aim_run.hash,
                        "optim_encoder": optim_encoder.state_dict(),
                        "optim_decoder": optim_decoder.state_dict(),
                        "recon_decoder": recon_decoder.state_dict(),
                        "optim_recon_decoder": optim_recon_decoder.state_dict(),
                    },
                    Path(tmp_res_dir)
                    / f"checkpoint_{curr_checkpoint}_state_dict_epoch_{epoch}_step_{step}.pt",
                )
            fabric.barrier()
            curr_checkpoint += 1

except KeyboardInterrupt as e:
    if fabric.is_global_zero:
        aim_run.add_tag("STOPPED")
        (tmp_res_dir / "STOPPED").touch()
    raise e
except Exception as e:
    if fabric.is_global_zero:
        aim_run.add_tag("FAILED")
        (tmp_res_dir / "FAILED").touch()
    raise e
finally:
    if fabric.is_global_zero:
        aim_run.close()

# Sync all pytorch-lightning processes.
fabric.barrier()
if fabric.is_global_zero:
    fabric.print("Saving model state dict")
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
        Path(tmp_res_dir) / f"final_state_dict_epoch_{epoch}_step_{step}.pt",
    )
    fabric.print("=" * 40)
    losses = pd.DataFrame.from_dict(losses)
    losses.to_csv(Path(tmp_res_dir) / "train_losses.csv")


# %%
