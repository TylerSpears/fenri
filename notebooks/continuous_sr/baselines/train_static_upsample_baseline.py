# -*- coding: utf-8 -*-
# %% [markdown]
# # Static Super-Res CNN Baseline
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
import importlib.util
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

# from lightning_fabric.fabric import Fabric
from natsort import natsorted

import pitn

# Crazy hack for relative imports in interactive mode...
# <https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly>
mod_path = Path("../inr_networks.py")
mod_name = "inr_networks"
spec = importlib.util.spec_from_file_location(mod_name, mod_path)
inr_networks = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = inr_networks
spec.loader.exec_module(inr_networks)
from inr_networks import INREncoder, StaticSizeDecoder, StaticSizeUpsampleEncoder

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
p.experiment_name = "baseline-sr-cnn_split-01"
p.override_experiment_name = False
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
p.train_val_test_split_file = (
    Path("../data_splits") / "HCP_train-val-test_split_01_seed_332781572.csv"
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
p.checkpoint_epoch_ratios = [0.5]
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
    samples_per_subj_per_epoch=170,
    # samples_per_subj_per_epoch=25,  #!testing/debug
    max_epochs=50,
    # max_epochs=5,  #!testing/debug
    dwi_recon_epoch_proportion=0.03,
    # dwi_recon_epoch_proportion=0.01,  #!testing/debug
    sample_mask_key="wm_mask",
)
p.train.augment = dict(
    # augmentation_prob=0.0,  #!testing/debug
    # augmentation_prob=1.0,  #!testing/debug
    augmentation_prob=0.3,
    baseline_iso_scale_factor_lr_spacing_mm_low_high=p.baseline_lr_spacing_scale,
    scale_prefilter_kwargs=p.scale_prefilter_kwargs,
    augment_iso_scale_factor_lr_spacing_mm_low_high=(
        p.baseline_lr_spacing_scale,
        p.baseline_lr_spacing_scale,
    ),
    augment_rand_rician_noise_kwargs={"prob": 0.0},
    augment_rand_rotate_90_kwargs={"prob": 0.5},
    augment_rand_flip_kwargs={"prob": 0.5},
)
# Optimizer kwargs for training.
p.train.optim.encoder.lr = 5e-4
p.train.optim.decoder.lr = 5e-4
p.train.optim.recon_decoder.lr = 1e-3
# Train dataloader kwargs.
p.train.dataloader = dict(num_workers=15, persistent_workers=True, prefetch_factor=3)

# Network/model parameters.
p.encoder = dict(
    spatial_upscale_factor=p.baseline_lr_spacing_scale,
    input_coord_channels=True,
    interior_channels=80,
    out_channels=96,
    n_res_units=3,
    n_dense_units=3,
    activate_fn="relu",
)
p.decoder = dict(
    in_channels=p.encoder.out_channels,
    interior_channels=48,
    out_channels=45,
    n_res_units=2,
    n_dense_units=2,
    activate_fn="relu",
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
# Define the bval/bvec sub-sample scheme according to the parameter dict kwargs.
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
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"]
            y_mask = batch_dict["brain_mask"].to(torch.bool)
            if batch_size == 1:
                if x_coords.shape[0] != 1:
                    x_coords.unsqueeze_(0)

            # Concatenate the input world coordinates as input features into the
            # encoder. Mask out the x coordinates that are not to be considered.
            x_coord_mask = einops.rearrange(x_mask, "b 1 x y z -> b x y z 1")
            x_coords_encoder = einops.rearrange(
                x_coords * x_coord_mask, "b x y z coord -> b coord x y z"
            )
            x = torch.cat([x, x_coords_encoder], dim=1)
            y_pred = encoder(x)
            y_pred = decoder(y_pred)
            # Pad or crop the prediction spatial size to match the target spatial size.
            y_pred = decoder.crop_pad_to_match_gt_shape(
                model_output=y_pred, ground_truth=y, mode="constant"
            )

            y_mask_broad = torch.broadcast_to(y_mask, y.shape)
            # Calculate performance metrics
            mse_loss = batchwise_masked_mse(y_pred, y, mask=y_mask_broad)
            val_metrics["mse"].append(mse_loss.detach().cpu().flatten())

            # If visualization subj_id is in this batch, create the visual and log it.
            if subj_id == val_viz_subj_id:
                with mpl.rc_context({"font.size": 6.0}):
                    fig = plt.figure(dpi=175, figsize=(10, 6))
                    fig = pitn.viz.plot_fodf_coeff_slices(
                        y_pred,
                        y,
                        torch.abs(y_pred - y) * y_mask_broad,
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
                error_fodf = F.mse_loss(y_pred, y, reduction="none")
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
            del y_pred

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
fabric = lightning.Fabric(accelerator="gpu", devices=1, precision=32)
fabric.launch()
device = fabric.device

if fabric.is_global_zero:
    if "cuda" in device.type:
        torch.cuda.empty_cache()

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
    encoder = StaticSizeUpsampleEncoder(
        **{**p.encoder.to_dict(), **{"in_channels": in_channels}}
    )
    # Initialize weight shape for the encoder.
    encoder(torch.randn(1, in_channels, 20, 20, 20))
    decoder = StaticSizeDecoder(**p.decoder.to_dict())
    recon_decoder = INREncoder(
        in_channels=encoder.interior_channels,
        interior_channels=48,
        out_channels=9,
        n_res_units=2,
        n_dense_units=2,
        activate_fn=p.encoder.activate_fn,
        input_coord_channels=False,
    )
    # Initialize weight shape for all models.
    encoder(torch.randn(1, encoder.in_channels, 20, 20, 20))
    recon_decoder(torch.randn(1, recon_decoder.in_channels, 20, 20, 20))
    decoder(torch.randn(1, decoder.in_channels, 20, 20, 20))
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
    def _pad_list_data_collate_to_tensor(d, **kwargs):
        ret = monai.data.utils.pad_list_data_collate(d, **kwargs)
        return {
            k: monai.utils.convert_to_tensor(v, track_meta=False)
            if isinstance(v, monai.data.MetaObj)
            else v
            for k, v in ret.items()
        }

    train_dataloader = monai.data.DataLoader(
        train_dataset,
        batch_size=p.train.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=partial(
            _pad_list_data_collate_to_tensor, method="symmetric", mode="constant"
        ),
        **p.train.dataloader.to_dict(),
    )
    val_dataloader = monai.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        collate_fn=partial(
            _pad_list_data_collate_to_tensor, method="symmetric", mode="constant"
        ),
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
            batch_size = x.shape[0]
            x_mask = batch_dict["lr_brain_mask"].to(torch.bool)
            x_affine_vox2world = batch_dict["affine_lr_vox2world"]
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            y = batch_dict["fodf"]
            y_mask = batch_dict["brain_mask"].to(torch.bool)
            if batch_size == 1:
                if x_coords.shape[0] != 1:
                    x_coords.unsqueeze_(0)

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

            if not train_recon:
                ctx_v = encoder(x)
                pred_fodf = decoder(ctx_v)
                pred_fodf = decoder.crop_pad_to_match_gt_shape(
                    model_output=pred_fodf, ground_truth=y, mode="constant"
                )
                y_mask_broad = torch.broadcast_to(y_mask, y.shape)
                loss_fodf = loss_fn(pred_fodf[y_mask_broad], y[y_mask_broad])
                loss_recon = y.new_zeros(1)
                recon_pred = None
            else:
                # Run encoder without the upsampling layers to keep the same spatial
                # shape.
                ctx_v = encoder(x, upsample=False)
                recon_pred = recon_decoder(ctx_v)
                # Index bvals to be 2 b=0s, 2 b=1000s, and 2 b=3000s.
                recon_y = x[:, (0, 1, 2, 11, 12, 13, -3, -2, -1)]
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
        del x, x_coords, y, pred_fodf, recon_pred

        fabric.print("\n==Validation==", flush=True)
        fabric.barrier()
        if fabric.is_global_zero:
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
