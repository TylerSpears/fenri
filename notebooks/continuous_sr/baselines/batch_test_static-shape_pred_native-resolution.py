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
import gc
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

# Crazy hack for relative imports in interactive mode...
# <https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly>
mod_path = Path("../inr_networks.py")
mod_name = "inr_networks"
spec = importlib.util.spec_from_file_location(mod_name, mod_path)
inr_networks = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = inr_networks
spec.loader.exec_module(inr_networks)
from inr_networks import StaticSizeDecoder, StaticSizeUpsampleEncoder

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
    gc.collect()
    torch.cuda.empty_cache()
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
p.experiment_name = "static-shape_pred_native-res"
p.model_name = "static-shape_CNN"
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
# p.train_val_test_split_file = random.choice(
#     list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
# )
p.model_weight_f = str(
    Path(p.results_dir)
    / "2023-06-22T16_01_59__static-shape-comparison_split-01"
    / "final_state_dict_epoch_49_step_28351.pt"
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
            # holdout subjects that have been processed
            # 581450,
            # 126426,
            # 191336,
            # 251833,
            # 581450,
            # 601127,
            # 825048,
            # # test set subjects
            # 110613,
            # 112112,
            # 123420,
            # 124422,
            # 126628,
            # 129028,
            # 130013,
            # 133019,
            # 134425,
            # 135225,
            # 138837,
            # 139637,
            # 139839,
            # 143830,
            # 144428,
            # 144933,
            # 148840,
            149539,
            150019,
            151526,
            153227,
            153732,
            155231,
            162329,
            187850,
            189349,
            192843,
            193239,
            198451,
            220721,
            268850,
            270332,
            299154,
            314225,
            316633,
            350330,
            368551,
            453542,
            480141,
            492754,
            497865,
            500222,
            519647,
            567961,
            571144,
            656253,
            656657,
            677968,
            683256,
            704238,
            727654,
            731140,
            765056,
            767464,
            917558,
            930449,
            972566,
            978578,
            993675,
            994273,
        ],
    )
)
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
    # test_dataset = monai.data.CacheDataset(
    #     test_dataset,
    #     cache_num=1,
    #     transform=None,
    #     progress=True,
    #     copy_cache=False,
    #     num_workers=1,
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
# ## Testing

# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
ts = ts.replace(":", "_")
experiment_name = f"{ts}_{p.experiment_name}"
tmp_res_dir = Path(p.tmp_results_dir) / experiment_name
tmp_res_dir.mkdir(parents=True)

# %%
model_pred_res_dir = tmp_res_dir / p.model_name
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

    encoder = StaticSizeUpsampleEncoder(
        **{**p.encoder.to_dict(), **{"in_channels": in_channels}}
    )
    # Initialize weight shape for the encoder.
    encoder(torch.randn(1, in_channels, 20, 20, 20))
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    decoder = StaticSizeDecoder(**p.decoder.to_dict())
    # Initialize weight shape for the decoder.
    decoder(torch.randn(1, decoder.in_channels, 20, 20, 20))
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
        shuffle=False,
        pin_memory=True,
        num_workers=3,
        persistent_workers=True,
        prefetch_factor=1,
    )

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        print("Starting inference", flush=True)
        if "cuda" in device.type:
            gc.collect()
            torch.cuda.empty_cache()
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
            y_vox_size = batch_dict["vox_size"].to(device)
            y_affine_vox2world = batch_dict["affine_vox2world"].to(device)
            y_mask = batch_dict["brain_mask"].to(torch.bool).to(device)
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
            pred_fodf = encoder(x)
            pred_fodf = decoder(pred_fodf)
            # Pad or crop the prediction spatial size to match the target spatial size.
            pred_fodf = decoder.crop_pad_to_match_gt_shape(
                model_output=pred_fodf, ground_truth=y, mode="constant"
            )

            print(f"Finished inference {subj_id}", flush=True)

            # Write out prediction to a .nii.gz file.
            input_vox_size = x_vox_size.flatten().cpu().numpy()[0]
            native_vox_size = y_vox_size.flatten().cpu().numpy()[0]
            pred_f = (
                model_pred_res_dir
                / f"{subj_id}_{p.model_name}_prediction_{input_vox_size}mm-to-{native_vox_size}mm.nii.gz"
            )
            pred_affine = y_affine_vox2world[0].cpu().numpy()
            # Mask the prediction to reduce the file size.
            pred_fodf = pred_fodf.detach().cpu() * y_mask.detach().cpu()
            pred_fodf_vol = einops.rearrange(
                pred_fodf.numpy(), "1 c x y z -> x y z c"
            ).astype(np.float32)
            pred_im = nib.Nifti1Image(pred_fodf_vol, affine=pred_affine)
            # Crop/pad prediction to align with the fodf image created directly from
            # mrtrix. This should not change any of the prediction values, only align
            # the images for easier comparison.
            print(f"Aligning and saving prediction image {subj_id}")
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
