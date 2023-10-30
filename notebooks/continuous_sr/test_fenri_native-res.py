#!/usr/bin/env python
# -*- coding: utf-8 -*-
# imports
import argparse
import collections
import concurrent.futures
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
import pprint
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
import lightning

# visualization libraries
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
from natsort import natsorted

import pitn


def log_prefix():
    return f"{datetime.datetime.now().replace(microsecond=0)}"


def pad_vol_to_template_mrtrix(vol_f: Path, template_f: Path, output_f: Path):

    cmd = rf"""set -eou pipefail
    mrconvert -quiet "{vol_f}" - |
        mrgrid -force -quiet \
            - \
            regrid \
            -template "{template_f}" \
            -interp nearest \
            -fill 0.0 \
            '{output_f}'"""
    popen_args = ["/usr/bin/bash", "-c", cmd]

    return_status = pitn.utils.proc_runner.call_shell_exec(
        cmd=cmd,
        args="",
        cwd=os.getcwd(),
        env=os.environ,
        popen_args_override=popen_args,
    )

    return return_status


if __name__ == "__main__":
    # Take some parameters from the command line.
    parser = argparse.ArgumentParser(
        description="Test trilinear DWI upsampling + mrtrix msmt-csd odf fitting"
    )
    parser.add_argument(
        "-i",
        "--input_subj_ids",
        type=Path,
        # default="test_subj_ids.txt",
        # default="split_01.1_test_subj_ids.txt",
        required=True,
        help=".txt file with subj ids to process",
    )
    parser.add_argument(
        "-w",
        "--input_weights_file",
        type=Path,
        # default="/data/srv/outputs/pitn/results/tmp/2023-10-15T01_59_59/best_val_score_state_dict_epoch_41_step_24529.pt",
        required=True,
        help="Pytorch .pt weights file with trained network weights",
    )
    parser.add_argument(
        "-d",
        "--data_root_dir",
        type=Path,
        default=Path("/data/srv/outputs/pitn/hcp"),
        help="Root directory that contains subj data",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=Path("/data/srv/outputs/pitn/results/runs/fenri"),
        type=Path,
        help="Output directory for all estimated ODF volumes",
    )
    parser.add_argument(
        "-c",
        "--create_output_subdir",
        type=int,
        default=1,
        help="Output directory for all estimated ODF volumes",
    )
    args = parser.parse_args()

    # Update environment variables with direnv.
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
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd()
    )
    # Store and format the subprocess' output.
    proc_out = proc.communicate()[0].strip().decode("utf-8")
    # Use python-dotenv to load the environment variables by using the output of
    # 'direnv exec ...' as a 'dummy' .env file.
    dotenv.load_dotenv(stream=io.StringIO(proc_out), override=True)

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

    rng = pitn.utils.fork_rng(torch.default_generator)

    # Experiment parameters, some should be set in a config file
    p = Box(default_box=True, default_box_none_transform=False)
    p.experiment_name = "fenri_test_native_res"
    # Network/model parameters.
    p.encoder = dict(
        in_channels=9 + 45 + 45 + 3,
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
            config_fname = pitn.utils.system.get_file_glob_unique(
                Path("."), r"config.*"
            )
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

    # Prep data loading functions.
    hcp_data_root_dir = Path(args.data_root_dir)
    assert hcp_data_root_dir.exists()
    # Set paths relative to the subj id root dir for each required image/file.
    rel_dwi_path = Path("ras/diffusion/dwi_norm.nii.gz")
    rel_grad_table_path = Path("ras/diffusion/ras_grad_mrtrix.b")
    rel_odf_path = Path("ras/odf/wm_msmt_csd_norm_odf.nii.gz")
    rel_fivett_seg_path = Path("ras/segmentation/fivett_dwi-space_segmentation.nii.gz")
    rel_brain_mask_path = Path("ras/brain_mask.nii.gz")
    # Set a common target set of gradient directions/strengths based on the standard HCP
    # protocol.
    first_n_b0s = 9
    first_n_b1000s = 45
    # Remove the b2000s entirely.
    first_n_b3000s = 45
    target_grad_table = pitn.data.HCP_STANDARD_3T_GRAD_MRTRIX_TABLE
    shells = target_grad_table.b.to_numpy().round(-2)
    row_select_template = np.zeros_like(shells).astype(bool)
    # Take first N b0s
    b0_idx = (np.where(shells == 0)[0][:first_n_b0s],)
    b0_select = row_select_template.copy()
    b0_select[b0_idx] = True
    # Take first N b1000s
    b1000_idx = (np.where(shells == 1000)[0][:first_n_b1000s],)
    b1000_select = row_select_template.copy()
    b1000_select[b1000_idx] = True
    # Take first N b3000s
    b3000_idx = (np.where(shells == 3000)[0][:first_n_b3000s],)
    b3000_select = row_select_template.copy()
    b3000_select[b3000_idx] = True
    dwi_select_mask = b0_select | b1000_select | b3000_select
    target_grad_table = target_grad_table.loc[dwi_select_mask]

    preproc_loaded_kwargs = dict(
        S0_noise_b0_quantile=p.preproc_loaded.S0_noise_b0_quantile,
        patch_sampling_w_erosion=p.preproc_loaded.patch_sampling_w_erosion,
        resample_target_grad_table=target_grad_table,
    )
    # Make one-param callable for map(), but with undefined values already defined on
    # the outer scope (bound).
    def load_and_tf_subj(subj_files: dict):
        print(
            f"{log_prefix()}| Loading subj {subj_files['subj_id']}...\n",
            end="",
            flush=True,
        )
        v = pitn.data.load_super_res_subj_sample(**subj_files)
        print(
            f"{log_prefix()}| Finished loading subj {subj_files['subj_id']}\n",
            end="",
            flush=True,
        )
        v = pitn.data.preproc.preproc_loaded_super_res_subj(v, **preproc_loaded_kwargs)
        # Perform random transforms with a set seed.
        subj_rng_seed = int(p.test.rng_seed) ^ int(subj_files["subj_id"])
        rng_device = rng.device
        v = pitn.data.preproc.preproc_super_res_sample(
            v,
            **p.test.vol_tf.to_dict(),
            rng=torch.Generator(device=rng_device).manual_seed(subj_rng_seed),
        )
        print(
            f"{log_prefix()}| Finished processing subj {subj_files['subj_id']}\n",
            end="",
            flush=True,
        )
        return v

    # Collect input files for test subjects.
    # DEBUG_TEST_DATA_SUBJS = 2
    test_subj_ids = list()
    with open(args.input_subj_ids, "r") as f:
        for line in f:
            if str(line).strip().startswith("#"):
                continue
            test_subj_ids.append(str(line).strip())
    # test_subj_ids = test_subj_ids[:DEBUG_TEST_DATA_SUBJS]  #!DEBUG
    test_subj_dicts = list()
    test_subj_files = dict()
    for subj_id in test_subj_ids:
        root_dir = hcp_data_root_dir / str(subj_id)
        d = dict(
            subj_id=str(subj_id),
            dwi_f=root_dir / rel_dwi_path,
            grad_mrtrix_f=root_dir / rel_grad_table_path,
            odf_f=root_dir / rel_odf_path,
            brain_mask_f=root_dir / rel_brain_mask_path,
            fivett_seg_f=root_dir / rel_fivett_seg_path,
        )
        test_subj_dicts.append(d)
        test_subj_files[str(subj_id)] = d

    out_dir = Path(args.output_dir)
    if bool(args.create_output_subdir):
        ts = datetime.datetime.now().replace(microsecond=0).isoformat()
        # Break ISO format because many programs don't like having colons ':' in a
        # filename.
        ts = ts.replace(":", "_")
        sub_dir = f"{ts}_{p.experiment_name}"
        out_dir = out_dir / sub_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap the entire test loop in a try...except statement to save out a failure
    # indicator file.
    try:
        # Set up inference model.
        weights_f = Path(args.input_weights_file)
        # Log model & test run parameters.
        log_test_f = out_dir / "log_test_params.txt"
        with open(log_test_f, "a") as f:
            f.write(f"model weights file: {weights_f}\n")
            f.write(f"Test Parameters\n{str(pprint.pformat(p.to_dict()))}\n")

        system_state_dict = torch.load(weights_f)
        encoder_state_dict = system_state_dict["encoder"]
        decoder_state_dict = system_state_dict["decoder"]
        encoder = INREncoder(**p.encoder.to_dict())
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device)
        decoder = SimplifiedDecoder(**p.decoder.to_dict())
        decoder.load_state_dict(decoder_state_dict)
        decoder.to(device)
        del (
            system_state_dict,
            encoder_state_dict,
            decoder_state_dict,
        )
        with open(log_test_f, "a") as f:
            f.write(f"encoder layers: \n{str(encoder)}\n")
            f.write(f"decoder layers: \n{str(decoder)}\n")
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            print(f"{log_prefix()}| Starting inference", flush=True)
            if "cuda" in device.type:
                torch.cuda.empty_cache()

            # Iterate over all preproced subjects.
            for sample_dict in map(load_and_tf_subj, test_subj_dicts):
                subj_id = sample_dict["subj_id"]
                print(f"{log_prefix()}| Starting subj {subj_id}", flush=True)

                x = sample_dict["lr_dwi"].unsqueeze(0).to(device)
                batch_size = x.shape[0]
                x_affine_vox2real = sample_dict["affine_lr_vox2real"].unsqueeze(0).to(x)
                x_spacing = (
                    torch.as_tensor(sample_dict["lr_spacing"])
                    .unsqueeze(0)
                    .to(x_affine_vox2real)
                )
                x_coords = einops.rearrange(
                    sample_dict["lr_real_coords"].unsqueeze(0),
                    "b coord x y z -> b x y z coord",
                ).to(device)

                y_mask = sample_dict["brain_mask"].unsqueeze(0).bool().to(device)
                y_spacing = (
                    torch.as_tensor(sample_dict["full_res_spacing"])
                    .unsqueeze(0)
                    .to(x_affine_vox2real)
                )
                y_coords = einops.rearrange(
                    sample_dict["full_res_real_coords"].unsqueeze(0),
                    "b coord x y z -> b x y z coord",
                ).to(device)

                # Append LR coordinates to the end of the input LR DWIs.
                x = torch.cat(
                    [x, einops.rearrange(x_coords, "b x y z coord -> b coord x y z")],
                    dim=1,
                )
                ctx_v = encoder(x)

                # Whole-volume inference is memory-prohibitive, so use a sliding
                # window inference method on the encoded volume.
                # Transform y_coords into a coordinates-first shape, for the interface,
                # and attach the mask for compatibility with the sliding inference
                # function.
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

                pred_odf = monai.inferers.sliding_window_inference(
                    y_slide_window,
                    roi_size=(72, 72, 72),
                    sw_batch_size=y_coords.shape[0],
                    predictor=lambda q: decoder(
                        # Rearrange back into coord-last format.
                        query_real_coords=fn_coordify(q[:, :-1]),
                        query_coords_mask=fn_coordify(q[:, -1:].bool()),
                        context_v=ctx_v,
                        context_real_coords=x_coords,
                        affine_context_vox2real=x_affine_vox2real,
                        context_spacing=x_spacing,
                        query_spacing=y_spacing,
                    ),
                    overlap=0,
                    padding_mode="replicate",
                )
                # Mask prediction.
                pred_odf = pred_odf * y_mask
                # Pad/align prediction volume with the full input volume shape. Should
                # not involve any interpolation, only padding/cropping.
                pred_odf = einops.rearrange(
                    pred_odf[0].cpu().numpy().astype(np.float32),
                    "coeff x y z -> x y z coeff",
                )
                affine_pred = sample_dict["affine_vox2real"].cpu().numpy()
                out_pred_f = out_dir / f"{subj_id}_fenri-pred_wm_msmt_csd_odf.nii.gz"
                template_f = test_subj_files[subj_id]["brain_mask_f"]
                with tempfile.TemporaryDirectory(prefix="test_fenri_") as tmp_dir_name:
                    tmp_dir = Path(tmp_dir_name)
                    pred_im = nib.Nifti1Image(pred_odf, affine_pred, dtype=np.float32)
                    tmp_pred_f = tmp_dir / "wm_odf.nii.gz"
                    nib.save(pred_im, tmp_pred_f)
                    pad_vol_to_template_mrtrix(
                        tmp_pred_f, template_f=template_f, output_f=out_pred_f
                    )

                print(f"{log_prefix()}| Finished subj {subj_id}", flush=True)

    except KeyboardInterrupt as e:
        (out_dir / "STOPPED").touch()
        raise e
    except Exception as e:
        (out_dir / "FAILED").touch()
        raise e
