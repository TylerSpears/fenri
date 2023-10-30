#!/usr/bin/env python
# -*- coding: utf-8 -*-
# imports
import argparse
import collections
import concurrent.futures
import copy
import datetime
import functools
import hashlib
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
from typing import Union

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


def create_subj_rng_seed(base_rng_seed: int, subj_id: Union[int, str]) -> int:
    try:
        subj_int = int(subj_id)
    except ValueError as e:
        if not isinstance(subj_id, str):
            raise e
        # Max hexdigest length that can fit into a 64-bit integer is length 8.
        hash_str = (
            hashlib.shake_128(subj_id.encode(), usedforsecurity=False)
            .hexdigest(8)
            .encode()
        )
        subj_int = int(hash_str, base=16)

    return base_rng_seed ^ subj_int


# MAIN
if __name__ == "__main__":
    # Take some parameters from the command line.
    parser = argparse.ArgumentParser(
        description="Test FENRI interpolation at an arbitrary resolution"
    )
    parser.add_argument("subj_id", type=str, help="Subject identification string")
    parser.add_argument(
        "weights",
        type=Path,
        # default="/data/srv/outputs/pitn/results/tmp/2023-10-15T01_59_59/best_val_score_state_dict_epoch_41_step_24529.pt",
        help="Pytorch .pt weights file with trained network weights",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Test parameters file to determine data parameters and network hyperparams, can be JSON, YAML, or TOML",
    )
    parser.add_argument("spacing", type=float, help="New isotropic voxel size in mm")
    parser.add_argument(
        "-d", "--dwi", type=Path, required=True, help="Native-resolution DWI NIFTI file"
    )
    parser.add_argument(
        "-g",
        "--grad_mrtrix",
        type=Path,
        required=True,
        help="Gradient table of the DWIs in MRtrix text file format",
    )
    parser.add_argument(
        "-f",
        "--odf",
        type=Path,
        required=True,
        help="Native-resolution ODF spherical harmonic coefficients NIFTI file",
    )
    parser.add_argument(
        "-t",
        "--fivett",
        type=Path,
        required=True,
        help="Native-resolution five tissue-type segmentation NIFTI file",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=Path,
        required=True,
        help="Native-resolution brain mask NIFTI file",
    )
    parser.add_argument(
        "-o",
        "--output",
        # default=Path("/data/srv/outputs/pitn/results/runs/fenri"),
        type=Path,
        required=True,
        help="Output ODF prediction NIFTI filename",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=argparse.FileType("at"),
        default="-",
        help="Output log file (optional, default stdout)",
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
    p.experiment_name = "fenri_test_super_res"

    # Load user config file.
    config_fname = args.config
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
    # Remove the default_box behavior now that params have been fully read in.
    _p = Box(default_box=False)
    _p.merge_update(p)
    p = _p

    # Prep data loading functions.
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
    # Make one-param callable, but with undefined values already defined on
    # the outer scope.
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
        subj_rng_seed = create_subj_rng_seed(
            int(p.test.rng_seed), subj_files["subj_id"]
        )
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

    # Collect input files for the test input.
    test_subj_id = args.subj_id
    test_subj_dict = dict(
        subj_id=test_subj_id,
        dwi_f=args.dwi,
        grad_mrtrix_f=args.grad_mrtrix,
        odf_f=args.odf,
        brain_mask_f=args.mask,
        fivett_seg_f=args.fivett,
    )
    test_subj_file = {test_subj_id: test_subj_dict}
    test_subj_dicts = [test_subj_dict]
    test_subj_files = [test_subj_file]

    out_f = args.output
    out_basename = out_f.name.replace("".join(out_f.suffixes), "")
    out_dir = out_f.parent
    log_stream = args.log

    target_spacing = (args.spacing,) * 3

    # Wrap the entire test loop in a try...except statement to save out a failure
    # indicator file.
    try:
        # Set up inference model.
        weights_f = Path(args.weights)
        # Log model & test run parameters.
        log_stream.write(f"model weights file: {weights_f}\n")
        log_stream.write(f"Test Parameters\n{str(pprint.pformat(p.to_dict()))}\n")

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
        log_stream.write(f"encoder layers: \n{str(encoder)}\n")
        log_stream.write(f"decoder layers: \n{str(decoder)}\n")
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            print(f"{log_prefix()}| Starting inference", flush=True)
            if "cuda" in device.type:
                torch.cuda.empty_cache()

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
                x_fov_bb = (
                    sample_dict["lr_fov_coords"].unsqueeze(0).to(x_affine_vox2real)
                )
                x_coords = einops.rearrange(
                    sample_dict["lr_real_coords"].unsqueeze(0),
                    "b coord x y z -> b x y z coord",
                ).to(device)

                # x is the LR real space, FR is the native-resolution real space, and
                # y is the super-resolution real space.
                x2y_scale_factor = (
                    np.array(target_spacing)
                    / np.array(sample_dict["lr_spacing"]).flatten()
                )
                (
                    y_fov_bb,
                    y_affine_vox2real,
                ) = pitn.transforms.functional.scale_fov_spacing(
                    x_fov_bb.squeeze(0).to(torch.float64),
                    affine_vox2real=x_affine_vox2real.squeeze(0).to(torch.float64),
                    spacing_scale_factors=x2y_scale_factor.tolist(),
                    set_affine_orig_to_fov_orig=True,
                    new_fov_align_direction="interior",
                )
                y_coords = (
                    pitn.transforms.functional.fov_coord_grid(
                        y_fov_bb.cpu(), y_affine_vox2real.cpu()
                    )
                    .cpu()
                    .unsqueeze(0)
                )
                y_spacing = (
                    torch.Tensor(target_spacing).unsqueeze(0).to(y_affine_vox2real)
                )
                # Upsample the FR mask to the y super resolution, for later masking.
                fr_mask = sample_dict["brain_mask"].unsqueeze(0).bool().to(device)
                fr_affine = sample_dict["affine_vox2real"].unsqueeze(0).to(device)
                y_mask = pitn.affine.sample_vol(
                    fr_mask.cpu(),
                    y_coords.cpu(),
                    affine_vox2mm=fr_affine.cpu(),
                    mode="nearest",
                    align_corners=True,
                )
                y_mask = y_mask.to(fr_mask)

                log_stream.write(f"Input spacing (mm) {sample_dict['lr_spacing']}\n")
                log_stream.write(f"Target spacing (mm) {target_spacing}\n")

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
                y_mask = y_mask.cpu()
                y_coords = y_coords.cpu().to(x_coords.dtype)
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

                print(
                    f"{log_prefix()}| Starting super-res prediction {subj_id}",
                    flush=True,
                )
                pred_odf = monai.inferers.sliding_window_inference(
                    y_slide_window.cpu(),
                    roi_size=(80, 80, 80),
                    sw_batch_size=y_coords.shape[0],
                    predictor=lambda q: decoder(
                        # Rearrange back into coord-last format.
                        query_real_coords=fn_coordify(q[:, :-1]).to(
                            device, non_blocking=True
                        ),
                        query_coords_mask=fn_coordify(q[:, -1:].bool()).to(
                            device, non_blocking=True
                        ),
                        context_v=ctx_v,
                        context_real_coords=x_coords,
                        affine_context_vox2real=x_affine_vox2real,
                        context_spacing=x_spacing,
                        query_spacing=y_spacing,
                    ).cpu(),
                    overlap=0,
                    padding_mode="replicate",
                ).cpu()
                print(f"{log_prefix()}| Finished super-res prediction", flush=True)

                # Mask prediction.
                pred_odf = pred_odf * y_mask
                # Pad/align prediction volume with the full input volume shape. Should
                # not involve any interpolation, only padding/cropping.
                pred_odf = einops.rearrange(
                    pred_odf[0].cpu().numpy().astype(np.float32),
                    "coeff x y z -> x y z coeff",
                )
                print(
                    f"{log_prefix()}| Output vox size {pred_odf.shape[:-1]} {subj_id}",
                    flush=True,
                )
                out_affine = y_affine_vox2real.squeeze(0).cpu().numpy()
                out_im = nib.Nifti1Image(pred_odf, affine=out_affine)
                print(
                    f"{log_prefix()}| Saving prediction subj {subj_id} to {out_f}",
                    flush=True,
                )
                nib.save(out_im, out_f)

                print(f"{log_prefix()}| Finished subj {subj_id}", flush=True)

    except KeyboardInterrupt as e:
        (out_dir / "STOPPED").touch()
        raise e
    except Exception as e:
        (out_dir / "FAILED").touch()
        raise e
    finally:
        log_stream.close()
