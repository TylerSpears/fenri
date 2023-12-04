#!/usr/bin/env python
# -*- coding: utf-8 -*-
# imports
import argparse
import collections
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

# visualization libraries
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

# Data management libraries.
import nibabel as nib
import nibabel.processing

# Computation & ML libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from box import Box
from icecream import ic
from natsort import natsorted

import pitn

N_THREADS_MRTRIX = 17


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


def fork_rng(rng: torch.Generator) -> torch.Generator:
    rng_fork = torch.Generator(device=rng.device)
    rng_fork.set_state(rng.get_state())
    return rng_fork


def fit_odf_msmt_mrtrix(
    subj_id: str,
    dwi_f: Path,
    grad_table_f: Path,
    fivett_seg_f: Path,
    brain_mask_f: Path,
    subj_extra_out_dir: Path,
    tmp_dir: Path,
):
    subj_extra_out_dir.mkdir(parents=True, exist_ok=True)
    mrtrix_tmp_dir = tmp_dir / "mrtrix"
    mrtrix_tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_dwi_f = tmp_dir / "dwi.mif"
    tmp_brain_mask_f = tmp_dir / "brain_mask.nii.gz"
    tmp_fivett_f = tmp_dir / "fivett.nii.gz"
    wm_response_f = Path(subj_extra_out_dir) / "wm_response.txt"
    gm_response_f = Path(subj_extra_out_dir) / "gm_response.txt"
    csf_response_f = Path(subj_extra_out_dir) / "csf_response.txt"
    wm_odf_f = Path(subj_extra_out_dir) / f"{subj_id}_lr_wm_msmt_csd_odf.nii.gz"
    gm_odf_f = Path(subj_extra_out_dir) / f"{subj_id}_lr_gm_msmt_csd_odf.nii.gz"
    csf_odf_f = Path(subj_extra_out_dir) / f"{subj_id}_lr_csf_msmt_csd_odf.nii.gz"

    cmd = rf"""set -eou pipefail
    export MRTRIX_NTHREADS={N_THREADS_MRTRIX}
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$MRTRIX_NTHREADS
    export MRTRIX_TMPFILE_DIR="{mrtrix_tmp_dir}"

    mrconvert -quiet "{dwi_f}" -grad "{grad_table_f}" "{tmp_dwi_f}"
    mrgrid -force -quiet \
        "{brain_mask_f}" \
        regrid \
        -template "{tmp_dwi_f}" \
        -interp nearest \
        -fill 0.0 \
        "{tmp_brain_mask_f}"
    mrgrid -force -quiet \
        "{fivett_seg_f}" \
        regrid \
        -template "{tmp_dwi_f}" \
        -interp nearest \
        -fill 0.0 \
        "{tmp_fivett_f}"

    dwi2response msmt_5tt -force \
        -wm_algo tournier \
        -mask "{tmp_brain_mask_f}" \
        "{tmp_dwi_f}" \
        "{tmp_fivett_f}" \
        "{wm_response_f}" \
        "{gm_response_f}" \
        "{csf_response_f}"
    dwi2fod msmt_csd -force -info \
        "{tmp_dwi_f}" \
        -mask "{tmp_brain_mask_f}" \
        "{wm_response_f}" "{wm_odf_f}" \
        "{gm_response_f}" "{gm_odf_f}" \
        "{csf_response_f}" "{csf_odf_f}" \
        -niter 50 -lmax 8,0,0"""

    popen_args = ["/usr/bin/bash", "-c", cmd]
    return_status = pitn.utils.proc_runner.call_shell_exec(
        cmd=cmd,
        args="",
        cwd=os.getcwd(),
        env=os.environ,
        popen_args_override=popen_args,
    )

    return return_status


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
        description="Test msmt-csd odf fitting + trilinear SH upsampling"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=("hcp", "ismrm-sim"),
        help="Testing dataset choice",
    )
    parser.add_argument(
        "-i",
        "--input_subj_ids",
        type=Path,
        required=True,
        help=".txt file with subj ids to process",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Config file for testing params; can be JSON, YAML, or TOML",
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
        default=Path("/data/srv/outputs/pitn/results/runs/trilinear"),
        type=Path,
        help="Output directory for all estimated ODF volumes",
    )
    parser.add_argument(
        "-s",
        "--create_output_subdir",
        type=int,
        default=1,
        help="Create subdirectory in output directory (default yes)",
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

    # keep device as the cpu
    device = torch.device("cpu")
    print(device)

    rng = fork_rng(torch.default_generator)

    # Experiment defaults, can be overridden in a config file
    p = Box(default_box=True, default_box_none_transform=False)
    p.experiment_name = "trilinear_resample-sh"
    # p.preproc_loaded = dict(S0_noise_b0_quantile=0.99, patch_sampling_w_erosion=17)
    # p.baseline_lr_spacing_scale = 1.6
    # p.baseline_snr = 30
    # p.test.rng_seed = 3967417599011123030
    # p.test.vol_tf = dict(
    #     downsample_factor_range=(
    #         p.baseline_lr_spacing_scale,
    #         p.baseline_lr_spacing_scale,
    #     ),
    #     noise_snr_range=(p.baseline_snr, p.baseline_snr),
    #     prefilter_sigma_scale_coeff=2.0,
    #     # Manually crop each side by 1 voxel to avoid NaNs in the LR resampling.
    #     manual_crop_lr_sides=((1, 1), (1, 1), (1, 1)),
    # )
    # Load test parameters from the config file.
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
    ic(p.to_dict())

    # Load data
    data_root_dir = Path(args.data_root_dir)
    assert data_root_dir.exists()
    # Set paths relative to the subj id root dir for each required image/file.
    if args.dataset_name == "hcp":
        rel_dwi_path = Path("ras/diffusion/dwi_norm.nii.gz")
        rel_grad_table_path = Path("ras/diffusion/ras_grad_mrtrix.b")
        rel_odf_path = Path("ras/odf/wm_msmt_csd_norm_odf.nii.gz")
        rel_fivett_seg_path = Path(
            "ras/segmentation/fivett_dwi-space_segmentation.nii.gz"
        )
        rel_brain_mask_path = Path("ras/brain_mask.nii.gz")
    elif args.dataset_name == "ismrm-sim":
        rel_dwi_path = Path("processed/diffusion/dwi_norm.nii.gz")
        rel_grad_table_path = Path("processed/diffusion/grad_mrtrix.b")
        rel_odf_path = Path("processed/odf/wm_norm_msmt_csd_fod.nii.gz")
        rel_fivett_seg_path = Path("processed/segmentation/fivett_segmentation.nii.gz")
        rel_brain_mask_path = Path("processed/brain_mask.nii.gz")
    else:
        raise RuntimeError("ERROR: No valid dataset name given")

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
            f"{os.getpid()} : Loading subj {subj_files['subj_id']}...\n",
            end="",
            flush=True,
        )
        v = pitn.data.load_super_res_subj_sample(**subj_files)
        print(
            f"{os.getpid()} : Finished loading subj {subj_files['subj_id']}\n",
            end="",
            flush=True,
        )
        v = pitn.data.preproc.preproc_loaded_super_res_subj(v, **preproc_loaded_kwargs)
        # Perform random transforms with a set seed.
        subj_rng_seed = create_subj_rng_seed(
            int(p.test.rng_seed), subj_id=subj_files["subj_id"]
        )
        rng_device = rng.device
        v = pitn.data.preproc.preproc_super_res_sample(
            v,
            **p.test.vol_tf.to_dict(),
            rng=torch.Generator(device=rng_device).manual_seed(subj_rng_seed),
        )
        print(
            f"{os.getpid()} : Finished processing subj {subj_files['subj_id']}\n",
            end="",
            flush=True,
        )
        return v

    # Load subjs to evaluation one at a time.
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
        root_dir = data_root_dir / str(subj_id)
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

    for subj_data in map(load_and_tf_subj, test_subj_dicts):
        subj_id = subj_data["subj_id"]

        print(f"{log_prefix()} | Starting {subj_id}")
        x = subj_data["lr_dwi"]
        x_affine_vox2real = subj_data["affine_lr_vox2real"].to(x.dtype)

        # Prep downgraded volume files for ODF fitting.
        subj_extra_out_dir = out_dir / subj_id
        subj_extra_out_dir.mkdir(parents=True, exist_ok=True)
        x_dwi = x.detach().cpu().numpy()
        x_dwi = einops.rearrange(x_dwi, "grad x y z -> x y z grad")
        x_dwi_im = nib.Nifti1Image(
            x_dwi, affine=x_affine_vox2real.cpu().numpy(), dtype=np.float32
        )
        grad_table = subj_data["grad_table"]
        fivett_seg_f = test_subj_files[subj_id]["fivett_seg_f"]
        brain_mask_f = test_subj_files[subj_id]["brain_mask_f"]

        print(f"{log_prefix()}| Fitting LR DWI to ODF {subj_id}")
        with tempfile.TemporaryDirectory(
            prefix=f"trilinear_dwi-up_{subj_id}_"
        ) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            dwi_f = tmp_dir / "dwi.nii.gz"
            nib.save(x_dwi_im, dwi_f)
            grad_table_f = tmp_dir / "grad_mrtrix.b"
            np.savetxt(grad_table_f, grad_table, fmt="%g")

            fit_odf_msmt_mrtrix(
                subj_id=subj_id,
                dwi_f=dwi_f,
                grad_table_f=grad_table_f,
                fivett_seg_f=fivett_seg_f,
                brain_mask_f=brain_mask_f,
                subj_extra_out_dir=subj_extra_out_dir,
                tmp_dir=tmp_dir,
            )

            lr_wm_odf_f = subj_extra_out_dir / f"{subj_id}_lr_wm_msmt_csd_odf.nii.gz"
            x_sh_coeff_im = nib.load(lr_wm_odf_f)
            x_sh_coeff = torch.from_numpy(x_sh_coeff_im.get_fdata().astype(np.float32))
            x_sh_coeff = einops.rearrange(
                x_sh_coeff.to(device), "x y z grad -> grad x y z"
            )

        y_coords = einops.rearrange(
            subj_data["full_res_real_coords"], "coord x y z -> x y z coord"
        )

        y_sh_pred = pitn.affine.sample_vol(
            x_sh_coeff,
            y_coords,
            affine_vox2mm=x_affine_vox2real,
            mode="trilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Save the predicted SH coeffs, padded to match the ground truth shape.
        out_wm_odf_f = out_dir / f"{subj_id}_trilin-sh-upsample_wm_msmt_csd_odf.nii.gz"
        affine_pred = subj_data["affine_vox2real"].detach().cpu().numpy()

        with tempfile.TemporaryDirectory(
            prefix=f"trilinear_dwi-up_{subj_id}_"
        ) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            tmp_odf_f = tmp_dir / "tmp_wm_odf.nii"
            sh_pred_im = nib.Nifti1Image(
                y_sh_pred.cpu().movedim(0, -1).numpy(), affine_pred, dtype=np.float32
            )
            nib.save(sh_pred_im, tmp_odf_f)
            template_f = test_subj_files[subj_id]["brain_mask_f"]
            pad_vol_to_template_mrtrix(
                tmp_odf_f, template_f=template_f, output_f=out_wm_odf_f
            )

        print(f"{log_prefix()} | Finished {subj_id}")
