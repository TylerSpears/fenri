#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Automatically re-import project-specific modules.
# imports
import collections
import concurrent.futures
import io
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from pprint import pprint as ppr
from typing import List, Optional, Sequence, Tuple

import dipy
import dipy.denoise
import dipy.denoise.gibbs
import dipy.denoise.localpca
import dipy.segment.mask
import dotenv
import matplotlib as mpl
import matplotlib.pyplot as plt

# Data management libraries.
import nibabel as nib

# Computation & ML libraries.
import numpy as np
import pandas as pd
import skimage
from box import Box

import docker
import pitn
from pitn.data.preproc.dwi import crop_nib_by_mask

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)


# ## Data Preprocessing


def vcu_dwi_preproc(
    subj_id: str,
    output_dir: Path,
    ap_dwi: Path,
    ap_bval: Path,
    ap_bvec: Path,
    pa_dwi: Path,
    pa_bval: Path,
    pa_bvec: Path,
    ap_json_header_dict: dict,
    pa_json_header_dict: dict,
    t1w: Path,
    eddy_seed: int,
    shared_resources=None,
) -> dict:

    t = time.time()
    t0 = t
    print(f"{subj_id} Start time {t}", flush=True)
    step_num = 0
    # Organize inputs
    # Set some fields to None as placeholders for later.
    subj_input = Box(
        ap=dict(
            dwi_f=ap_dwi,
            bval_f=ap_bval,
            bvec_f=ap_bvec,
            sidecar=ap_json_header_dict,
            dwi=None,
            bval=None,
            bvec=None,
        ),
        pa=dict(
            dwi_f=pa_dwi,
            bval_f=pa_bval,
            bvec_f=pa_bvec,
            sidecar=pa_json_header_dict,
            dwi=None,
            bval=None,
            bvec=None,
        ),
        t1w=t1w,
        eddy_seed=eddy_seed,
        output_path=Path(output_dir).resolve(),
    )
    subj_input.output_path.mkdir(parents=True, exist_ok=True)
    for pe_direct in ("ap", "pa"):
        subj_input[pe_direct].dwi = nib.load(subj_input[pe_direct].dwi_f)
        subj_input[pe_direct].bval = np.loadtxt(subj_input[pe_direct].bval_f)
        subj_input[pe_direct].bvec = np.loadtxt(subj_input[pe_direct].bvec_f)

    # Keep track of all output files for final output.
    subj_output = Box(default_box=True)
    subj_output.tmp_dir = subj_input.output_path / "tmp"
    subj_output.tmp_dir.mkdir(exist_ok=True, parents=True)

    # ###### 0.5 Create highly inclusive initial DWI mask.
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 5
    uncorrected_dwi_mask_path = (
        subj_input.output_path / f"{step_prefix}_initial_dwi_mask"
    )
    uncorrected_dwi_mask_path.mkdir(exist_ok=True, parents=True)
    docker_img = "mrtrix3/mrtrix3:3.0.3"

    uncorrected_dwi_mask_f = uncorrected_dwi_mask_path / "uncorrected-dwi_mask.nii.gz"
    rerun = pitn.utils.rerun_indicator_from_mtime(
        input_files=[
            subj_input.ap.dwi_f,
            subj_input.ap.bval_f,
            subj_input.ap.bvec_f,
            subj_input.pa.dwi_f,
            subj_input.pa.bval_f,
            subj_input.pa.bvec_f,
        ],
        output_files=[uncorrected_dwi_mask_f],
    )
    if rerun:

        # Convert AP and PA scans to .mif files
        dwi_ap_mif_f = subj_output.tmp_dir / "ap_dwi.mif"
        create_ap_mif_script = pitn.mrtrix.mr_convert_cmd(
            subj_input.ap.dwi_f,
            dwi_ap_mif_f,
            fslgrad=(subj_input.ap.bvec_f, subj_input.ap.bval_f),
            nthreads=n_procs,
            force=True,
        )
        dwi_pa_mif_f = subj_output.tmp_dir / "pa_dwi.mif"
        create_pa_mif_script = pitn.mrtrix.mr_convert_cmd(
            subj_input.pa.dwi_f,
            dwi_pa_mif_f,
            fslgrad=(subj_input.pa.bvec_f, subj_input.pa.bval_f),
            nthreads=n_procs,
            force=True,
        )
        # Calculate masks for each AP and PA batch.
        mask_ap_f = subj_output.tmp_dir / "ap_mask.nii.gz"
        mask_ap_script = pitn.mrtrix.dwi2mask_cmd(
            dwi_ap_mif_f, mask_ap_f, clean_scale=2, nthreads=n_procs, force=True
        )
        mask_pa_f = subj_output.tmp_dir / "pa_mask.nii.gz"
        mask_pa_script = pitn.mrtrix.dwi2mask_cmd(
            dwi_pa_mif_f, mask_pa_f, clean_scale=2, nthreads=n_procs, force=True
        )
        # Merge AP-PA masks by hand.
        # Binary open (4) -> binary dilate (2)
        dilate_1_script = pitn.mrtrix.mask_filter_cmd(
            uncorrected_dwi_mask_f,
            "dilate",
            uncorrected_dwi_mask_f,
            npass=4,
            force=True,
        )
        erode_1_script = pitn.mrtrix.mask_filter_cmd(
            uncorrected_dwi_mask_f,
            "erode",
            uncorrected_dwi_mask_f,
            npass=4,
            force=True,
        )
        dilate_2_script = pitn.mrtrix.mask_filter_cmd(
            uncorrected_dwi_mask_f,
            "dilate",
            uncorrected_dwi_mask_f,
            npass=2,
            force=True,
        )
        keep_largest_cc_script = pitn.mrtrix.mask_filter_cmd(
            uncorrected_dwi_mask_f,
            "connect",
            uncorrected_dwi_mask_f,
            largest=True,
            force=True,
        )

        vols = pitn.utils.union_parent_dirs(
            subj_input.ap.dwi_f,
            subj_input.ap.bval_f,
            subj_input.ap.bvec_f,
            subj_input.pa.dwi_f,
            subj_input.pa.bval_f,
            subj_input.pa.bvec_f,
            dwi_ap_mif_f,
            dwi_pa_mif_f,
            mask_ap_f,
            mask_pa_f,
            uncorrected_dwi_mask_f,
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)

        # Run commands.
        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: shared_resources["cpus"].value >= n_procs
                )
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value - n_procs
                )
        # Run all scripts in order.
        # Create .mif files for masking, mask AP and PA individually.
        for script in (
            create_ap_mif_script,
            create_pa_mif_script,
            mask_ap_script,
            mask_pa_script,
        ):
            result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=script, run_config=docker_config
            )
        # Merge AP-PA masks.
        mask_ap_nib = nib.as_closest_canonical(nib.load(mask_ap_f))
        mask_ap = mask_ap_nib.get_fdata().astype(bool)
        mask_pa = nib.as_closest_canonical(nib.load(mask_pa_f)).get_fdata().astype(bool)
        uncorrected_dwi_mask = mask_ap | mask_pa
        # Fill back in bottom voxels, if some are absent from the mask.
        mask_presence_z = np.amax(uncorrected_dwi_mask, axis=(0, 1))
        if (mask_presence_z[:5] < 1).any():
            min_slice_present_idx = np.argmax(mask_presence_z)
            for to_fill_idx in range(min_slice_present_idx):
                uncorrected_dwi_mask[..., to_fill_idx] = uncorrected_dwi_mask[
                    ..., min_slice_present_idx
                ]

        nib.save(
            nib.Nifti1Image(
                uncorrected_dwi_mask.astype(np.uint8),
                mask_ap_nib.affine,
                mask_ap_nib.header,
            ),
            uncorrected_dwi_mask_f,
        )

        # Binary closing -> binary dilation
        for script in (
            dilate_1_script,
            erode_1_script,
            dilate_2_script,
            keep_largest_cc_script,
        ):
            result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=script, run_config=docker_config
            )

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value + n_procs
                )
                shared_resources["condition"].notify_all()

        dwi_ap_mif_f.unlink(missing_ok=True)
        dwi_pa_mif_f.unlink(missing_ok=True)
        mask_ap_f.unlink(missing_ok=True)
        mask_pa_f.unlink(missing_ok=True)
        del dwi_ap_mif_f, dwi_pa_mif_f, mask_ap_f, mask_pa_f, uncorrected_dwi_mask

    subj_output.uncorrected_dwi_mask.mask_f = uncorrected_dwi_mask_f

    t1 = time.time()
    print(
        f"{subj_id} Time taken for initial DWI mask: {t1 - t}",
        flush=True,
    )
    t = t1

    # ###### 1. Denoise with MP-PCA.
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    if shared_resources is not None:
        n_procs = min(10, shared_resources["MAX_CPUS"])
    else:
        n_procs = 1
    mppca_out_path = subj_input.output_path / f"{step_prefix}_denoise_mppca"
    mppca_out_path.mkdir(exist_ok=True, parents=True)
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        # Check if computation should be redone.
        rough_mask_f = subj_output.uncorrected_dwi_mask.mask_f
        # Placeholder for PE-specific mask.
        pe_direct_mask_f = mppca_out_path / f"{pe_direct}_init_dwi_mask.nii.gz"

        # Define output files.
        denoise_dwi_f = (
            mppca_out_path / f"{subj_id}_{pe_direct}_mppca-denoise_dwi.nii.gz"
        )
        denoise_std_f = (
            mppca_out_path
            / f"{subj_id}_{pe_direct}_mppca-denoise_dwi-std-estimate.nii.gz"
        )
        rerun = pitn.utils.rerun_indicator_from_mtime(
            input_files=[subj_pe.dwi_f, subj_pe.bvec_f, subj_pe.bval_f, rough_mask_f],
            output_files=[denoise_dwi_f, denoise_std_f, pe_direct_mask_f],
        )
        if rerun:

            shutil.copyfile(rough_mask_f, pe_direct_mask_f)
            rough_mask_f = pe_direct_mask_f
            rough_dwi_mask = nib.load(rough_mask_f).get_fdata().astype(bool)

            # Estimate patch radius. Note that the total patch volume must be > the number
            # of gradient directions, so default params don't work for images with more than
            # 125 gradient directions.
            n_dwis = subj_pe.dwi.shape[-1]
            patch_radius = int(np.ceil(((n_dwis + 1) ** (1 / 3) - 1) / 2).astype(int))
            # Make sure patch radius is at least 2.
            patch_radius = max(2, patch_radius)

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["condition"].wait_for(
                        lambda: shared_resources["cpus"].value >= n_procs
                    )
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value - n_procs
                    )

            print(f"{subj_id} starting MP-PCA denoising PE {pe_direct}", flush=True)
            denoised_dwi, noise_std = dipy.denoise.localpca.mppca(
                arr=subj_pe.dwi.get_fdata(),
                mask=rough_dwi_mask,
                patch_radius=patch_radius,
                pca_method="eig",
                return_sigma=True,
            )

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value + n_procs
                    )
                    shared_resources["condition"].notify_all()

            nib.save(
                nib.Nifti1Image(denoised_dwi, affine=subj_pe.dwi.affine),
                str(denoise_dwi_f),
            )
            nib.save(
                nib.Nifti1Image(noise_std, affine=subj_pe.dwi.affine),
                str(denoise_std_f),
            )
            del denoised_dwi
            del noise_std

        subj_output.denoise_mppca[pe_direct].mask_f = rough_mask_f
        subj_output.denoise_mppca[pe_direct].dwi_f = denoise_dwi_f
        subj_output.denoise_mppca[pe_direct].std_f = denoise_std_f
        t1 = time.time()
        print(f"{subj_id} Time taken for MP-PCA {pe_direct}: {t1 - t}", flush=True)
        t = t1

    # ###### 2. Remove Gibbs ringing artifacts.
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 4
    gibbs_out_path = subj_input.output_path / f"{step_prefix}_gibbs_remove"
    gibbs_out_path.mkdir(exist_ok=True, parents=True)
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        # Check if computation should be redone.
        # Define output files.
        gibbs_corrected_dwi_f = (
            gibbs_out_path / f"{subj_id}_{pe_direct}_gibbs-correct_dwi.nii.gz"
        )
        rerun = pitn.utils.rerun_indicator_from_mtime(
            input_files=[
                subj_output.denoise_mppca[pe_direct].dwi_f,
                subj_output.denoise_mppca[pe_direct].mask_f,
                subj_pe.bvec_f,
                subj_pe.bval_f,
            ],
            output_files=[gibbs_corrected_dwi_f],
        )
        if rerun:
            dwi = nib.load(subj_output.denoise_mppca[pe_direct].dwi_f)

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["condition"].wait_for(
                        lambda: shared_resources["cpus"].value >= n_procs
                    )
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value - n_procs
                    )

            dwi_data = dwi.get_fdata()
            dwi_degibbsed = dipy.denoise.gibbs.gibbs_removal(
                dwi_data,
                slice_axis=2,
                n_points=3,
                inplace=False,
                num_processes=n_procs,
            )
            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value + n_procs
                    )
                    shared_resources["condition"].notify_all()

            mask = nib.load(subj_output.denoise_mppca[pe_direct].mask_f).get_fdata()
            dwi_degibbsed = dwi_degibbsed * mask[..., None].astype(np.uint8)
            nib.save(
                nib.Nifti1Image(dwi_degibbsed, dwi.affine, header=dwi.header),
                gibbs_corrected_dwi_f,
            )
            del dwi_degibbsed, dwi_data

        subj_output.gibbs_remove[pe_direct].dwi_f = gibbs_corrected_dwi_f
        t1 = time.time()
        print(
            f"{subj_id} Time taken for Gibbs removal {pe_direct}: {t1 - t}", flush=True
        )
        t = t1

    # ###### 3. Remove $B_1$ magnetic field bias.
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 4
    b1_debias_out_path = subj_input.output_path / f"{step_prefix}_b1_debias"
    b1_debias_out_path.mkdir(exist_ok=True, parents=True)
    docker_img = "mrtrix3/mrtrix3:3.0.3"
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        # Check if computation should be redone.
        # Define output files.
        b1_debias_dwi_f = (
            b1_debias_out_path / f"{subj_id}_{pe_direct}_b1_debias_dwi.nii.gz"
        )
        b1_debias_bias_field_f = (
            b1_debias_out_path / f"{subj_id}_{pe_direct}_b1_bias_field.nii.gz"
        )
        rerun = pitn.utils.rerun_indicator_from_mtime(
            input_files=[
                subj_output.gibbs_remove[pe_direct].dwi_f,
                subj_output.denoise_mppca[pe_direct].mask_f,
                subj_pe.bvec_f,
                subj_pe.bval_f,
            ],
            output_files=[b1_debias_dwi_f, b1_debias_bias_field_f],
        )
        if rerun:
            dwi_f = subj_output.gibbs_remove[pe_direct].dwi_f
            mask_f = subj_output.denoise_mppca[pe_direct].mask_f
            bval_f = subj_pe.bval_f
            bvec_f = subj_pe.bvec_f
            vols = pitn.utils.union_parent_dirs(
                b1_debias_dwi_f, b1_debias_bias_field_f, bval_f, bvec_f, dwi_f, mask_f
            )
            vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
            docker_config = dict(volumes=vols)

            dwi_debias_script = pitn.mrtrix.dwi_bias_correct_cmd(
                "ants",
                dwi_f,
                output=b1_debias_dwi_f,
                bias=b1_debias_bias_field_f,
                fslgrad=(bvec_f, bval_f),
                mask=mask_f,
                nthreads=n_procs,
                force=True,
            )

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["condition"].wait_for(
                        lambda: shared_resources["cpus"].value >= n_procs
                    )
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value - n_procs
                    )

            dwi_debias_result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=dwi_debias_script, run_config=docker_config
            )

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value + n_procs
                    )
                    shared_resources["condition"].notify_all()

            # Make sure to re-mask the de-biased output, can drastically reduce
            # file size (up to 2x).
            mask = nib.load(mask_f).get_fdata()
            dwi_debiased = nib.load(b1_debias_dwi_f)
            dwi_debiased_data = dwi_debiased.get_fdata() * mask[..., None].astype(
                np.uint8
            )
            nib.save(
                nib.Nifti1Image(
                    dwi_debiased_data, dwi_debiased.affine, header=dwi_debiased.header
                ),
                b1_debias_dwi_f,
            )
            del dwi_debiased_data, dwi_debiased

        subj_output.b1_debias[pe_direct].dwi_f = b1_debias_dwi_f
        subj_output.b1_debias[pe_direct].bias_field_f = b1_debias_bias_field_f
        t1 = time.time()
        print(
            f"{subj_id} Time taken for B1 field bias removal {pe_direct}: {t1 - t}",
            flush=True,
        )
        t = t1

    # ###### 4. Check bvec orientations
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 4
    bvec_flip_correct_path = subj_input.output_path / f"{step_prefix}_bvec_flip_correct"
    bvec_flip_correct_path.mkdir(exist_ok=True, parents=True)
    tmp_d = bvec_flip_correct_path / "tmp"
    tmp_d.mkdir(exist_ok=True, parents=True)

    docker_img = "dsistudio/dsistudio:chen-2022-08-18"
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        tmp_d_pe = tmp_d / pe_direct
        tmp_d_pe.mkdir(exist_ok=True, parents=True)
        correct_bvec_file = bvec_flip_correct_path / f"{subj_id}_{pe_direct}_dwi.bvec"
        rerun = pitn.utils.rerun_indicator_from_mtime(
            input_files=[
                subj_output.denoise_mppca[pe_direct].mask_f,
                subj_output.b1_debias[pe_direct].dwi_f,
                subj_pe.bvec_f,
                subj_pe.bval_f,
            ],
            output_files=[correct_bvec_file],
        )
        if rerun:
            vols = pitn.utils.union_parent_dirs(
                bvec_flip_correct_path,
                tmp_d_pe,
            )
            vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
            docker_config = dict(volumes=vols)
            dwi = nib.load(subj_output.b1_debias[pe_direct].dwi_f)
            bval = np.loadtxt(subj_pe.bval_f)
            bvec = np.loadtxt(subj_pe.bvec_f)
            mask = (
                nib.load(subj_output.denoise_mppca[pe_direct].mask_f)
                .get_fdata()
                .astype(np.uint8)
            )
            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["condition"].wait_for(
                        lambda: shared_resources["cpus"].value >= n_procs
                    )
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value - n_procs
                    )

            correct_bvec = pitn.data.preproc.dwi.bvec_flip_correct(
                dwi_data=dwi.get_fdata(),
                dwi_affine=dwi.affine,
                bval=bval,
                bvec=bvec,
                mask=mask,
                tmp_dir=tmp_d_pe,
                docker_img=docker_img,
                docker_config=docker_config,
            )
            np.savetxt(correct_bvec_file, correct_bvec, fmt="%g")

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value = (
                        shared_resources["cpus"].value + n_procs
                    )
                    shared_resources["condition"].notify_all()
            del dwi, mask

        t1 = time.time()
        print(
            f"{subj_id} Time taken for bvec flip detection {pe_direct}: {t1 - t}",
            flush=True,
        )
        t = t1
        subj_output.bvec_flip_correct[pe_direct].bvec_f = correct_bvec_file

    # ###### 5. Run topup
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 1
    num_b0s_per_pe = 3
    b0_max = 50
    topup_img = "tylerspears/fsl-cuda10.2:6.0.5"
    # Define (at least the primary) output files.
    topup_out_path = subj_input.output_path / f"{step_prefix}_topup"
    topup_out_path.mkdir(exist_ok=True, parents=True)
    topup_input_dwi_f = topup_out_path / f"{subj_id}_ap_pa_b0_input.nii.gz"
    acqparams_f = topup_out_path / "acqparams.txt"

    # We can get the topup output files if we supply the full command parameters.
    # Most parameters were taken from configuration provided by FSL in `b02b0_1.cnf`,
    # which is supposedly optimized for topup runs on b0s when image dimensions are
    # divisible by 1 (a.k.a., all image sizes).
    topup_script, _, topup_out_files = pitn.fsl.topup_cmd_explicit_in_out_files(
        imain=topup_input_dwi_f,
        datain=acqparams_f,
        out=str(topup_out_path / f"{subj_id}_topup"),
        iout=str(topup_out_path / f"{subj_id}_topup_corrected_dwi.nii.gz"),
        fout=str(topup_out_path / f"{subj_id}_topup_displ_field.nii.gz"),
        # Resolution (knot-spacing) of warps in mm
        warpres=(20, 16, 14, 12, 10, 6, 4, 4, 4),
        # Subsampling level (a value of 2 indicates that a 2x2x2 neighbourhood is
        # collapsed to 1 voxel)
        subsamp=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        # FWHM of gaussian smoothing
        fwhm=(8, 6, 4, 3, 3, 2, 1, 0, 0),
        # Maximum number of iterations
        miter=(5, 5, 5, 5, 5, 10, 10, 20, 20),
        # Relative weight of regularisation
        lambd=(
            0.0005,
            0.0001,
            0.00001,
            0.0000015,
            0.0000005,
            0.0000005,
            0.00000005,
            0.0000000005,
            0.00000000001,
        ),
        # If set to 1 lambda is multiplied by the current average squared difference
        ssqlambda=True,
        # Regularisation model
        regmod="bending_energy",
        # If set to 1 movements are estimated along with the field
        estmov=(True, True, True, True, True, False, False, False, False),
        # 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient
        minmet=("LM", "LM", "LM", "LM", "LM", "SCG", "SCG", "SCG", "SCG"),
        # Quadratic or cubic splines
        splineorder=3,
        # Precision for calculation and storage of Hessian
        numprec="double",
        # Linear or spline interpolation
        interp="spline",
        # If set to 1 the images are individually scaled to a common mean intensity
        scale=True,
        verbose=True,
        log_stdout=True,
    )

    rerun = pitn.utils.rerun_indicator_from_mtime(
        input_files=[
            subj_output.b1_debias.ap.dwi_f,
            subj_output.b1_debias.pa.dwi_f,
            subj_output.bvec_flip_correct.ap.bvec_f,
            subj_output.bvec_flip_correct.pa.bvec_f,
            subj_input.ap.bval_f,
            subj_input.pa.bval_f,
        ],
        output_files=[
            Path(str(f))
            for f in pitn.utils.flatten(topup_out_files, as_dict=True).values()
        ]
        + [topup_input_dwi_f, acqparams_f],
    )
    if rerun:
        # Topup really only needs a few b0s in each PE direction, so use image similarity to
        # find the "least distorted" b0s in each PE direction. Then, save out to a file for
        # topup to read.
        select_dwis = dict()
        for pe_direct in ("ap", "pa"):
            subj_pe = subj_input[pe_direct]
            dwi = nib.load(subj_output.b1_debias[pe_direct].dwi_f)
            bval = np.loadtxt(subj_pe.bval_f)
            bvec = np.loadtxt(subj_output.bvec_flip_correct[pe_direct].bvec_f)

            top_b0s = pitn.data.preproc.dwi.top_k_b0s(
                dwi.get_fdata(),
                bval=bval,
                bvec=bvec,
                n_b0s=num_b0s_per_pe,
                b0_max=b0_max,
                seed=24023,
            )
            select_dwis[pe_direct] = top_b0s["dwi"]
            # Topup doesn't make use of bvec and bval, probably because it only expects to
            # operate over a handful of b0 DWIs.

        # Merge selected b0s into one file.
        # If shapes are not the same, then the previous DWI masking did not produce a
        # consistent shape between AP and PA volumes. So, pad each volume to the max
        # size of each spatial dimension.
        # These shapes and affines should be equivilent from the correction done right
        # after MP-PCA denoising.
        if select_dwis["ap"].shape != select_dwis["pa"].shape:
            raise RuntimeError(
                "ERROR: AP/PA volumes have different shapes.",
                f"Expected equivalent, got {select_dwis['ap'].shape}",
                f"and {select_dwis['pa'].shape}",
            )
        select_b0_data = [select_dwis["ap"], select_dwis["pa"]]
        select_b0_data = np.concatenate(select_b0_data, axis=-1)
        affine = nib.load(subj_output.b1_debias.ap.dwi_f).affine
        aff_pa = nib.load(subj_output.b1_debias.pa.dwi_f).affine
        assert np.isclose(affine, aff_pa).all()
        header = nib.load(subj_output.b1_debias.ap.dwi_f).header
        nib.save(
            nib.Nifti1Image(select_b0_data, affine=affine, header=header),
            topup_input_dwi_f,
        )

        # Create the acquisition parameters file.
        ap_readout_time = float(subj_input.ap.sidecar["EstimatedTotalReadoutTime"])
        ap_pe_direct = subj_input.ap.sidecar["PhaseEncodingAxis"]
        ap_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
            ap_readout_time, *([ap_pe_direct] * num_b0s_per_pe)
        )
        pa_readout_time = float(subj_input.pa.sidecar["EstimatedTotalReadoutTime"])
        pa_pe_direct = subj_input.pa.sidecar["PhaseEncodingAxis"]
        # The negation of the axis isn't present in these data, for whatever reason.
        if "-" not in pa_pe_direct:
            pa_pe_direct = f"{pa_pe_direct}-"
        pa_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
            pa_readout_time, *([pa_pe_direct] * num_b0s_per_pe)
        )
        acqparams = np.concatenate([ap_acqparams, pa_acqparams], axis=0)
        np.savetxt(acqparams_f, acqparams, fmt="%g")
        subj_output.topup.acqparams_f = acqparams_f

        # Set up docker configuration for running topup.
        vols = pitn.utils.union_parent_dirs(
            subj_output.topup.acqparams_f, topup_out_path
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)
        topup_script = pitn.utils.proc_runner.multiline_script2docker_cmd(topup_script)

        # Finally, run topup.
        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: shared_resources["cpus"].value >= n_procs
                )
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value - n_procs
                )

        topup_result = pitn.utils.proc_runner.call_docker_run(
            img=topup_img,
            cmd=topup_script,
            run_config=docker_config,
        )
        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value + n_procs
                )
                shared_resources["condition"].notify_all()

    t1 = time.time()
    print(f"{subj_id} Time taken for topup: {t1 - t}", flush=True)
    t = t1
    subj_output.topup.acqparams_f = acqparams_f
    subj_output.topup.merge_update(topup_out_files)

    # ###### 6. Extract mask of unwarped diffusion data.
    # Run BET on the topup b0s for eddy, as suggested by FSL:
    # <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--mask>
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 1
    docker_img = "tylerspears/fsl-cuda10.2:6.0.5"
    output_path = subj_input.output_path
    bet_out_path = output_path / f"{step_prefix}_bet_topup2eddy"
    bet_out_path.mkdir(exist_ok=True, parents=True)
    tmp_dir = bet_out_path / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    input_dwi_f = subj_output.topup.corrected_im
    out_mask_f = bet_out_path / f"{subj_id}_bet_topup2eddy_mask.nii.gz"

    rerun = pitn.utils.rerun_indicator_from_mtime(
        input_files=[input_dwi_f],
        output_files=[out_mask_f],
    )
    if rerun:
        # Set up docker configuration for running bet.
        vols = pitn.utils.union_parent_dirs(input_dwi_f, out_mask_f, tmp_dir)
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: shared_resources["cpus"].value >= n_procs
                )
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value - n_procs
                )

        # Fractional intensity of 0.15 and a gradient of -0.15 experimentally seems to
        # produce a decent (but still very rough) mask for eddy. If the default bet
        # parameters are left as-is, the frontal cortex will get cut off.
        topup2eddy_mask_f = pitn.data.preproc.dwi.bet_mask_median_dwis(
            input_dwi_f,
            out_mask_f=out_mask_f,
            robust_iters=True,
            fractional_intensity_threshold=0.15,
            vertical_grad_in_f=-0.2,
            tmp_dir=tmp_dir,
            docker_img=docker_img,
            docker_config=docker_config,
        )

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value + n_procs
                )
                shared_resources["condition"].notify_all()

        assert topup2eddy_mask_f.exists()

    t1 = time.time()
    print(f"{subj_id} Time taken for post-topup mask: {t1 - t}", flush=True)
    t = t1
    subj_output.bet_topup2eddy.mask_f = out_mask_f

    # ###### 7. Run eddy correction
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 2
    docker_img = "tylerspears/fsl-cuda10.2:6.0.5"
    eddy_out_path = subj_input.output_path / f"{step_prefix}_eddy"
    eddy_out_path.mkdir(exist_ok=True, parents=True)
    slspec_path = eddy_out_path / "slspec.txt"
    index_path = eddy_out_path / "index.txt"
    ap_pa_basename = f"{subj_id}_ap-pa_pre-topup_dwi"
    ap_pa_dwi_path = eddy_out_path / (ap_pa_basename + ".nii.gz")
    ap_pa_bval_path = eddy_out_path / (ap_pa_basename + ".bval")
    ap_pa_bvec_path = eddy_out_path / (ap_pa_basename + ".bvec")

    eddy_script, _, eddy_out_files = pitn.fsl.eddy_cmd_explicit_in_out_files(
        imain=ap_pa_dwi_path,
        bvecs=ap_pa_bvec_path,
        bvals=ap_pa_bval_path,
        mask=subj_output.bet_topup2eddy.mask_f,
        index=index_path,
        acqp=subj_output.topup.acqparams_f,
        slspec=slspec_path,
        topup_fieldcoef=subj_output.topup.fieldcoef,
        topup_movpar=subj_output.topup.movpar,
        out=str(eddy_out_path / f"{subj_id}_eddy"),
        niter=10,
        fwhm=[10, 8, 4, 2, 1, 0, 0, 0, 0, 0],
        repol=True,
        rep_noise=True,
        ol_type="both",
        flm="quadratic",
        slm="linear",
        mporder=8,
        s2v_niter=7,
        s2v_lambda=2,
        estimate_move_by_susceptibility=True,
        mbs_niter=10,
        mbs_lambda=10,
        initrand=subj_input.eddy_seed,
        cnr_maps=True,
        fields=True,
        dfields=True,
        very_verbose=True,
        log_stdout=True,
        use_cuda=True,
        auto_select_gpu=False,
    )

    rerun = pitn.utils.rerun_indicator_from_mtime(
        input_files=[
            subj_output.b1_debias.ap.dwi_f,
            subj_output.b1_debias.pa.dwi_f,
            subj_output.bvec_flip_correct.ap.bvec_f,
            subj_output.bvec_flip_correct.pa.bvec_f,
            subj_input.ap.dwi_f,
            subj_input.ap.bval_f,
            subj_input.pa.bval_f,
            subj_output.bet_topup2eddy.mask_f,
            subj_output.topup.acqparams_f,
            subj_output.topup.fieldcoef,
            subj_output.topup.movpar,
        ],
        output_files=list(pitn.utils.flatten(eddy_out_files).values())
        + [slspec_path, index_path, ap_pa_dwi_path, ap_pa_bval_path, ap_pa_bvec_path],
    )

    if rerun:
        # Create slspec file.
        orig_dwi_nib = nib.load(subj_input.ap.dwi_f)
        cropped_dwi_nib = nib.load(subj_output.b1_debias.ap.dwi_f)
        slspec = pitn.fsl.estimate_slspec(
            subj_input.ap.sidecar,
            n_slices=orig_dwi_nib.shape[2],
        )
        slspec = pitn.fsl.sub_select_slspec(
            slspec,
            orig_nslices=orig_dwi_nib.shape[2],
            orig_affine=orig_dwi_nib.affine,
            subselect_nslices=cropped_dwi_nib.shape[2],
            subselect_affine=cropped_dwi_nib.affine,
        )
        if np.isnan(slspec).any():
            slspec_filter = slspec.copy()
            slspec_filter[np.isnan(slspec)] = -1
            slspec_str = list()
            for row in slspec_filter:
                row_str = list()
                for val in row:
                    val = int(val)
                    if val >= 0:
                        row_str.append(str(val))
                    else:
                        row_str.append(" ")
                row_str = " ".join(row_str)
                slspec_str.append(row_str)
            slspec_str = "\n".join(slspec_str)
            with open(slspec_path, "w") as f:
                f.write(slspec_str)
        else:
            np.savetxt(slspec_path, slspec, fmt="%g")

        # Create index file that relates DWI index to the acquisition params.
        # The index file is 1-indexed, not 0-indexed.
        ap_acqp_idx = 1
        pa_acqp_idx = num_b0s_per_pe + 1
        index_acqp = np.asarray(
            [ap_acqp_idx] * nib.load(subj_output.b1_debias.ap.dwi_f).shape[-1]
            + [pa_acqp_idx] * nib.load(subj_output.b1_debias.ap.dwi_f).shape[-1]
        ).reshape(1, -1)
        np.savetxt(index_path, index_acqp, fmt="%g")

        # Merge and save both AP and PA DWIs, bvals, and bvecs together.
        # DWIs
        ap_pa_dwi = nib.Nifti1Image(
            np.concatenate(
                [
                    nib.load(subj_output.b1_debias.ap.dwi_f).get_fdata(),
                    nib.load(subj_output.b1_debias.pa.dwi_f).get_fdata(),
                ],
                axis=-1,
            ),
            affine=nib.load(subj_output.b1_debias.pa.dwi_f).affine,
            header=nib.load(subj_output.b1_debias.ap.dwi_f).header,
        )
        # If some z-axis slices were removed from cropping, we must pad them back for
        # eddy to work with slice2vol motion correction.
        # if ap_pa_dwi.shape[2] != nib.load(subj_input.ap.dwi_f).shape[2]:
        #     pass
        nib.save(ap_pa_dwi, ap_pa_dwi_path)
        # bvals
        ap_pa_bval = np.concatenate(
            [np.loadtxt(subj_input.ap.bval_f), np.loadtxt(subj_input.pa.bval_f)],
            axis=0,
        )
        np.savetxt(ap_pa_bval_path, ap_pa_bval)
        # bvecs
        ap_pa_bvec = np.concatenate(
            [np.loadtxt(subj_input.ap.bvec_f), np.loadtxt(subj_input.pa.bvec_f)],
            axis=1,
        )
        np.savetxt(ap_pa_bvec_path, ap_pa_bvec)

        # Set up docker configuration for running eddy with cuda.
        vols = pitn.utils.union_parent_dirs(
            eddy_out_path,
            subj_output.bet_topup2eddy.mask_f,
            subj_output.topup.acqparams,
            subj_output.topup.fieldcoef,
            subj_output.topup.movpar,
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(
            volumes=vols,
            runtime="nvidia",
        )
        eddy_script = pitn.utils.proc_runner.multiline_script2docker_cmd(eddy_script)

        # Finally, run eddy.
        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: (shared_resources["cpus"].value >= n_procs)
                    and (not shared_resources["gpus"].empty())
                )
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value - n_procs
                )
                gpu_idx = shared_resources["gpus"].get(timeout=5)
        else:
            gpu_idx = "0"

        eddy_result = pitn.utils.proc_runner.call_docker_run(
            img=docker_img,
            cmd=eddy_script,
            run_config=docker_config,
            env={
                "NVIDIA_VISIBLE_DEVICES": gpu_idx,
                "CUDA_VISIBLE_DEVICES": gpu_idx,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
        )

        if shared_resources is not None:
            shared_resources["gpus"].put(gpu_idx)
            with shared_resources["condition"]:
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value + n_procs
                )
                shared_resources["condition"].notify_all()

    t1 = time.time()
    print(f"{subj_id} Time taken for eddy: {t1 - t}", flush=True)
    t = t1

    subj_output.eddy.input_bval = ap_pa_bval_path
    subj_output.eddy.input_bvec = ap_pa_bvec_path
    subj_output.eddy.input_dwi = ap_pa_dwi_path
    subj_output.eddy.index = index_path
    subj_output.eddy.slspec = slspec_path
    subj_output.eddy.merge_update(eddy_out_files)

    # ###### 8. Extract final mask of diffusion data, and crop volumes.
    step_num += 1
    step_prefix = f"{step_num:g}".zfill(2)
    n_procs = 5
    docker_img = "mrtrix3/mrtrix3:3.0.3"
    eddy_out_mask_f = eddy_out_path / f"{subj_id}_eddy_mask.nii.gz"
    postproc_out_path = subj_input.output_path / f"{step_prefix}_final"
    postproc_out_path.mkdir(exist_ok=True, parents=True)
    postproc_dwi_path = postproc_out_path / f"{subj_id}_dwi.nii.gz"
    postproc_mask_path = postproc_out_path / f"{subj_id}_dwi_mask.nii.gz"
    postproc_bvec_path = postproc_out_path / f"{subj_id}.bvec"
    postproc_bval_path = postproc_out_path / f"{subj_id}.bval"

    rerun = pitn.utils.rerun_indicator_from_mtime(
        input_files=[
            subj_output.eddy.corrected,
            subj_output.eddy.input_bval,
            subj_output.eddy.rotated_bvecs,
        ],
        output_files=[
            eddy_out_mask_f,
            postproc_dwi_path,
            postproc_mask_path,
            postproc_bvec_path,
            postproc_bval_path,
        ],
    )

    if rerun:
        eddy_mask_script = pitn.mrtrix.dwi2mask_cmd(
            subj_output.eddy.corrected,
            output=eddy_out_mask_f,
            fslgrad=(subj_output.eddy.rotated_bvecs, subj_output.eddy.input_bval),
            clean_scale=2,
            nthreads=n_procs,
            force=True,
        )
        # Binary closing (2) -> binary dilate (2)
        dilate_1_script = pitn.mrtrix.mask_filter_cmd(
            eddy_out_mask_f,
            "dilate",
            eddy_out_mask_f,
            npass=2,
            force=True,
        )
        erode_1_script = pitn.mrtrix.mask_filter_cmd(
            eddy_out_mask_f,
            "erode",
            eddy_out_mask_f,
            npass=2,
            force=True,
        )
        dilate_2_script = pitn.mrtrix.mask_filter_cmd(
            eddy_out_mask_f,
            "dilate",
            eddy_out_mask_f,
            npass=2,
            force=True,
        )
        keep_largest_cc_script = pitn.mrtrix.mask_filter_cmd(
            eddy_out_mask_f,
            "connect",
            eddy_out_mask_f,
            largest=True,
            force=True,
        )
        apply_mask_dwi_script = pitn.mrtrix.mr_grid_cmd(
            subj_output.eddy.corrected,
            operation="crop",
            output=postproc_dwi_path,
            mask=eddy_out_mask_f,
            force=True,
        )
        apply_mask_mask_script = pitn.mrtrix.mr_grid_cmd(
            eddy_out_mask_f,
            operation="crop",
            output=postproc_mask_path,
            mask=eddy_out_mask_f,
            force=True,
        )

        # Generate a DWI mask from the final preprocessed DWIs.
        vols = pitn.utils.union_parent_dirs(
            eddy_out_mask_f,
            postproc_dwi_path,
            subj_output.eddy.corrected,
            subj_output.eddy.input_bval,
            subj_output.eddy.rotated_bvecs,
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: shared_resources["cpus"].value >= n_procs
                )
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value - n_procs
                )

        # Run all scripts in order.
        for script in (
            eddy_mask_script,
            dilate_1_script,
            erode_1_script,
            dilate_2_script,
            keep_largest_cc_script,
            apply_mask_dwi_script,
            apply_mask_mask_script,
        ):
            result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=script, run_config=docker_config
            )

        cropped_dwi_nib = nib.load(postproc_dwi_path)
        cropped_dwi = cropped_dwi_nib.get_fdata()
        cropped_mask = nib.load(postproc_mask_path).get_fdata().astype(bool)
        cropped_mask = cropped_mask[..., None]
        cropped_dwi = cropped_dwi * cropped_mask
        nib.save(
            nib.Nifti1Image(
                cropped_dwi.astype(np.float32),
                affine=cropped_dwi_nib.affine,
                header=cropped_dwi_nib.header,
            ),
            postproc_dwi_path,
        )
        bvals = np.loadtxt(subj_output.eddy.input_bval)
        np.savetxt(postproc_bval_path, bvals, fmt="%g")
        bvecs = np.loadtxt(subj_output.eddy.rotated_bvecs)
        np.savetxt(postproc_bvec_path, bvecs, fmt="%g")

        cropped_dwi_nib.uncache()
        del cropped_dwi, cropped_mask

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value = (
                    shared_resources["cpus"].value + n_procs
                )
                shared_resources["condition"].notify_all()

    subj_output.postproc.dwi = postproc_dwi_path
    subj_output.postproc.mask = postproc_mask_path
    subj_output.postproc.bval = postproc_bval_path
    subj_output.postproc.bvec = postproc_bvec_path

    return subj_output.to_dict()


if __name__ == "__main__":
    # Iterate over subjects and run preprocessing task.

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
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd()
    )
    # Store and format the subprocess' output.
    proc_out = proc.communicate()[0].strip().decode("utf-8")
    # Use python-dotenv to load the environment variables by using the output of
    # 'direnv exec ...' as a 'dummy' .env file.
    dotenv.load_dotenv(stream=io.StringIO(proc_out), override=True)

    # ## Directory Setup
    bids_dir = Path("/data/VCU_MS_Study/bids")
    assert bids_dir.exists()
    output_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
    assert output_dir.exists()
    selected_subjs = (
        "P_32",
        "P_38",
        "HC_01",
        "HC_02",
        "HC_03",
        "HC_04",
        "HC_05",
    )

    # ## Subject File Locations
    subj_files = Box(default_box=True)
    for subj in selected_subjs:
        p_dir = bids_dir / subj
        for encode_direct in ("AP", "PA"):
            direct_files = list(p_dir.glob(f"*DKI*{encode_direct}*_[0-9][0-9][0-9].*"))
            dwi = list(filter(lambda p: p.name.endswith(".nii.gz"), direct_files))[0]
            bval = list(filter(lambda p: p.name.endswith("bval"), direct_files))[0]
            bvec = list(filter(lambda p: p.name.endswith("bvec"), direct_files))[0]
            json_info = list(filter(lambda p: p.name.endswith(".json"), direct_files))[
                0
            ]
            subj_files[subj][encode_direct.lower()] = Box(
                dwi=dwi, bval=bval, bvec=bvec, json=json_info
            )
        # Grab structural files
        t1w = pitn.utils.system.get_file_glob_unique(
            p_dir, "*T1*_[0-9][0-9][0-9].nii.gz"
        )
        subj_files[subj].t1w = Path(t1w)

    eddy_random_seed = 42138

    processed_outputs = dict()
    with concurrent.futures.ProcessPoolExecutor(
        multiprocessing.cpu_count()
    ) as executor:
        with multiprocessing.Manager() as manager:

            resources = manager.dict(
                MAX_CPUS=multiprocessing.cpu_count(),
                condition=manager.Condition(),
                cpus=manager.Value("i", multiprocessing.cpu_count()),
                gpus=manager.Queue(2),
            )
            # resources["gpus"].put("0")
            # resources["gpus"].put("1")
            resources["gpus"].put("GPU-ed20d87f-e88e-692f-0b56-548b8a05ddea")
            resources["gpus"].put("GPU-0636ee40-2eab-9533-1be7-dbbadade95c4")
            sub_count = 0
            # Grab files and process subject data.
            for subj_id, files in subj_files.items():

                subj_out_dir = output_dir / subj_id / "diffusion" / "preproc"
                subj_out_dir.mkdir(parents=True, exist_ok=True)

                ap_files = files.ap.to_dict()
                ap_json = ap_files.pop("json")
                with open(ap_json, "r") as f:
                    ap_json_info = dict(json.load(f))
                ap_files = {
                    k: Path(str(ap_files[k])).resolve() for k in ap_files.keys()
                }

                pa_files = files.pa.to_dict()
                pa_json = pa_files.pop("json")
                with open(pa_json, "r") as f:
                    pa_json_info = dict(json.load(f))
                pa_files = {
                    k: Path(str(pa_files[k])).resolve() for k in pa_files.keys()
                }

                subj_output = executor.submit(
                    vcu_dwi_preproc,
                    subj_id=subj_id,
                    output_dir=str(subj_out_dir),
                    ap_dwi=ap_files["dwi"],
                    ap_bval=ap_files["bval"],
                    ap_bvec=ap_files["bvec"],
                    ap_json_header_dict=ap_json_info,
                    pa_dwi=pa_files["dwi"],
                    pa_bval=pa_files["bval"],
                    pa_bvec=pa_files["bvec"],
                    pa_json_header_dict=pa_json_info,
                    t1w=files.t1w,
                    eddy_seed=eddy_random_seed,
                    shared_resources=resources,
                )
                processed_outputs[subj_id] = subj_output
                # # !DEBUG
                # sub_count += 1
                # if sub_count >= 1:
                #     break
            concurrent.futures.wait([v for v in processed_outputs.values()])
            subj_outputs = dict()
            for k, v in processed_outputs.items():
                subj_outputs[k] = v.result()

    print(subj_outputs)
