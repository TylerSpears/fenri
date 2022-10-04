#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:


# Automatically re-import project-specific modules.
# imports
import collections
import concurrent.futures
import io
import json
import multiprocessing
import os
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
    eddy_seed: int,
    shared_resources=None,
) -> dict:

    t = time.time()
    t0 = t
    print(f"{subj_id} Start time {t}", flush=True)
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

    # ###### 1. Denoise with MP-PCA.
    if shared_resources is not None:
        n_procs = shared_resources["MAX_CPUS"]
    else:
        n_procs = 1
    mppca_out_path = subj_input.output_path / "denoise_mppca"
    mppca_out_path.mkdir(exist_ok=True, parents=True)
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        # Check if computation should be redone.
        # Define output files.
        rough_mask_f = (
            mppca_out_path / f"{subj_id}_{pe_direct}_rough-median-otsu-dwi-mask.nii.gz"
        )
        denoise_dwi_f = (
            mppca_out_path / f"{subj_id}_{pe_direct}_mppca-denoise_dwi.nii.gz"
        )
        denoise_std_f = (
            mppca_out_path
            / f"{subj_id}_{pe_direct}_mppca-denoise_dwi-std-estimate.nii.gz"
        )
        rerun = pitn.utils.rerun_indicator_from_mtime(
            input_files=[subj_pe.dwi_f, subj_pe.bvec_f, subj_pe.bval_f],
            output_files=[rough_mask_f, denoise_dwi_f, denoise_std_f],
        )
        if rerun:
            # Estimate a rough mask to speed up MPPCA.
            # DWIs are not registered at all, so the mask needs to be generous across all
            # gradient directions.
            # Standardize intensity across DWIs, as higher gradient strengths produce lower
            # signal intensity.
            norm_dwis = (
                subj_pe.dwi.get_fdata()
                - np.mean(subj_pe.dwi.get_fdata(), axis=(0, 1, 2), keepdims=True)
            ) / (np.std(subj_pe.dwi.get_fdata(), axis=(0, 1, 2), keepdims=True))

            _, rough_dwi_mask = dipy.segment.mask.median_otsu(
                np.mean(norm_dwis, axis=-1), median_radius=2, numpass=1, dilate=6
            )
            mask_labels, num_labels = skimage.measure.label(
                rough_dwi_mask.astype(np.uint8), return_num=True
            )
            # Keep only the largest blob if there's more than one.
            if num_labels > 1:
                properties = skimage.measure.regionprops(mask_labels)
                areas = [p.area for p in properties]
                max_area = max(areas)
                l = properties[areas.index(max_area)].label
                rough_dwi_mask = mask_labels == l
            # Crop by mask, because MP-PCA will just zero-out everything outside the
            # mask anyway.
            crop_dwi = crop_nib_by_mask(subj_pe.dwi, rough_dwi_mask)
            crop_mask_nib = crop_nib_by_mask(
                nib.Nifti1Image(rough_dwi_mask.astype(np.uint8), subj_pe.dwi.affine),
                rough_dwi_mask,
            )
            nib.save(crop_mask_nib, rough_mask_f)
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
                    shared_resources["cpus"].value -= n_procs
                    shared_resources["condition"].notify_all()

            denoised_dwi, noise_std = dipy.denoise.localpca.mppca(
                arr=crop_dwi.get_fdata(),
                mask=crop_mask_nib.get_fdata(),
                patch_radius=patch_radius,
                pca_method="eig",
                return_sigma=True,
            )

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value += n_procs
                    shared_resources["condition"].notify_all()

            nib.save(
                nib.Nifti1Image(denoised_dwi, affine=crop_dwi.affine),
                str(denoise_dwi_f),
            )
            nib.save(
                nib.Nifti1Image(noise_std, affine=crop_dwi.affine), str(denoise_std_f)
            )

        subj_output.denoise_mppca[pe_direct].mask_f = rough_mask_f
        subj_output.denoise_mppca[pe_direct].dwi_f = denoise_dwi_f
        subj_output.denoise_mppca[pe_direct].std_f = denoise_std_f
        t1 = time.time()
        print(f"{subj_id} Time taken for MP-PCA {pe_direct}: {t1 - t}", flush=True)
        t = t1

    # ###### 1.5 Re-align AP and PA volumes after mask cropping.
    ap_dwi = nib.load(subj_output.denoise_mppca.ap.dwi_f)
    pa_dwi = nib.load(subj_output.denoise_mppca.pa.dwi_f)

    if (ap_dwi.shape != pa_dwi.shape) and (
        not np.isclose(ap_dwi.affine, pa_dwi.affine).all()
    ):
        # Also re-align the mask and std fields. Each volume in a PE direction should
        # have the same corresponding shape and affine.
        ap_mask = nib.load(subj_output.denoise_mppca.ap.mask_f)
        ap_std = nib.load(subj_output.denoise_mppca.ap.std_f)
        pa_mask = nib.load(subj_output.denoise_mppca.pa.mask_f)
        pa_std = nib.load(subj_output.denoise_mppca.pa.std_f)

        target_shape = np.maximum(
            np.asarray(ap_dwi.shape[:-1]),
            np.asarray(pa_dwi.shape[:-1]),
        )
        ap_reshaped_dwi = pitn.data.preproc.dwi.crop_or_pad_nib(ap_dwi, target_shape)
        ap_reshaped_mask = pitn.data.preproc.dwi.crop_or_pad_nib(ap_mask, target_shape)
        ap_reshaped_mask.set_data_dtype(np.uint8)
        ap_reshaped_std = pitn.data.preproc.dwi.crop_or_pad_nib(ap_std, target_shape)
        pa_reshaped_dwi = pitn.data.preproc.dwi.crop_or_pad_nib(pa_dwi, target_shape)
        pa_reshaped_mask = pitn.data.preproc.dwi.crop_or_pad_nib(pa_mask, target_shape)
        pa_reshaped_std = pitn.data.preproc.dwi.crop_or_pad_nib(pa_std, target_shape)
        if not np.isclose(ap_reshaped_dwi.affine, pa_reshaped_dwi.affine).all():
            # Arbitrarily choose AP as the reference volume.
            ref_aff = ap_reshaped_dwi.affine
            tf_pa2ap = pitn.data.preproc.dwi.transform_translate_nib_fov_to_target(
                pa_reshaped_dwi, target_affine=ref_aff
            )
            pa_reshaped_dwi = pitn.data.preproc.dwi.apply_torchio_tf_to_nib(
                tf_pa2ap, pa_reshaped_dwi
            )
            assert np.isclose(pa_reshaped_dwi.affine, ap_reshaped_dwi.affine).all()
            pa_reshaped_mask = pitn.data.preproc.dwi.apply_torchio_tf_to_nib(
                tf_pa2ap, pa_reshaped_mask
            )
            assert np.isclose(pa_reshaped_mask.affine, ap_reshaped_mask.affine).all()
            pa_reshaped_mask.set_data_dtype(np.uint8)
            pa_reshaped_std = pitn.data.preproc.dwi.apply_torchio_tf_to_nib(
                tf_pa2ap, pa_reshaped_std
            )
            assert np.isclose(pa_reshaped_std.affine, ap_reshaped_dwi.affine).all()

        # Save out new nib volumes into MPPCA output location.
        nib.save(ap_reshaped_dwi, subj_output.denoise_mppca.ap.dwi_f)
        nib.save(ap_reshaped_mask, subj_output.denoise_mppca.ap.mask_f)
        nib.save(ap_reshaped_std, subj_output.denoise_mppca.ap.std_f)
        nib.save(pa_reshaped_dwi, subj_output.denoise_mppca.pa.dwi_f)
        nib.save(pa_reshaped_mask, subj_output.denoise_mppca.pa.mask_f)
        nib.save(pa_reshaped_std, subj_output.denoise_mppca.pa.std_f)
        for im in (
            ap_reshaped_dwi,
            ap_reshaped_mask,
            ap_reshaped_std,
            pa_reshaped_dwi,
            pa_reshaped_mask,
            pa_reshaped_std,
        ):
            im.uncache()

    # If both shapes and affines match, then there's no need for correction.
    elif (ap_dwi.shape == pa_dwi.shape) and (
        np.isclose(ap_dwi.affine, pa_dwi.affine).all()
    ):
        pass
    else:
        raise RuntimeError(
            "ERROR: AP/PA shape-affine mismatch.",
            "Expected either 1) both shapes and affines are different,",
            "or 2) both shapes and affines are the same;",
            f"got\nAP > {ap_dwi.shape}, {ap_dwi.affine}\n",
            f"PA > {pa_dwi.shape}, {pa_dwi.affine}",
        )

    # ###### 2. Remove Gibbs ringing artifacts.
    n_procs = 4
    gibbs_out_path = subj_input.output_path / "gibbs_remove"
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
                    shared_resources["cpus"].value -= n_procs
                    shared_resources["condition"].notify_all()

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
                    shared_resources["cpus"].value += n_procs
                    shared_resources["condition"].notify_all()

            mask = nib.load(subj_output.denoise_mppca[pe_direct].mask_f).get_fdata()
            dwi_degibbsed = dwi_degibbsed * mask[..., None].astype(np.uint8)
            nib.save(
                nib.Nifti1Image(dwi_degibbsed, dwi.affine, header=dwi.header),
                gibbs_corrected_dwi_f,
            )

        subj_output.gibbs_remove[pe_direct].dwi_f = gibbs_corrected_dwi_f
        t1 = time.time()
        print(
            f"{subj_id} Time taken for Gibbs removal {pe_direct}: {t1 - t}", flush=True
        )
        t = t1

    # ###### 3. Remove $B_1$ magnetic field bias.
    n_procs = 4
    b1_debias_out_path = subj_input.output_path / "b1_debias"
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
                    shared_resources["cpus"].value -= n_procs
                    shared_resources["condition"].notify_all()

            dwi_debias_result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=dwi_debias_script, run_config=docker_config
            )

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value += n_procs
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

        subj_output.b1_debias[pe_direct].dwi_f = b1_debias_dwi_f
        subj_output.b1_debias[pe_direct].bias_field_f = b1_debias_bias_field_f
        t1 = time.time()
        print(
            f"{subj_id} Time taken for B1 field bias removal {pe_direct}: {t1 - t}",
            flush=True,
        )
        t = t1

    # ###### 4. Check bvec orientations
    n_procs = 4
    bvec_flip_correct_path = subj_input.output_path / "bvec_flip_correct"
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
                    shared_resources["cpus"].value -= n_procs
                    shared_resources["condition"].notify_all()

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

            if shared_resources is not None:
                with shared_resources["condition"]:
                    shared_resources["cpus"].value += n_procs
                    shared_resources["condition"].notify_all()

            np.savetxt(correct_bvec_file, correct_bvec, fmt="%g")
        t1 = time.time()
        print(
            f"{subj_id} Time taken for bvec flip detection {pe_direct}: {t1 - t}",
            flush=True,
        )
        t = t1
        subj_output.bvec_flip_correct[pe_direct].bvec_f = correct_bvec_file

    # ###### 5. Run topup
    n_procs = 1
    num_b0s_per_pe = 3
    b0_max = 50
    topup_img = "tylerspears/fsl-cuda10.2:6.0.5"
    # Define (at least the primary) output files.
    topup_out_path = subj_input.output_path / "topup"
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
                shared_resources["cpus"].value -= n_procs
                shared_resources["condition"].notify_all()

        topup_result = pitn.utils.proc_runner.call_docker_run(
            img=topup_img,
            cmd=topup_script,
            run_config=docker_config,
        )

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value += n_procs
                shared_resources["condition"].notify_all()

    t1 = time.time()
    print(f"{subj_id} Time taken for topup: {t1 - t}", flush=True)
    t = t1
    subj_output.topup.acqparams_f = acqparams_f
    subj_output.topup.merge_update(topup_out_files)

    # ###### 6. Extract mask of unwarped diffusion data.
    n_procs = 1
    docker_img = "tylerspears/fsl-cuda10.2:6.0.5"
    output_path = subj_input.output_path
    bet_out_path = output_path / "bet_topup2eddy"
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

        # Finally, run topup.
        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["condition"].wait_for(
                    lambda: shared_resources["cpus"].value >= n_procs
                )
                shared_resources["cpus"].value -= n_procs
                shared_resources["condition"].notify_all()

        topup2eddy_mask_f = pitn.data.preproc.dwi.bet_mask_median_dwis(
            input_dwi_f,
            out_mask_f=out_mask_f,
            robust_iters=True,
            tmp_dir=tmp_dir,
            docker_img=docker_img,
            docker_config=docker_config,
        )

        if shared_resources is not None:
            with shared_resources["condition"]:
                shared_resources["cpus"].value += n_procs
                shared_resources["condition"].notify_all()

        assert topup2eddy_mask_f.exists()

    t1 = time.time()
    print(f"{subj_id} Time taken for post-topup mask: {t1 - t}", flush=True)
    t = t1
    subj_output.bet_topup2eddy.mask_f = out_mask_f

    # ###### 7. Run eddy correction
    n_procs = 2
    docker_img = "tylerspears/fsl-cuda10.2:6.0.5"
    eddy_out_path = subj_input.output_path / "eddy"
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
                shared_resources["cpus"].value -= n_procs
                gpu_idx = shared_resources["gpus"].get()
                shared_resources["condition"].notify_all()
        else:
            gpu_idx = "0"

        # <https://stackoverflow.com/a/71429712/13225248>
        # docker_config["device_requests"] = [
        #     docker.types.DeviceRequest(
        #         device_ids=[str(gpu_idx)], capabilities=[["gpu"]]
        #     )
        # ]
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
                shared_resources["cpus"].value += n_procs
                shared_resources["condition"].notify_all()

    t1 = time.time()
    print(f"{subj_id} Time taken for eddy: {t1 - t}", flush=True)
    t = t1

    subj_output.eddy.input_bval = ap_pa_bval_path
    subj_output.eddy.input_bvec = ap_pa_bvec_path
    subj_outputs.eddy.input_dwi = ap_pa_dwi_path
    subj_outputs.eddy.index = index_path
    subj_output.eddy.slspec = slspec_path
    subj_output.eddy.merge_update(eddy_out_files)

    # ###### 8. Extract final mask of diffusion data and crop.

    return subj_outputs.to_dict()


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
    selected_subjs = ("P_01", "P_03", "P_04", "P_05", "P_06", "P_07", "P_08", "P_11")

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

    eddy_random_seed = 42138

    processed_outputs = dict()
    with concurrent.futures.ProcessPoolExecutor(
        multiprocessing.cpu_count()
    ) as executor:
        with multiprocessing.Manager() as manager:

            resources = manager.dict(
                MAX_CPUS=multiprocessing.cpu_count() - 2,
                condition=manager.Condition(),
                cpus=manager.Value("i", multiprocessing.cpu_count() - 2),
                gpus=manager.Queue(2),
            )
            # resources["gpus"].put("0")
            # resources["gpus"].put("1")
            resources["gpus"].put("GPU-ed20d87f-e88e-692f-0b56-548b8a05ddea")
            resources["gpus"].put("GPU-0636ee40-2eab-9533-1be7-dbbadade95c4")
            sub_count = 0
            # Grab files and process subject data.
            for subj_id, files in subj_files.items():

                subj_out_dir = output_dir / subj_id
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
