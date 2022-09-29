#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:


# Automatically re-import project-specific modules.
# imports
import collections
import io
import json
import os
import subprocess
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

import pitn
from pitn.data.preproc.dwi import crop_nib_by_mask

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)


# In[2]:


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


# ## Directory Setup

# In[3]:


bids_dir = Path("/data/VCU_MS_Study/bids")
assert bids_dir.exists()

output_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
assert output_dir.exists()

selected_subjs = ("P_01", "P_03", "P_04", "P_05", "P_06", "P_07", "P_08", "P_11")


# ## Subject File Locations

# In[4]:


subj_files = Box(default_box=True)
for subj in selected_subjs:
    p_dir = bids_dir / subj
    for encode_direct in ("AP", "PA"):
        direct_files = list(p_dir.glob(f"*DKI*{encode_direct}*_[0-9][0-9][0-9].*"))
        dwi = list(filter(lambda p: p.name.endswith(".nii.gz"), direct_files))[0]
        bval = list(filter(lambda p: p.name.endswith("bval"), direct_files))[0]
        bvec = list(filter(lambda p: p.name.endswith("bvec"), direct_files))[0]
        json_info = list(filter(lambda p: p.name.endswith(".json"), direct_files))[0]
        subj_files[subj][encode_direct.lower()] = Box(
            dwi=dwi, bval=bval, bvec=bvec, json=json_info
        )


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
) -> dict:

    t = time.time()
    t0 = t
    print(f"Start time {t}")
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
            denoised_dwi, noise_std = dipy.denoise.localpca.mppca(
                arr=crop_dwi.get_fdata(),
                mask=crop_mask_nib.get_fdata(),
                patch_radius=patch_radius,
                pca_method="eig",
                return_sigma=True,
            )
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
        print(f"Time taken for MP-PCA {pe_direct}: {t1 - t}")
        t = t1

    # ###### 2. Remove Gibbs ringing artifacts.
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
            dwi_degibbsed = dipy.denoise.gibbs.gibbs_removal(
                dwi.get_fdata(),
                slice_axis=2,
                n_points=3,
                inplace=False,
                num_processes=4,
            )
            mask = nib.load(subj_output.denoise_mppca[pe_direct].mask_f).get_fdata()
            dwi_degibbsed = dwi_degibbsed * mask[..., None].astype(np.uint8)
            nib.save(
                nib.Nifti1Image(dwi_degibbsed, dwi.affine, header=dwi.header),
                gibbs_corrected_dwi_f,
            )

        subj_output.gibbs_remove[pe_direct].dwi_f = gibbs_corrected_dwi_f
        t1 = time.time()
        print(f"Time taken for Gibbs removal {pe_direct}: {t1 - t}")
        t = t1

    # ###### 3. Remove $B_1$ magnetic field bias.
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
                nthreads=4,
                force=True,
            )
            dwi_debias_result = pitn.utils.proc_runner.call_docker_run(
                img=docker_img, cmd=dwi_debias_script, run_config=docker_config
            )
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
        print(f"Time taken for B1 field bias removal {pe_direct}: {t1 - t}")
        t = t1

    # !DEBUG
    return subj_output
    # ###### 4. Check bvec orientations
    bvec_flip_correct_path = subj_input.output_path / "bvec_flip_correct"
    bvec_flip_correct_path.mkdir(exist_ok=True, parents=True)
    tmp_d = bvec_flip_correct_path / "tmp"
    tmp_d.mkdir(exist_ok=True, parents=True)

    docker_img = "dsistudio/dsistudio:chen-2022-08-18"
    for pe_direct in ("ap", "pa"):
        subj_pe = subj_input[pe_direct]
        tmp_d_pe = tmp_d / pe_direct
        tmp_d_pe.mkdir(exist_ok=True, parents=True)
        vols = pitn.utils.union_parent_dirs(
            bvec_flip_correct_path,
            tmp_d_pe,
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)
        correct_bvec = pitn.data.preproc.dwi.bvec_flip_correct(
            dwi_data=subj_pe.dwi.get_fdata(),
            dwi_affine=subj_pe.dwi.affine,
            bval=subj_pe.bval,
            bvec=subj_pe.bvec,
            tmp_dir=tmp_d_pe,
            docker_img=docker_img,
            docker_config=docker_config,
        )
        correct_bvec_file = bvec_flip_correct_path / f"{subj_id}_{pe_direct}_dwi.bvec"
        np.savetxt(correct_bvec_file, correct_bvec)
        subj_output.bvec_flip_correct[pe_direct].bvec_f = correct_bvec_file

    # ###### 2. Run topup
    # Topup really only needs a few b0s in each PE direction, so use image similarity to
    # find the "least distorted" b0s in each PE direction. Then, save out to a file for
    # topup to read.
    num_b0s_per_pe = 3
    b0_max = 50
    output_path = subj_info.output_path
    topup_out_path = output_path / "topup"
    topup_out_path.mkdir(exist_ok=True, parents=True)

    select_dwis = dict()
    for pe_direct in ("ap", "pa"):
        dwi = nib.load(subj_info[pe_direct].dwi.path)
        bval = np.loadtxt(subj_info[pe_direct].bval.path)
        bvec = np.loadtxt(subj_outputs.bvec_flip_correct[subj_id][pe_direct].path)
        # select_pe = Box()
        top_b0s = top_k_b0s.func(
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

    dwis_out_path = topup_out_path / f"{subj_id}_ap_pa_b0_input.nii.gz"
    select_b0_data = [select_dwis["ap"], select_dwis["pa"]]
    if dwis_out_path.exists():
        prev_select_b0_data = nib.load(str(dwis_out_path)).get_fdata()
        if not np.isclose(
            np.concatenate(select_b0_data, axis=-1), prev_select_b0_data
        ).all():
            combine_b0_file = join_save_dwis.func(
                dwis=select_b0_data,
                affine=nib.load(subj_info.ap.dwi.path).affine,
                dwis_out_f=str(dwis_out_path),
            )["dwi"]
        else:
            combine_b0_file = File(str(dwis_out_path))
    else:
        combine_b0_file = join_save_dwis.func(
            dwis=select_b0_data,
            affine=nib.load(subj_info.ap.dwi.path).affine,
            dwis_out_f=str(dwis_out_path),
        )["dwi"]

        # Create the acquisition parameters file.
        ap_readout_time = float(
            subj_info.ap.json_header_dict["EstimatedTotalReadoutTime"]
        )
        ap_pe_direct = subj_info.ap.json_header_dict["PhaseEncodingAxis"]
        ap_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
            ap_readout_time, *([ap_pe_direct] * num_b0s_per_pe)
        )
        pa_readout_time = float(
            subj_info.pa.json_header_dict["EstimatedTotalReadoutTime"]
        )
        pa_pe_direct = subj_info.pa.json_header_dict["PhaseEncodingAxis"]
        # The negation of the axis isn't present in these data, for whatever reason.
        if "-" not in pa_pe_direct:
            pa_pe_direct = f"{pa_pe_direct}-"
        pa_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
            pa_readout_time, *([pa_pe_direct] * num_b0s_per_pe)
        )

        acqparams = np.concatenate([ap_acqparams, pa_acqparams], axis=0)
        acqparams_path = topup_out_path / "acqparams.txt"
        if acqparams_path.exists():
            prev_acqparams = np.loadtxt(acqparams_path)
            if not np.isclose(prev_acqparams, acqparams).all():
                np.savetxt(acqparams_path, acqparams, fmt="%g")
        else:
            np.savetxt(acqparams_path, acqparams, fmt="%g")
        topup_acqparams_f = File(str(acqparams_path))
        subj_outputs.topup[subj_id].acqparams = topup_acqparams_f

        # Set up docker configuration for running topup.
        docker_vols = {
            str(topup_out_path),
        }
        docker_vols = list((v, v) for v in tuple(docker_vols))
        script_exec_config = dict(
            image="tylerspears/fsl-cuda10.2:6.0.5",
            executor="docker",
            volumes=docker_vols,
            gpus=0,
            memory=12,
            vcpus=3,
        )
        # Finally, run topup.
        # Most parameters were taken from configuration provided by FSL in `b02b0_1.cnf`,
        # which is supposedly optimized for topup runs on b0s when image dimensions are
        # divisible by 1 (a.k.a., all image sizes).
        subj_topup_expr = pitn.redun.fsl.topup.options(cache=True, limits={"cpu": 3})(
            imain=combine_b0_file,
            datain=subj_outputs.topup[subj_id].acqparams,
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
            script_exec_config=script_exec_config,
        )
        topup_exprs[subj_id] = subj_topup_expr
    concrete_topup = scheduler.run(topup_exprs)
    for subj_id in subj_ids:
        subj_outputs.topup[subj_id].merge_update(concrete_topup[subj_id])

    # ###### 3. Extract mask of unwarped diffusion data.
    for subj_id, subj_info in subj_inputs.items():
        output_path = subj_info.output_path
        bet_out_path = output_path / "bet_topup2eddy"
        bet_out_path.mkdir(exist_ok=True, parents=True)

        # Set up docker configuration for running bet.
        docker_vols = {
            str(bet_out_path),
            str(Path(subj_outputs.topup[subj_id].corrected_im.path).parent),
        }
        docker_vols = list((v, v) for v in tuple(docker_vols))
        script_exec_config = dict(
            image="tylerspears/fsl-cuda10.2:6.0.5",
            executor="docker",
            volumes=docker_vols,
            gpus=0,
            memory=8,
            vcpus=4,
        )

        bet_topup2eddy_mask_expr = pitn.redun.data.preproc.dwi.bet_mask_median_dwis(
            subj_outputs.topup[subj_id]["corrected_im"],
            out_file=str(bet_out_path / f"{subj_id}_bet_topup2eddy_mask.nii.gz"),
            robust_iters=True,
            script_exec_config=script_exec_config,
        )
        subj_outputs.bet_topup2eddy[subj_id] = bet_topup2eddy_mask_expr

    concrete_bet_topup2eddy = scheduler.run(subj_outputs.bet_topup2eddy.to_dict())
    subj_outputs.bet_topup2eddy = concrete_bet_topup2eddy

    # ###### 4. Run eddy correction
    eddy_exprs = dict()
    for subj_id, subj_info in subj_inputs.items():
        output_path = subj_info.output_path
        eddy_out_path = output_path / "eddy"
        eddy_out_path.mkdir(exist_ok=True, parents=True)

        # Create slspec file.
        slspec = pitn.fsl.estimate_slspec(
            subj_info.ap.json_header_dict,
            n_slices=nib.load(subj_info.ap.dwi.path).header.get_n_slices(),
        )
        slspec_path = eddy_out_path / "slspec.txt"
        if slspec_path.exists():
            prev_slspec = np.loadtxt(slspec_path)
            if not np.isclose(slspec, prev_slspec).all():
                np.savetxt(slspec_path, slspec, fmt="%g")
        else:
            np.savetxt(slspec_path, slspec, fmt="%g")
        slspec_f = File(str(slspec_path))
        subj_outputs.eddy[subj_id].slspec = slspec_f

        # Create index file that relates DWI index to the acquisition params.
        index_path = eddy_out_path / "index.txt"
        # The index file is 1-indexed, not 0-indexed.
        ap_acqp_idx = 1
        pa_acqp_idx = num_b0s_per_pe + 1
        index_acqp = np.asarray(
            [ap_acqp_idx] * nib.load(subj_info.ap.dwi.path).shape[-1]
            + [pa_acqp_idx] * nib.load(subj_info.pa.dwi.path).shape[-1]
        ).reshape(1, -1)
        if index_path.exists():
            prev_index_acqp = np.loadtxt(index_path)
            if not np.isclose(index_acqp, prev_index_acqp).all():
                np.savetxt(str(index_path), index_acqp, fmt="%g")
        else:
            np.savetxt(str(index_path), index_acqp, fmt="%g")
        index_acqp_f = File(str(index_path))
        subj_outputs.eddy[subj_id].index = index_acqp_f

        # Merge and save both AP and PA DWIs, bvals, and bvecs together.
        ap_pa_basename = f"{subj_id}_ap_pa_uncorrected_dwi"
        ap_pa_dwi_path = eddy_out_path / (ap_pa_basename + ".nii.gz")
        ap_pa_bval_path = eddy_out_path / (ap_pa_basename + ".bval")
        ap_pa_bvec_path = eddy_out_path / (ap_pa_basename + ".bvec")

        ap_pa_dwi = nib.Nifti1Image(
            np.concatenate(
                [
                    nib.load(subj_info.ap.dwi.path).get_fdata(),
                    nib.load(subj_info.pa.dwi.path).get_fdata(),
                ],
                axis=-1,
            ),
            affine=nib.load(subj_info.ap.dwi.path).affine,
        )
        if ap_pa_dwi_path.exists():
            prev_ap_pa_dwi = nib.load(str(ap_pa_dwi_path)).get_fdata()
            if not np.isclose(ap_pa_dwi.get_fdata(), prev_ap_pa_dwi).all():
                nib.save(ap_pa_dwi, str(ap_pa_dwi_path))
        else:
            nib.save(ap_pa_dwi, str(ap_pa_dwi_path))
        ap_pa_dwi_f = File(str(ap_pa_dwi_path))
        subj_outputs.eddy[subj_id].input_dwi = ap_pa_dwi_f

        ap_pa_bval = np.concatenate(
            [np.loadtxt(subj_info.ap.bval.path), np.loadtxt(subj_info.pa.bval.path)],
            axis=0,
        )
        if ap_pa_bval_path.exists():
            prev_ap_pa_bval = np.loadtxt(str(ap_pa_bval_path))
            if not np.isclose(ap_pa_bval, prev_ap_pa_bval).all():
                np.savetxt(str(ap_pa_bval_path), ap_pa_bval)
        else:
            np.savetxt(str(ap_pa_bval_path), ap_pa_bval)
        ap_pa_bval_f = File(str(ap_pa_bval_path))
        subj_outputs.eddy[subj_id].input_bval = ap_pa_bval_f

        ap_pa_bvec = np.concatenate(
            [np.loadtxt(subj_info.ap.bvec.path), np.loadtxt(subj_info.pa.bvec.path)],
            axis=1,
        )
        if ap_pa_bvec_path.exists():
            prev_ap_pa_bvec = np.loadtxt(str(ap_pa_bvec_path))
            if not np.isclose(ap_pa_bvec, prev_ap_pa_bvec).all():
                np.savetxt(str(ap_pa_bvec_path), ap_pa_bvec)
        else:
            np.savetxt(str(ap_pa_bvec_path), ap_pa_bvec)
        ap_pa_bvec_f = File(str(ap_pa_bvec_path))
        subj_outputs.eddy[subj_id].input_bvec = ap_pa_bvec_f

        # Set up docker configuration for running eddy with cuda.
        docker_vols = {
            str(eddy_out_path),
            str(Path(subj_outputs.topup[subj_id].corrected_im.path).parent),
            str(Path(subj_outputs.bet_topup2eddy[subj_id].path).parent),
        }
        docker_vols = list((v, v) for v in tuple(docker_vols))
        script_exec_config = dict(
            image="tylerspears/fsl-cuda10.2:6.0.5",
            executor="docker",
            volumes=docker_vols,
            gpus=1,
            memory=16,
            vcpus=6,
            # Task limits must go in the script config, not the script task creator!
            # Otherwise, the resource is only consumed to make the *task expression*,
            # but not consumed when actually *running* the script.
            limits={"gpu": 1},
        )
        # Run eddy.
        eddy_subj_expr = pitn.redun.fsl.eddy(
            imain=subj_outputs.eddy[subj_id].input_dwi,
            bvecs=subj_outputs.eddy[subj_id].input_bvec,
            bvals=subj_outputs.eddy[subj_id].input_bval,
            mask=subj_outputs.bet_topup2eddy[subj_id],
            index=subj_outputs.eddy[subj_id].index,
            acqp=subj_outputs.topup[subj_id].acqparams,
            slspec=subj_outputs.eddy[subj_id].slspec,
            topup_fieldcoef=subj_outputs.topup[subj_id].fieldcoef,
            topup_movpar=subj_outputs.topup[subj_id].movpar,
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
            initrand=subj_info.eddy_seed,
            cnr_maps=True,
            fields=True,
            dfields=True,
            # write_predictions=True,
            very_verbose=True,
            log_stdout=True,
            use_cuda=True,
            auto_select_gpu=True,
            out=str(eddy_out_path / f"{subj_id}_eddy"),
            script_exec_config=script_exec_config,
        )
        eddy_exprs[subj_id] = eddy_subj_expr

    concrete_eddy = scheduler.run(eddy_exprs)

    for subj_id in subj_ids:
        subj_outputs.eddy[subj_id].merge_update(concrete_eddy[subj_id])

    return subj_outputs.to_dict()


# Iterate over subjects and run preprocessing task.

eddy_random_seed = 42138

processed_outputs = dict()
preproc_inputs = collections.defaultdict(list)
for subj_id, files in subj_files.items():

    subj_out_dir = output_dir / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)

    ap_files = files.ap.to_dict()
    ap_json = ap_files.pop("json")
    with open(ap_json, "r") as f:
        ap_json_info = dict(json.load(f))
    ap_files = {k: Path(str(ap_files[k])).resolve() for k in ap_files.keys()}

    pa_files = files.pa.to_dict()
    pa_json = pa_files.pop("json")
    with open(pa_json, "r") as f:
        pa_json_info = dict(json.load(f))
    pa_files = {k: Path(str(pa_files[k])).resolve() for k in pa_files.keys()}

    subj_outputs = vcu_dwi_preproc(
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
    )
    # !DEBUG
    break

# Allow all subjects to be scheduled at once.

# In[ ]:
