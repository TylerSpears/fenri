#!/usr/bin/bash

# Example commands for postprocessing each simulation subject.

set -eou pipefail

subj_id="$1"

export N_PROCS=20

# Create initial 5tt image by warping ISMRM 5tt segmentation and thresholding pv maps.
tissue_mask_creation.sh \
        "$subj_id" \
        "hcp-100307_warped_ismrm-2015_fivett_seg.nii.gz" \
        "mrtrix_ismrm-2015_to_hcp-${subj_id}_pullback_warp_corrected.nii.gz" \
        "${subj_id}/processed/brain_mask.nii.gz" \
        "${subj_id}/processed/brain_mask.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_1.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_2.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_3.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_4.nii.gz" \
        "."


mkdir -p "${subj_id}/processed/segmentation"
# Regrid the warped 5tt image to simulation subject space.
mrgrid -strides 1,2,3 -force \
        "hcp-100307_warp_hcp-${subj_id}_fivett_seg.nii.gz" \
        regrid \
        -template "${subj_id}/processed/brain_mask.nii.gz" \
        -interp nearest \
        "/tmp/${subj_id}_tmp_fivett.nii.gz"
# Refine the subject 5tt image by thresholding the pv fraction maps.
fs_sim_fivett_merge.py \
        "/tmp/${subj_id}_tmp_fivett.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_1.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_2.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_3.nii.gz" \
        "${subj_id}/processed/diffusion/pv_compartment_4.nii.gz" \
        "${subj_id}/processed/brain_mask.nii.gz" \
        "${subj_id}/processed/segmentation/fivett_segmentation.nii.gz"

rm -vf "/tmp/${subj_id}_tmp_fivett.nii.gz"

# Use MSMT-CSD to estimate ODFs from the simulation DWIs. Also performs normalization of
# DWIs and ODF coefficients.
fit_odf_ismrm-2015-sims.sh \
        "${subj_id}/processed/diffusion/dwi.nii.gz" \
        "${subj_id}/processed/diffusion/grad_mrtrix.b" \
        "${subj_id}/processed/brain_mask.nii.gz" \
        "${subj_id}/processed/segmentation/fivett_segmentation.nii.gz" \
        "${subj_id}/processed"
