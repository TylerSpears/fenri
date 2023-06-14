#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
DWI_DIR="$2"
ODF_DIR="$3"
CLUSTER_OUT_BASE_DIR="$4"

odf_mask_im="$ODF_DIR/postproc_nodif_brain_mask.nii.gz"
aseg_im="$ODF_DIR/postproc_aparc.a2009s+aseg.nii.gz"
regrid_ref_im="$ODF_DIR/postproc_wm_msmt_csd_fod.nii.gz"
t1w_brain_im="$DWI_DIR/../T1w_acpc_dc_restore_brain.nii.gz"
aseg_t1w_space_im="$DWI_DIR/../aparc.a2009s+aseg.nii.gz"

dti_im="$ODF_DIR/postproc_dti.nii.gz"
dwi2tensor \
    "$DWI_DIR/data.nii.gz" \
    -mask "$DWI_DIR/nodif_brain_mask.nii.gz" \
    -fslgrad "$DWI_DIR/bvecs" "$DWI_DIR/bvals" \
    - \
    -info -force |
    mrgrid \
    - \
    regrid \
    -template "$regrid_ref_im" \
    -interp nearest \
    "$dti_im" \
    -info -force

fa_im="$ODF_DIR/postproc_fa.nii.gz"
v1_im="$ODF_DIR/postproc_v1.nii.gz"
tensor2metric \
    "$dti_im" \
    -mask "$odf_mask_im" \
    -fa "$fa_im" \
    -num 1 \
    -vector "$v1_im" \
    -info -force

cnn_dti_thal_out_dir="$CLUSTER_OUT_BASE_DIR/$SUBJ_ID/tregidgo_etal_2023_cnn-dti-segment"
mkdir --parents "$cnn_dti_thal_out_dir"


thal_seg_t1_space_im="$cnn_dti_thal_out_dir/thal-nuclei_t1w-space_dti-cnn_segment.nii.gz"
mri_segment_thalamic_nuclei_dti_cnn \
    --t1 "$t1w_brain_im" \
    --aseg "$aseg_t1w_space_im" \
    --fa "$fa_im" \
    --v1 "$v1_im" \
    --o "$thal_seg_t1_space_im" \
    --vol "$cnn_dti_thal_out_dir/thal-nuclei_t1w-space_dti-cnn_segment_volume-measures.csv" \
    --threads 20 --cpu

# Resample the thal. seg. into DTI space from T1 space.
thal_seg_dwi_space_im="$cnn_dti_thal_out_dir/thal-nuclei_dwi-space_dti-cnn_segment.nii.gz"
mrgrid \
    "$thal_seg_t1_space_im" \
    regrid \
    -template "$regrid_ref_im" \
    -interp nearest \
    "$thal_seg_dwi_space_im" \
    -info -force
