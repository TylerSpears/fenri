#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1

subj_dir=$(realpath "./$SUBJ_ID")
tmp_dwi_mif="$subj_dir/_tmp_dwi.mif"
fivett_f="/data/srv/outputs/pitn/hcp/full-res/fodf/$SUBJ_ID/T1w/postproc_5tt_parcellation.nii.gz"
mask_f="/data/srv/outputs/pitn/hcp/full-res/fodf/$SUBJ_ID/T1w/postproc_nodif_brain_mask.nii.gz"
echo "$subj_dir"
dwi_f="$subj_dir/postproc_${SUBJ_ID}_trilinear_dwi_prediction_2.0mm-to-1.25mm.nii.gz"
echo "$dwi_f"
mrconvert -info \
    "$dwi_f" \
    -fslgrad "$subj_dir/bvecs" "$subj_dir/bvals" \
    "$tmp_dwi_mif"
dwi2response -info -nthreads 5 \
    msmt_5tt \
    -wm_algo tournier \
    -mask "$mask_f" \
    "$tmp_dwi_mif" \
    "$fivett_f" \
    "$subj_dir/wm_response.txt" \
    "$subj_dir/gm_response.txt" \
    "$subj_dir/csf_response.txt"

wm_fod_f=$(basename "$dwi_f")
wm_fod_f="${wm_fod_f/dwi/wm_msmt_csd_fod}"
wm_fod_f="./$wm_fod_f"

dwi2fod -info -nthreads 5 \
    msmt_csd \
    "$tmp_dwi_mif" \
    "$subj_dir/wm_response.txt" "$wm_fod_f" \
    "$subj_dir/gm_response.txt" "$subj_dir/postproc_gm_msmt_csd_fod.nii.gz" \
    "$subj_dir/csf_response.txt" "$subj_dir/postproc_csf_msmt_csd_fod.nii.gz" \
    -lmax 8,4,0 \
    -niter 100 \
    -mask "$mask_f"
rm "$tmp_dwi_mif"
