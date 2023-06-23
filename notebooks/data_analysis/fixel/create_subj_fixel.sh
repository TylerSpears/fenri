#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
ODF_F="$2"
FIVETT_F="$3"
FIXEL_OUT_DIR="$4"

SCRIPT_F="$(basename "$0")"
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}


fmls_integral=0
fmls_peak_value=0.1
fmls_lobe_merge_ratio=0.9

peak_amp_f="wm_peak_amp.mif"
wm_mask_f="/tmp/_${SUBJ_ID}_tmp_wm_mask.mif"
mrconvert "$FIVETT_F" -coord 3 2 -strides 1,2,3 "$wm_mask_f"
# Make sure input odf is in RAS format, to simplify later directional comparisons.
mrconvert "$ODF_F" -strides 1,2,3 - |
     fod2fixel -info -nthreads $N_PROCS \
         - \
         "$FIXEL_OUT_DIR" \
         -mask "$wm_mask_f" \
         -fmls_integral $fmls_integral \
         -fmls_peak_value $fmls_peak_value \
         -fmls_lobe_merge_ratio $fmls_lobe_merge_ratio \
         -dirpeak \
         -afd "wm_afd.mif" \
         -peak_amp "$peak_amp_f"
rm -f "$wm_mask_f"

fixel2peaks -info -nthreads $N_PROCS \
    "${FIXEL_OUT_DIR}/${peak_amp_f}" \
    "${FIXEL_OUT_DIR}/wm_peak_dirs.nii.gz" \
    -number 5 \
    -nan

subj_code_dir=$(dirname "$(dirname "$(realpath "$ODF_F")")")/code
mkdir --parents "$subj_code_dir"
cp --update --archive --force "$0" "${subj_code_dir}/${SCRIPT_F}"

