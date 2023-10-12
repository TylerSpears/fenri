#!/usr/bin/bash

set -eou pipefail

dwi_f="$1"
bvec_f="$2"
bval_f="$3"
mask_f="$4"
fivett_f="$5"
output_dir="$6"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
script_f="$0"

tmp_dir="${output_dir}/tmp"
mkdir --parents "$output_dir" "$tmp_dir"

# Crop and prepare volumes for odf fitting.
dwi_mif_f="${tmp_dir}/dwi.mif"
# Convert dwi to .mif for convenience
mrconvert -force -fslgrad \
    "$bvec_f" \
    "$bval_f" \
    "$dwi_f" \
    "$dwi_mif_f"

# Estimate response functions for 3 tissues
non_norm_output_dir="${output_dir}/non_normalised"
mkdir --parent "$non_norm_output_dir"
dwi2response msmt_5tt -force \
    -wm_algo tournier \
    -mask "$mask_f" \
    "$dwi_mif_f" \
    "$fivett_f" \
    "${non_norm_output_dir}/wm_response.txt" \
    "${non_norm_output_dir}/gm_response.txt" \
    "${non_norm_output_dir}/csf_response.txt"

# Estimate FODs with CSD.
wm_fod_f="${non_norm_output_dir}/wm_msmt_csd_fod.nii.gz"
gm_fod_f="${non_norm_output_dir}/gm_msmt_csd_fod.nii.gz"
csf_fod_f="${non_norm_output_dir}/csf_msmt_csd_fod.nii.gz"
dwi2fod -nthreads $N_PROCS -force \
    -lmax 8,0,0 \
    -mask "$mask_f" \
    msmt_csd \
    "$dwi_mif_f" \
    "${non_norm_output_dir}/wm_response.txt" "$wm_fod_f" \
    "${non_norm_output_dir}/gm_response.txt" "$gm_fod_f" \
    "${non_norm_output_dir}/csf_response.txt" "$csf_fod_f"

# Normalize FODs across the entire brain.
out_wm_fod_f="${output_dir}/wm_norm_msmt_csd_fod.nii.gz"
out_gm_fod_f="${output_dir}/gm_norm_msmt_csd_fod.nii.gz"
out_csf_fod_f="${output_dir}/csf_norm_msmt_csd_fod.nii.gz"
mtnormalise -force -nthreads $N_PROCS \
    "$wm_fod_f" "$out_wm_fod_f" \
    "$gm_fod_f" "$out_gm_fod_f" \
    "$csf_fod_f" "$out_csf_fod_f" \
    -mask "$mask_f" \
    -niter 15,7 -order 3 -reference 0.282095

mkdir --parents "$(dirname "$output_dir")/code"
cp --update -v "$script_f" "$(dirname "$output_dir")/code"

rm -rvf "$tmp_dir"
