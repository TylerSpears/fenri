#!/usr/bin/bash

set -eou pipefail

dwi_f="$1"
grad_table_f="$2"
mask_f="$3"
fivett_f="$4"
output_root_dir="$5"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
script_f="$0"

tmp_dir="${output_root_dir}/tmp"
mkdir --parents "$output_root_dir" "$tmp_dir"

# Crop and prepare volumes for odf fitting.
dwi_out_dir="${output_root_dir}/diffusion"
seg_dir="${output_root_dir}/segmentation"
# Normalize DWI
DWINORMALISE_CSF_PERCENTILE=${DWINORMALISE_CSF_PERCENTILE:-"75"}
mkdir --parents "$dwi_out_dir" "$seg_dir"
dwi_mif_f="${tmp_dir}/dwi.mif"
# Convert dwi to .mif for convenience
mrconvert -force -grad \
    "$grad_table_f" \
    "$dwi_f" \
    "$dwi_mif_f"
# Create strict CSF mask for normalizing the DWI.
csf_mask_f="${seg_dir}/strict_dwinormalise_csf_mask.nii.gz"
mrconvert \
    "$fivett_f" \
    -coord 3 3 -axes 0,1,2 \
    -strides "$dwi_f" \
    - |
    maskfilter -force -nthreads $N_PROCS \
        - \
        erode -npass 1 \
        "$csf_mask_f"
norm_dwi_f="${dwi_out_dir}/dwi_norm.nii.gz"
dwinormalise individual -force -nthreads $N_PROCS \
    "$dwi_mif_f" \
    "$csf_mask_f" \
    -intensity 1000 -percentile "$DWINORMALISE_CSF_PERCENTILE" \
    "$norm_dwi_f"
# Recreate the DWI mif file with the norm dwi
rm -fv "$dwi_mif_f"
mrconvert -force -grad \
    "$grad_table_f" \
    "$norm_dwi_f" \
    "$dwi_mif_f"

# Estimate response functions for 3 tissues
odf_out_dir="${output_root_dir}/odf"
non_norm_output_dir="${odf_out_dir}/non_norm"
mkdir --parents "$non_norm_output_dir"
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
out_wm_fod_f="${odf_out_dir}/wm_norm_msmt_csd_fod.nii.gz"
out_gm_fod_f="${odf_out_dir}/gm_norm_msmt_csd_fod.nii.gz"
out_csf_fod_f="${odf_out_dir}/csf_norm_msmt_csd_fod.nii.gz"
mtnormalise -force -nthreads $N_PROCS \
    "$wm_fod_f" "$out_wm_fod_f" \
    "$gm_fod_f" "$out_gm_fod_f" \
    "$csf_fod_f" "$out_csf_fod_f" \
    -mask "$mask_f" \
    -niter 15,7 -order 3 -reference 0.282095

mkdir --parents "${output_root_dir}/code"
cp --update -v "$script_f" "${output_root_dir}/code"

rm -rvf "$tmp_dir"
