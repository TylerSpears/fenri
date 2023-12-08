#!/usr/bin/bash

set -eou pipefail

subj_id="$1"
ismrm_fivett_f="$2"
ismrm_to_subj_pullback_warp_f="$3"
subj_template_f="$4"
subj_brain_mask_f="$5"
subj_comp1_pv_f="$6"
subj_comp2_pv_f="$7"
subj_comp3_pv_f="$8"
subj_comp4_pv_f="$9"
output_dir="${10}"

script_dir="$(dirname "$0")"
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

mkdir --parents "$output_dir"

warped_fivett_f="${output_dir}/hcp-100307_warp_hcp-${subj_id}_fivett_seg.nii.gz"
mrtransform -force -info \
    "$ismrm_fivett_f" \
    -warp "$ismrm_to_subj_pullback_warp_f" \
    -template "$subj_template_f" \
    "$warped_fivett_f"

out_fivett_f="${output_dir}/hcp-${subj_id}_fivett_seg.nii.gz"
"${script_dir}/fs_sim_fivett_merge.py" \
    "$warped_fivett_f" \
    "$subj_comp1_pv_f" \
    "$subj_comp2_pv_f" \
    "$subj_comp3_pv_f" \
    "$subj_comp4_pv_f" \
    "$subj_brain_mask_f" \
    "$out_fivett_f"

# Flip across the y axis to align with the simulation files.
mrtransform \
    "$out_fivett_f" \
    -flip 1 \
    - |
    mrconvert -force \
    - \
    "$out_fivett_f"

# mrtransform -force \
#     -interp nearest \
#     ../src_subj_100307/rps-hcp-100307_warped_ismrm-2015_fivett_seg.nii.gz \
#     -warp 02_warp_creation/mrtrix_ismrm-2015_to_hcp-121618_pullback_warp_corrected.nii.gz \
#     -template 03_fiberfox_prep/ismrm_warp_hcp-121618_target_space_t1w.nii.gz \
#     hcp-100307_warp_hcp-12168_fivett_seg.nii.gz
