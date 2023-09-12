#!/usr/bin/bash

set -eou pipefail

hcp_subj_id="$1"
target_vox_size="$2"
ismrm_tck_dir="$3"
ismrm_part_vols_prefix="$4"
ismrm_t1_f="$5"
ismrm_tissue_mask_f="$6"
push_warp_f="$7"
pullback_warp_f="$8"
lps_subj_t1w_f="$9"
fiberfox_param_f="${10}"
fiberfox_dwi_template_f="${11}"
fiberfox_bvec_f="${12}"
fiberfox_bval_f="${13}"
output_dir="${14}"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

mkdir --parents "$output_dir"
mitk_exp_basename="fiberfox_ismrm_warp_hcp-${hcp_subj_id}"

# Copy parameter file
cp --archive --update "$fiberfox_param_f" "${output_dir}/${mitk_exp_basename}.ffp"

# Resample the HCP T1 to the target vox size, which will define the uncropped output
# space.
uncropped_subj_target_space_ref_f="${output_dir}/hcp-${hcp_subj_id}_uncropped_target_space_t1w.nii.gz"
mrgrid -nthreads $N_PROCS -interp cubic -force \
    "$lps_subj_t1w_f" \
    regrid \
    -voxel "$target_vox_size" \
    "$uncropped_subj_target_space_ref_f"

# Warp the tissue mask and dilate to form a buffer around the brain. The mask will
# eventually form the basis of the subject output space.
uncropped_tissue_mask_f="${output_dir}/hcp-${hcp_subj_id}_uncropped_target_space_tissue_mask.nii.gz"
# uncropped_mask_f="${output_dir}/hcp-${hcp_subj_id}_uncropped_target_space_brain_mask.nii.gz"
mrtransform -force -nthreads $N_PROCS -interp nearest \
    "$ismrm_tissue_mask_f" \
    -warp "$pullback_warp_f" \
    -template "$uncropped_subj_target_space_ref_f" \
    "$uncropped_tissue_mask_f"

buffer_uncropped_tissue_mask_f="${output_dir}/hcp-${hcp_subj_id}_dilated_uncropped_target_space_tissue_mask.nii.gz"
maskfilter -force \
    "$uncropped_tissue_mask_f" \
    dilate \
    -npass 2 \
    "$buffer_uncropped_tissue_mask_f"
# Crop the tissue mask and T1 image, which will define the output space and spatial
# extent/fov.
subj_target_space_ref_f="${output_dir}/${mitk_exp_basename}.ffp_MASK.nii.gz"
mrgrid -force \
        "$uncropped_tissue_mask_f" \
        crop \
        -mask "$buffer_uncropped_tissue_mask_f" \
        "$subj_target_space_ref_f"
mrgrid -force \
        "$uncropped_subj_target_space_ref_f" \
        crop \
        -mask "$buffer_uncropped_tissue_mask_f" \
        "${output_dir}/hcp-${hcp_subj_id}_target_space_t1w.nii.gz"

rm -fv "$uncropped_subj_target_space_ref_f"

# Construct the dwi template volume by stacking single volume files in the
# target space.
subj_dwi_basename="hcp-${hcp_subj_id}_target_space_dwi"
n_dwis=$(mrinfo "$fiberfox_dwi_template_f" -size | sed -re 's/[0-9]+ [0-9]+ [0-9]+ ([0-9]+)/\1/')
mrcat -force -nthreads $N_PROCS -quiet \
    $(for i in $(seq $n_dwis); do echo "$subj_target_space_ref_f"; done) \
    -datatype float32 \
    - |
    mrcalc -force -nthreads $N_PROCS \
    - 0 -mult \
    "${output_dir}/${subj_dwi_basename}.nii.gz"
# We want the gradient directions to be consistent relative to each subject space, so
# just copy them over without rotation.
cp --archive "$fiberfox_bvec_f" "${output_dir}/${subj_dwi_basename}.bvecs"
cp --archive "$fiberfox_bval_f" "${output_dir}/${subj_dwi_basename}.bvals"

# Warp tck files to hcp subject
single_tck_out_dir="${output_dir}/track_bundles"
mkdir --parents "$single_tck_out_dir"
for tck_f in "${ismrm_tck_dir}"/*.tck; do
    tck_name="$(basename "$tck_f" .tck)"
    echo "$tck_name"
    tcktransform -force -nthreads $N_PROCS \
        "$tck_f" \
        "$push_warp_f" \
        "${single_tck_out_dir}/${tck_name}_ismrm_warp_hcp-${hcp_subj_id}.tck"
done
# Check for sub-bundles
if [ -d "${ismrm_tck_dir}/sub_bundles" ]; then
    mkdir --parents "${single_tck_out_dir}/sub_bundles"
    for tck_f in "${ismrm_tck_dir}/sub_bundles"/*.tck; do
        tck_name="$(basename "$tck_f" .tck)"
        echo "$tck_name"
        tcktransform -force -nthreads $N_PROCS \
            "$tck_f" \
            "$push_warp_f" \
            "${single_tck_out_dir}/sub_bundles/${tck_name}_ismrm_warp_hcp-${hcp_subj_id}.tck"
    done
fi


# Combine all individual tracks into one track file.
tckedit -force -nthreads $N_PROCS \
    "${single_tck_out_dir}"/*.tck \
    "${output_dir}/ismrm_warp_hcp-${hcp_subj_id}_tracks.tck"

# Warp partial volumes, ismrm T1, and the template DWI into the final subj target space.
for part_vol_f in "${ismrm_part_vols_prefix}"*.nii.gz; do
    part_vol_name="$(basename "$part_vol_f" .nii.gz)"
    pv_digit="$(echo "$part_vol_name" | sed -re 's/.*VOLUME([0-9])/\1/')"
    echo "$part_vol_name" "$pv_digit"
    mrtransform -nthreads $N_PROCS -force \
        -interp linear \
        "$part_vol_f" \
        -warp "$pullback_warp_f" \
        -template "$subj_target_space_ref_f" \
        "${output_dir}/${mitk_exp_basename}.ffp_VOLUME${pv_digit}.nii.gz"
done

# Warp ismrm T1 to subj target space
mrtransform -force -nthreads $N_PROCS -interp cubic \
    "$ismrm_t1_f" \
    -warp "$pullback_warp_f" \
    -template "$subj_target_space_ref_f" \
    "${output_dir}/ismrm_warp_hcp-${hcp_subj_id}_target_space_t1w.nii.gz"
