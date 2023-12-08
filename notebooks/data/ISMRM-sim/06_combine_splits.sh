#!/usr/bin/bash

set -eou pipefail

subj_id="$1"
warped_t1w_f="$2"
tck_all_bundles_f="$3"
tck_bundles_dir="$4"

N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

out_dir="${subj_id}/processed"
diffusion_out_dir="${out_dir}/diffusion"
mkdir -p "$out_dir" "$diffusion_out_dir"
mask_f="${subj_id}/split_01/01_ismrm_warp_hcp-${subj_id}_fiberfox_sim.ffp_MASK.nii.gz"

echo $subj_id
tmp_dir="${subj_id}/tmp"
mkdir -p "$tmp_dir"

crop_mask_f="${out_dir}/brain_mask.nii.gz"
mrgrid -force \
    "$mask_f" \
    crop \
    -mask "$mask_f" \
    - |
    mrconvert -force \
        - \
        -strides 1,2,3 \
        "$crop_mask_f"

for i_comp in {1..4}; do
    echo $i_comp
    comp_f="${subj_id}/split_01/01_ismrm_warp_hcp-${subj_id}_fiberfox_sim_Compartment${i_comp}.nii.gz"
    mrgrid -force \
        "$comp_f" \
        crop \
        -mask "$mask_f" \
        -strides 1,2,3 \
        "${diffusion_out_dir}/pv_compartment_${i_comp}.nii.gz"
done

dwi_mifs=()
for split in split_01 split_02 split_03 split_04; do
    echo $split
    split_mif="${tmp_dir}/${split}_dwi.mif"
    mrconvert -force \
        "${subj_id}/${split}"/[0-9][0-9]_*"${subj_id}"*fiberfox_sim.nii.gz \
        -fslgrad "${subj_id}/${split}"/[0-9][0-9]_*.ffp.bvecs \
        "${subj_id}/${split}"/[0-9][0-9]_*.ffp.bvals \
        -bvalue_scaling false \
        "$split_mif"
    dwi_mifs+=("$split_mif")
done

mrcat \
    "${dwi_mifs[@]}" \
    -axis 3 \
    - |
    mrgrid -nthreads $N_PROCS \
        - \
        crop \
        -mask "$mask_f" \
        - |
    mrconvert -force \
        - \
        -bvalue_scaling false \
        -strides 1,2,3 \
        "${tmp_dir}/dwi.mif"
# Realign gradient directions according to dwigradcheck
dwigradcheck -force -nthreads $N_PROCS \
    "${tmp_dir}/dwi.mif" \
    -mask "$crop_mask_f" \
    -export_grad_mrtrix "${diffusion_out_dir}/grad_mrtrix.b"
mrconvert -force -nthreads $N_PROCS \
    "${tmp_dir}/dwi.mif" \
    "${diffusion_out_dir}/dwi.nii.gz"

mrconvert \
    "$warped_t1w_f" \
    -strides 1,2,3 \
    - |
    mrgrid -force \
        - \
        crop \
        -mask "$mask_f" \
        "${out_dir}/t1w_warped.nii.gz"

track_dir="${out_dir}/tracks"
mkdir -p "$track_dir"
cp --update -v "$tck_all_bundles_f" "${track_dir}/tracks_warped.tck"
cp -vR "$tck_bundles_dir" "$track_dir/"

mkdir --parents "${out_dir}/code"
cp --update -v "$0" "${out_dir}/code"

rm -rvf "$tmp_dir"
