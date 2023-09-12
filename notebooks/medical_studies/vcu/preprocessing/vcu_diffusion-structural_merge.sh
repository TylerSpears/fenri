#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
SUBJ_ROOT_DIR="$2"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS

FS_LESION_LABEL="99"
DWI_MASK_INTERP_PV_THRESH="0.1"
LUT="${FREESURFER_HOME}/luts/FreeSurferColorLUT.txt"

dwi_src_dir="${SUBJ_ROOT_DIR}/diffusion/preproc/09_final"
dwi_out_dir="${SUBJ_ROOT_DIR}/diffusion"
tmp_dir="${SUBJ_ROOT_DIR}/diffusion/tmp"
mkdir --parents "$tmp_dir"

anat_dir="${SUBJ_ROOT_DIR}/anat"
strides_vol="${anat_dir}/anat_mask.nii.gz"

# 1. Extract & compute mean b0 as a registration target.
mean_b0="${tmp_dir}/mean_b0.nii.gz"
dwiextract -info \
    "${dwi_src_dir}/${SUBJ_ID}_dwi.nii.gz" \
    -fslgrad "${dwi_src_dir}/${SUBJ_ID}.bvec" "${dwi_src_dir}/${SUBJ_ID}.bval" \
    -bzero \
    -strides "$strides_vol" \
    - |
    mrmath -info -force \
        - \
        mean -axis 3 \
        "$mean_b0"
strides_vol="$mean_b0"

# 2. Register (rigid? affine?) t2 to the mean b0 to get the anat -> diffusion transform.
# The t2w should already be registered to the T1.
t2_vol="${anat_dir}/t2w_brain.nii.gz"

"${ANTSPATH}/antsRegistrationSyNQuick.sh" \
    -d 3 \
    -n $N_PROCS \
    -f "$mean_b0" \
    -m "$t2_vol" \
    -j 1 \
    -t a \
    -o "${dwi_out_dir}/anat2diffusion_affine_ants_" \
    -z 1

# We don't want the actual registered image yet, so delete the ants outputs aside from
# the affine transform.
rm -fv "${dwi_out_dir}"/anat2diffusion_affine_ants_*.nii.gz

# Convert to an mrtrix-friendly format.
anat2dwi_tf_ants="${dwi_out_dir}/anat2diffusion_affine_ants_0GenericAffine.mat"
"${ANTSPATH}/ConvertTransformFile" 3 \
    "$anat2dwi_tf_ants" \
    "${dwi_out_dir}/anat2diffusion_affine_itk.txt"
anat2dwi_tf_mrtrix="${dwi_out_dir}/anat2diffusion_affine_mrtrix.txt"
transformconvert -info -force \
    "${dwi_out_dir}/anat2diffusion_affine_itk.txt" \
    itk_import \
    "$anat2dwi_tf_mrtrix"

# 3. Transform the higher-quality freesurfer brain mask into diffusion space, with some
# wiggle room for differences between T2 and b0s.
anat_mask="${anat_dir}/anat_mask.nii.gz"
dwi_mask="${dwi_out_dir}/dwi_mask.nii.gz"
mrtransform -info \
    "$anat_mask" \
    -linear "$anat2dwi_tf_mrtrix" \
    -template "$mean_b0" \
    -strides "$strides_vol" \
    -interp linear \
    - |
    mrthreshold -info \
        - \
        -abs $DWI_MASK_INTERP_PV_THRESH \
        - |
    maskfilter -info -force \
        - \
        dilate -npass 1 \
        "$dwi_mask"

# 4. Transform the anatomical volumes to the diffusion data.
# t1 volumes can just be directly transformed to diffusion space.
vol_f="${anat_dir}/t1w_brain.nii.gz"
out_f="${dwi_out_dir}/t1w_reg-dwi.nii.gz"
"${ANTSPATH}/antsApplyTransforms" --verbose 1 \
    -d 3 \
    --input "$vol_f" \
    --reference-image "$mean_b0" \
    --transform "$anat2dwi_tf_ants" \
    --interpolation BSpline \
    --output "$out_f"
# Correct overshoot/undershoot from registration interpolation.
vol_min=$(mrstats "$vol_f" -output min)
vol_max=$(mrstats "$vol_f" -output max)
mrcalc -force -info \
    "$out_f" $vol_min -max $vol_max -min "$out_f"

# T2 and FLAIR must include the native->T1 transform, as well.
# Used to mask the T2 and FLAIR in dwi-space.
anat_preproc_src_dir="${SUBJ_ROOT_DIR}/freesurfer/pre_freesurfer"
diff_space_mask_strict="${tmp_dir}/anat_mask_dwi-space.nii.gz"
mrtransform -force -info \
    "$anat_mask" \
    -linear "$anat2dwi_tf_mrtrix" \
    -template "$mean_b0" \
    -strides "$strides_vol" \
    -interp nearest \
    "$diff_space_mask_strict"

# t2w
vol_f="${anat_preproc_src_dir}/03_denoise/t2w_n4_denoise.nii.gz"
out_f="${dwi_out_dir}/t2w_reg-dwi.nii.gz"
native_anat2t1w_tf_ants="${anat_preproc_src_dir}/02_reg_t1/t2_reg_t1_0GenericAffine.mat"
"${ANTSPATH}/antsApplyTransforms" --verbose 1 \
    -d 3 \
    --input "$vol_f" \
    --reference-image "$mean_b0" \
    --transform "$anat2dwi_tf_ants" --transform "$native_anat2t1w_tf_ants" \
    --interpolation BSpline \
    --output "$out_f"
# Correct overshoot/undershoot from registration interpolation, and also mask
# the diffusion-registered anatomical image.
vol_min=$(mrstats "$vol_f" -output min)
vol_max=$(mrstats "$vol_f" -output max)
mrcalc -info \
    "$out_f" $vol_min -max $vol_max -min "$diff_space_mask_strict" -mult - |
    mrconvert -force -info \
        - \
        -strides "$strides_vol" \
        "$out_f"

# flair
vol_f="${anat_preproc_src_dir}/03_denoise/flair_n4_denoise.nii.gz"
out_f="${dwi_out_dir}/flair_reg-dwi.nii.gz"
native_anat2t1w_tf_ants="${anat_preproc_src_dir}/02_reg_t1/flair_reg_t1_0GenericAffine.mat"
"${ANTSPATH}/antsApplyTransforms" --verbose 1 \
    -d 3 \
    --input "$vol_f" \
    --reference-image "$mean_b0" \
    --transform "$anat2dwi_tf_ants" --transform "$native_anat2t1w_tf_ants" \
    --interpolation BSpline \
    --output "$out_f"
# Correct overshoot/undershoot from registration interpolation, and also mask
# the diffusion-registered anatomical image.
vol_min=$(mrstats "$vol_f" -output min)
vol_max=$(mrstats "$vol_f" -output max)
mrcalc -info \
    "$out_f" $vol_min -max $vol_max -min "$diff_space_mask_strict" -mult - |
    mrconvert -force -info \
        - \
        -strides "$strides_vol" \
        "$out_f"

# 5. Transform all segmentation maps into diffusion space, then re-extract the binary
# ROI masks.
anat_roi_dir="${anat_dir}/freesurfer_segmentations"
dwi_roi_dir="${dwi_out_dir}/freesurfer_segmentations_reg-dwi"
mkdir --parents "$dwi_roi_dir"

for seg_src in "$anat_roi_dir"/*; do
    segname="$(basename "$seg_src")"
    segment_map_anat_space="${seg_src}/${segname}.nii.gz"

    seg_roi_dir_dwi_space="${dwi_roi_dir}/${segname}/roi_masks"
    mkdir --parents "$seg_roi_dir_dwi_space"

    segment_map_dwi_space="${dwi_roi_dir}/${segname}/${segname}.nii.gz"
    mrtransform -force -info \
        "$segment_map_anat_space" \
        -linear "$anat2dwi_tf_mrtrix" \
        -template "$mean_b0" \
        -strides "$strides_vol" \
        -interp nearest \
        "$segment_map_dwi_space"

    "${SCRIPT_DIR}/structural_images/split_freesurfer_segmentations_to_masks.py" \
        --fs_seg="$segment_map_dwi_space" \
        --lut="$LUT" \
        --output_dir="$seg_roi_dir_dwi_space"
done

# Generate the 5tt map according to a freesurfer parcellation.
fivett_seg_dir="${dwi_roi_dir}/mrtrix_5tt"
mkdir --parents "$fivett_seg_dir/roi_masks"
fivett_seg="${fivett_seg_dir}/5tt.nii.gz"
5ttgen -info \
    freesurfer \
    "${dwi_roi_dir}/aparc.a2009s+aseg/aparc.a2009s+aseg.nii.gz" \
    -lut "$LUT" \
    -nocrop \
    -sgm_amyg_hipp \
    "$fivett_seg" \
    -force
# Extract all 5 tissue types.
mrconvert -info \
    "$fivett_seg" \
    -coord 3 0 \
    -axes 0,1,2 \
    "${fivett_seg_dir}/roi_masks/cortical_gray_matter.nii.gz" \
    -force
mrconvert -info \
    "$fivett_seg" \
    -coord 3 1 \
    -axes 0,1,2 \
    "${fivett_seg_dir}/roi_masks/sub-cortical_gray_matter.nii.gz" \
    -force
mrconvert -info \
    "$fivett_seg" \
    -coord 3 2 \
    -axes 0,1,2 \
    "${fivett_seg_dir}/roi_masks/white_matter.nii.gz" \
    -force
mrconvert -info \
    "$fivett_seg" \
    -coord 3 3 \
    -axes 0,1,2 \
    "${fivett_seg_dir}/roi_masks/csf.nii.gz" \
    -force
mrconvert -info \
    "$fivett_seg" \
    -coord 3 4 \
    -axes 0,1,2 \
    "${fivett_seg_dir}/roi_masks/pathological_tissue.nii.gz" \
    -force

# Transform the lesion mask into diffusion space.
out_lesion_mask="${dwi_out_dir}/ms_lesion_mask.nii.gz"
source_mask="${anat_dir}/ms_lesion_mask.nii.gz"
mrtransform -force -info \
    "$source_mask" \
    -linear "$anat2dwi_tf_mrtrix" \
    -template "$mean_b0" \
    -strides "$strides_vol" \
    -interp nearest \
    "$out_lesion_mask"

# 6. Apply the improved dwi mask onto the dwi itself, and all derived parameter maps.
mrcalc -info \
    "${dwi_src_dir}/${SUBJ_ID}_dwi.nii.gz" \
    "$dwi_mask" \
    -mult \
    - |
    mrconvert -info \
        - \
        -strides "$strides_vol" \
        "${dwi_out_dir}/${SUBJ_ID}_dwi.nii.gz" \
        -force
cp --archive --update "${dwi_src_dir}/${SUBJ_ID}.bvec" "${dwi_src_dir}/${SUBJ_ID}.bval" \
    "$dwi_out_dir/"

dwi_out_param_maps="${dwi_out_dir}/parameter_maps"
mkdir --parents "$dwi_out_param_maps"
src_param_maps="${SUBJ_ROOT_DIR}/diffusion/preproc/parameter_maps"

for param_group in "$src_param_maps"/*; do
    group_name="$(basename "$param_group")"
    group_out="${dwi_out_param_maps}/${group_name}"
    mkdir --parents "$group_out"

    for vol in "$param_group"/*.nii.gz; do
        vol_name="$(basename "$vol")"

        # Remove subj id from the filename, if present.
        if [[ "$vol_name" == "$SUBJ_ID"* ]]; then
            vol_name=${vol_name//"${SUBJ_ID}_"/}
        fi

        # Rename the noddi parameters to have more readable names.
        if [[ "$group_name" == "noddi" ]]; then
            case "$vol_name" in
            "partial_volume_0"*)
                vol_name="partial_volume_fraction_csf-isotropic-ball.nii.gz"
                ;;
            "partial_volume_1"*)
                vol_name="partial_volume_fraction_wm-intra-cellular-Watson-bundle.nii.gz"
                ;;
            "SD1WatsonDistributed_1_partial_volume_0"*)
                vol_name="normalized_partial_volume_fraction_intra-axonal-stick_within-wm-Watson-bundle.nii.gz"
                ;;
            "SD1WatsonDistributed_1_SD1Watson_1_mu"*)
                vol_name="wm-Watson-bundle_mu.nii.gz"
                ;;
            "SD1WatsonDistributed_1_SD1Watson_1_odi"*)
                vol_name="wm-Watson-bundle_odi.nii.gz"
                ;;
            esac
            echo
        fi

        mrcalc -info \
            "$vol" \
            "$dwi_mask" \
            -mult \
            - |
            mrconvert -info \
                - \
                -strides "$strides_vol" \
                "${group_out}/${vol_name}" \
                -force
    done
done

# 7. Copy this script into the subject's code directory.
code_dir="${SUBJ_ROOT_DIR}/code"
mkdir --parents "$code_dir"
cp --archive --update "$(realpath "$0")" "$code_dir/"
cp --archive --update "${SCRIPT_DIR}/"*parameter_maps.* "${SCRIPT_DIR}/vcu_preproc.py" "$code_dir/"

# Delete the tmp directory.
rm -rvf "$tmp_dir"
