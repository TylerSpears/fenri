#!/usr/bin/env bash

set -eou pipefail

SUBJ_ID=$1
t1="$2"
t2="$3"
flair="$4"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS

LGA_KAPPA="0.25"
LGA_PROB_THRESH="0.25"
FS_LESION_LABEL="99"

export SUBJECTS_DIR="$(pwd)/${SUBJ_ID}/freesurfer"
PRE_FS_DIR="${SUBJECTS_DIR}/pre_freesurfer"
mkdir --parents "$PRE_FS_DIR"
# NU_CORRECT_DIR="${SUBJECTS_DIR}/nu_debiased"
# mkdir --parents "$NU_CORRECT_DIR"
# REG_DIR="${SUBJECTS_DIR}/reg_t1"
# mkdir --parents "$REG_DIR"
FREESURFER_EXPERT_OPTIONS="${SCRIPT_DIR}/freesurfer_expert_options.txt"
SPM_LST_DIR="${SUBJECTS_DIR}/spm-lst_lga"
mkdir --parents "$SPM_LST_DIR"
SPM_ROOT_DIR="/opt/spm"

ANAT_OUT_DIR="$(realpath "${SUBJECTS_DIR}/../anat")"
mkdir --parents "$ANAT_OUT_DIR"

# 1. De-bias all volumes with N4.
DEBIAS_DIR="${PRE_FS_DIR}/01_n4_debias"
mkdir --parents "$DEBIAS_DIR"

t1_debiased="${DEBIAS_DIR}/t1w_n4.nii.gz"
t2_debiased="${DEBIAS_DIR}/t2w_n4.nii.gz"
flair_debiased="${DEBIAS_DIR}/flair_n4.nii.gz"
if [ ! -s "$t1_debiased" ] ||
    [ ! -s "$t2_debiased" ] ||
    [ ! -s "$flair_debiased" ] ||
    [ "$t1" -nt "$t1_debiased" ] ||
    [ "$t2" -nt "$t2_debiased" ] ||
    [ "$flair" -nt "$flair_debiased" ]; then

    "${ANTSPATH}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$t1" \
        --shrink-factor 3 \
        --output [ "$t1_debiased","${DEBIAS_DIR}/t1_bias_field.nii.gz" ]

    "${ANTSPATH}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$t2" \
        --shrink-factor 3 \
        --output [ "$t2_debiased","${DEBIAS_DIR}/t2_bias_field.nii.gz" ]

    "${ANTSPATH}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$flair" \
        --shrink-factor 3 \
        --output [ "$flair_debiased","${DEBIAS_DIR}/flair_bias_field.nii.gz" ]
else
    echo "****** $SUBJ_ID | Already completed image de-biasing******"
fi

# 2. Rigid register de-biased FLAIR and T2 to T1 with ANTS
REG_DIR="${PRE_FS_DIR}/02_reg_t1"
mkdir --parents "$REG_DIR"

# Register T2 to T1
t2_reg_t1="${REG_DIR}/t2_reg_t1.nii.gz"
t2_reg_t1_tf="${REG_DIR}/t2_reg_t1_0GenericAffine.mat"
if [ ! -s "$t2_reg_t1" ] || [ ! -s "$t2_reg_t1_tf" ] ||
    [ "$t1_debiased" -nt "$t2_reg_t1" ] ||
    [ "$t2_debiased" -nt "$t2_reg_t1" ]; then
    # Full registration command with bspline interpolation.
    "${ANTSPATH}/antsRegistration" --verbose 1 \
        --dimensionality 3 \
        --float 0 \
        --collapse-output-transforms 1 \
        --output [ "${REG_DIR}/t2_reg_t1_","${REG_DIR}/t2_reg_t1.nii.gz","${REG_DIR}/t1_reg_t2.nii.gz" ] \
        --interpolation BSpline[ 3 ] \
        --use-histogram-matching 0 \
        --winsorize-image-intensities [ 0.005,0.995 ] \
        --initial-moving-transform [ "$t1_debiased","$t2_debiased",1 ] \
        --transform Rigid[ 0.1 ] \
        --metric MI[ "$t1_debiased","$t2_debiased",1,32,Regular,0.25 ] \
        --convergence [ 1000x500x250x100,1e-6,10 ] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox
else
    echo "****** $SUBJ_ID | Already completed T2 -> T1 registration******"
fi

# Register FLAIR to T1
flair_reg_t1="${REG_DIR}/flair_reg_t1.nii.gz"
flair_reg_t1_tf="${REG_DIR}/flair_reg_t1_0GenericAffine.mat"
if [ ! -s "$flair_reg_t1" ] ||
    [ ! -s "$flair_reg_t1_tf" ] ||
    [ "$t1_debiased" -nt "$flair_reg_t1" ] ||
    [ "$flair_debiased" -nt "$flair_reg_t1" ]; then
    "${ANTSPATH}/antsRegistration" --verbose 1 \
        --dimensionality 3 \
        --float 0 \
        --collapse-output-transforms 1 \
        --output [ "${REG_DIR}/flair_reg_t1_","${REG_DIR}/flair_reg_t1.nii.gz","${REG_DIR}/t1_reg_flair.nii.gz" ] \
        --interpolation BSpline[ 3 ] \
        --use-histogram-matching 0 \
        --winsorize-image-intensities [ 0.005,0.995 ] \
        --initial-moving-transform [ "$t1_debiased","$flair_debiased",1 ] \
        --transform Rigid[ 0.1 ] \
        --metric MI[ "$t1_debiased","$flair_debiased",1,32,Regular,0.25 ] \
        --convergence [ 1000x500x250x100,1e-6,10 ] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox
else
    echo "****** $SUBJ_ID | Already completed FLAIR -> T1 registration******"
fi

# 3. Denoise all debiased volumes with ants.
DENOISE_DIR="${PRE_FS_DIR}/03_denoise"
mkdir --parents "$DENOISE_DIR"

t1_denoised="${DENOISE_DIR}/t1w_n4_denoise.nii.gz"
t2_denoised="${DENOISE_DIR}/t2w_n4_denoise.nii.gz"
flair_denoised="${DENOISE_DIR}/flair_n4_denoise.nii.gz"
t2_denoised_reg_t1="${DENOISE_DIR}/t2w_n4_denoise_reg_t1.nii.gz"
flair_denoised_reg_t1="${DENOISE_DIR}/flair_n4_denoise_reg_t1.nii.gz"
if [ ! -s "$t1_denoised" ] ||
    [ ! -s "$t2_denoised" ] ||
    [ ! -s "$flair_denoised" ] ||
    [ ! -s "$t2_denoised_reg_t1" ] ||
    [ ! -s "$flair_denoised_reg_t1" ] ||
    [ "$t1_debiased" -nt "$t1_denoised" ] ||
    [ "$t2_debiased" -nt "$t2_denoised" ] ||
    [ "$flair_debiased" -nt "$flair_denoised" ] ||
    [ "$t2_reg_t1_tf" -nt "$t2_denoised_reg_t1" ] ||
    [ "$flair_reg_t1_tf" -nt "$flair_denoised_reg_t1" ]; then

    "${ANTSPATH}/DenoiseImage" --verbose 1 \
        -d 3 \
        --input-image "$t1_debiased" \
        --noise-model Rician \
        --shrink-factor 1 --patch-radius 1 --search-radius 2 \
        --output "$t1_denoised"

    "${ANTSPATH}/DenoiseImage" --verbose 1 \
        -d 3 \
        --input-image "$t2_debiased" \
        --noise-model Rician \
        --shrink-factor 1 --patch-radius 1 --search-radius 2 \
        --output "$t2_denoised"

    "${ANTSPATH}/DenoiseImage" --verbose 1 \
        -d 3 \
        --input-image "$flair_debiased" \
        --noise-model Rician \
        --shrink-factor 1 --patch-radius 1 --search-radius 2 \
        --output "$flair_denoised"

    # Apply previous registration transform to the denoised T2 and FLAIR images.
    "${ANTSPATH}/antsApplyTransforms" --verbose 1 \
        -d 3 \
        --input "$t2_denoised" \
        --reference-image "$t1_denoised" \
        --transform "$t2_reg_t1_tf" \
        --interpolation BSpline \
        --output "$t2_denoised_reg_t1"
    "${ANTSPATH}/antsApplyTransforms" --verbose 1 \
        -d 3 \
        --input "$flair_denoised" \
        --reference-image "$t1_denoised" \
        --transform "$flair_reg_t1_tf" \
        --interpolation BSpline \
        --output "$flair_denoised_reg_t1"

    # Correct overshoot/undershoot from registration interpolation.
    t2_min=$(mrstats "$t2_denoised" -output min)
    t2_max=$(mrstats "$t2_denoised" -output max)
    mrcalc -force \
        "$t2_denoised_reg_t1" $t2_min -max $t2_max -min "$t2_denoised_reg_t1"

    flair_min=$(mrstats "$flair_denoised" -output min)
    flair_max=$(mrstats "$flair_denoised" -output max)
    mrcalc -force \
        "$flair_denoised_reg_t1" $flair_min -max $flair_max -min "$flair_denoised_reg_t1"

else
    echo "****** $SUBJ_ID | Already completed image denoising******"
fi

# 4. Run recon-all with the original T1, up to and including the creation of
# aseg.auto_no_CC for the initial GM labels and nu-corrected T1 volume.
FREESURFER_SAMPLE_DIR="${SUBJECTS_DIR}/${SUBJ_ID}"
pre_lesion_seg="${SUBJECTS_DIR}/${SUBJ_ID}/mri/_prelesion.aseg.auto_noCCseg.mgz"
if [ ! -s "$pre_lesion_seg" ] ||
    [ "$t1" -nt "$pre_lesion_seg" ] ||
    [ "$t2" -nt "$pre_lesion_seg" ]; then

    rm -rvf "${SUBJECTS_DIR:?}/${SUBJ_ID:?}"
    set +e
    recon-all -autorecon1 -gcareg -canorm -careg -calabel \
        -subjid $SUBJ_ID \
        -i "$t1" \
        -wsatlas \
        -gcut \
        -3T \
        -expert "$FREESURFER_EXPERT_OPTIONS" \
        -time -openmp $N_PROCS -parallel -threads $N_PROCS
    set -e
    cp "${SUBJECTS_DIR}/${SUBJ_ID}/mri/aseg.auto_noCCseg.mgz" "$pre_lesion_seg"

    # Remove the auto aseg and the presurf aseg, as they will be re-generated later with
    # lesion segmentations included.
    rm -v "${FREESURFER_SAMPLE_DIR}/mri/aseg.auto.mgz" \
        "${FREESURFER_SAMPLE_DIR}/mri/aseg.presurf.mgz"

else
    echo "****** $SUBJ_ID | Already completed initial recon-all segmentation******"
fi

# 5. Run SPM lesion segmentation (LGA) to get a lesion segmentation mask.
lst_lesion_prob_map="$SPM_LST_DIR/ples_lga_${LGA_KAPPA}_rmflair.nii.gz"
if [ ! -s "$lst_lesion_prob_map" ] ||
    [ "$t1" -nt "$lst_lesion_prob_map" ] ||
    [ "$flair" -nt "$lst_lesion_prob_map" ]; then

    # Create temporary .nii variants of the original, raw T1 and FLAIR volumes; LGA
    # will perform bias correction and registration to the T1.
    lst_t1="${SPM_LST_DIR}/t1.nii"
    lst_flair="${SPM_LST_DIR}/flair.nii"
    mrconvert -quiet -force "$t1" "$lst_t1"
    mrconvert -quiet -force "$flair" "$lst_flair"

    lga_output="$SPM_LST_DIR/$(basename "$lst_lesion_prob_map" .nii.gz).nii"
    matlab -nojvm \
        -sd "$SCRIPT_DIR" \
        -batch \
        "spm_lst_lesion_seg('$SPM_ROOT_DIR', '$lst_t1', '$lst_flair', $LGA_KAPPA, 1, 150)"
    # Compress LGA output.
    mrconvert "$lga_output" "$lst_lesion_prob_map"

    rm -v "$lga_output"
    # Remove the bias-corrected, registered flair image.
    rm -vf "${SPM_LST_DIR}/rmflair.nii"
    # Remove temporary uncompressed T1 and FLAIR images.
    rm -vf "$lst_t1" "$lst_flair"
    rm -vf "${SPM_LST_DIR}/*.mat"
else
    echo "****** $SUBJ_ID | Already completed SPM LGA******"
fi
lst_lesion_mask="$SPM_LST_DIR/lga_kappa-${LGA_KAPPA}_p-thresh-${LGA_PROB_THRESH}_reg-t1w_lesion_mask.nii.gz"
if [ ! -s "$lst_lesion_mask" ] || [ "$lst_lesion_prob_map" -nt "$lst_lesion_mask" ]; then
    mrthreshold -quiet -force "$lst_lesion_prob_map" -abs $LGA_PROB_THRESH \
        "$lst_lesion_mask"
fi

# 6. Merge LGA lesion segmentation with recon-all segmentation and replicate the
# `calabel` steps in autorecon2.
reconall_lga_merged="${FREESURFER_SAMPLE_DIR}/mri/_calabel-lga_aseg-noCCseg_merged.mgz"
reconall_aseg_nocc_orig_output="${FREESURFER_SAMPLE_DIR}/mri/aseg.auto_noCCseg.mgz"
reconall_aseg_calabel_orig_output="${FREESURFER_SAMPLE_DIR}/mri/aseg.presurf.mgz"

if [ ! -s "$reconall_lga_merged" ] ||
    [ ! -s "$reconall_aseg_nocc_orig_output" ] ||
    [ ! -s "$reconall_aseg_calabel_orig_output" ] ||
    [ "$lst_lesion_mask" -nt "$reconall_lga_merged" ] ||
    [ "$pre_lesion_seg" -nt "$reconall_lga_merged" ]; then

    rm -fv "$reconall_aseg_nocc_orig_output" "$reconall_aseg_calabel_orig_output"

    lga_lesion_mask_fs_format="${FREESURFER_SAMPLE_DIR}/mri/_$(basename "$lst_lesion_mask" .nii.gz)_freesurfer_space.mgz"
    mrgrid "$lst_lesion_mask" regrid -force \
        -template "$pre_lesion_seg" \
        -strides "$pre_lesion_seg" \
        -interp nearest \
        "$lga_lesion_mask_fs_format"
    # These must be performed in two steps with an output to a .nii.gz due to some bug
    # (or unexpected problem) in mrtrix regarding .mgz files.
    mrcalc -force "$lga_lesion_mask_fs_format" $FS_LESION_LABEL "$pre_lesion_seg" -if "${FREESURFER_SAMPLE_DIR}/tmp/tmp-merge.nii.gz"
    mrconvert -force "${FREESURFER_SAMPLE_DIR}/tmp/tmp-merge.nii.gz" "$reconall_lga_merged"
    cp "$reconall_lga_merged" "${FREESURFER_SAMPLE_DIR}/mri/aseg.auto_noCCseg.mgz"

    mri_cc -lta "${FREESURFER_SAMPLE_DIR}/mri/transforms/cc_up.lta" -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz $SUBJ_ID
    cp "${FREESURFER_SAMPLE_DIR}/mri/aseg.auto.mgz" "${FREESURFER_SAMPLE_DIR}/mri/aseg.presurf.mgz"
else
    echo "****** $SUBJ_ID | Already completed lga-fs merge and recreation of freesurfer 'calabel'******"
fi

# 7. Re-run parts of autorecon2 and run all of autorecon3
manual_calabel_output="${FREESURFER_SAMPLE_DIR}/mri/aseg.presurf.mgz"
final_fs_seg_output="${SUBJECTS_DIR}/${SUBJ_ID}/mri/aseg.mgz"
if [ ! -s "$final_fs_seg_output" ] ||
    [ "$manual_calabel_output" -nt "$final_fs_seg_output" ]; then

    recon-all -subjid $SUBJ_ID \
        -T2 "$t2" \
        -autorecon2-noaseg \
        -autorecon3 \
        -T2pial \
        -openmp $N_PROCS -parallel -threads $N_PROCS
else
    echo "****** $SUBJ_ID | Already completed freesurfer recon-all autorecon2 and autorecon3******"
fi

# 10. Run brainstem and thalamic nuclei segmentations.
brainstem_seg="${FREESURFER_SAMPLE_DIR}/mri/brainstemSsLabels.v13.FSvoxelSpace.mgz"
if [ ! -s "$brainstem_seg" ] || [ "$final_fs_seg_output" -nt "$brainstem_seg" ]; then

    segmentBS.sh $SUBJ_ID

else
    echo "****** $SUBJ_ID | Already segmented brainstem structures******"
fi

# Thalamus & amygdala segmentation.
THAL_AMG_SEG_ID="t1w+t2w_n4"
lh_thal_amg_seg="${FREESURFER_SAMPLE_DIR}/mri/lh.hippoAmygLabels-T1-${THAL_AMG_SEG_ID}.v22.CA.FSvoxelSpace.mgz"
rh_thal_amg_seg="${FREESURFER_SAMPLE_DIR}/mri/rh.hippoAmygLabels-T1-${THAL_AMG_SEG_ID}.v22.CA.FSvoxelSpace.mgz"
thal_amg_t1="${FREESURFER_SAMPLE_DIR}/mri/nu.mgz"
# Extra t2w does not have to be registered to the T1, and the program will register
# the T2 regardless if we've done it already, so may as well only push the T2 through
# one interpolation instead of 2 by only giving it the de-biased version (unregistered)
thal_amg_t2="$t2_debiased"

if [ ! -s "$lh_thal_amg_seg" ] ||
    [ ! -s "$rh_thal_amg_seg" ] ||
    [ "$thal_amg_t1" -nt "$lh_thal_amg_seg" ] ||
    [ "$thal_amg_t2" -nt "$lh_thal_amg_seg" ] ||
    [ "$thal_amg_t1" -nt "$rh_thal_amg_seg" ] ||
    [ "$thal_amg_t2" -nt "$rh_thal_amg_seg" ]; then

    segmentHA_T2.sh $SUBJ_ID "$thal_amg_t2" $THAL_AMG_SEG_ID 1

else
    echo "****** $SUBJ_ID | Already segmented amygdala + hippocampus + thalamus structures******"
fi

# 11. Create anatomical image mask from the freesurfer brain mask.
brain_mask="${ANAT_OUT_DIR}/anat_mask.nii.gz"
fs_brain_mask="${FREESURFER_SAMPLE_DIR}/mri/brainmask.mgz"
template_vol="$t1_denoised"

if [ ! -s "$brain_mask" ] || [ "$fs_brain_mask" -nt "$brain_mask" ]; then

    # Resample to match the native subject FOV (should only involve cropping, no
    # interpolation from freesurfer -> native anatomical space).
    # Then threshold the mask, and fill in small holes.
    mrgrid "$fs_brain_mask" regrid \
        -template "$template_vol" \
        -interp nearest \
        -strides 1,2,3 \
        -datatype uint8 \
        - |
        mrthreshold \
            - \
            -abs 0.0001 \
            - |
        maskfilter \
            - \
            dilate -npass 2 \
            - |
        maskfilter --force \
            - \
            erode -npass 2 \
            "$brain_mask"

else
    echo "****** $SUBJ_ID | Already extracted binary brain mask******"
fi
# Set the brain mask to the template volume, as the strides are not RAS.
template_vol="$brain_mask"

# 11. Copy & extract freesurfer segmentations into the final output directory.
LUT="${FREESURFER_HOME}/luts/FreeSurferColorLUT.txt"
seg_files=(
    "aparc.a2009s+aseg.mgz"
    "aparc.DKTatlas+aseg.mgz"
    "aparc+aseg.mgz"
    "aseg.mgz"
    "brainstemSsLabels.v13.FSvoxelSpace.mgz"
    "hippoAmygLabels-T1-${THAL_AMG_SEG_ID}.v22.CA.FSvoxelSpace.mgz"
    "hippoAmygLabels-T1-${THAL_AMG_SEG_ID}.v22.FS60.FSvoxelSpace.mgz"
    "hippoAmygLabels-T1-${THAL_AMG_SEG_ID}.v22.HBT.FSvoxelSpace.mgz"
    "wmparc.mgz"
)

for seg_f in "${seg_files[@]}"; do

    # The hippocampus/amygdala/thalamus segmentation splits the labels into lh and rh,
    # so they must be joined back together. Luckily, they are mutually exclusive
    # positionally.
    if [[ $seg_f == *"hippoAmygLabels"* ]]; then
        joined_input_seg="${FREESURFER_SAMPLE_DIR}/mri/_lh+rh.${seg_f}"
        lh="${FREESURFER_SAMPLE_DIR}/mri/lh.${seg_f}"
        rh="${FREESURFER_SAMPLE_DIR}/mri/rh.${seg_f}"
        if [ ! -s "$joined_input_seg" ] ||
            [ "$lh" -nt "$joined_input_seg" ] ||
            [ "$rh" -nt "$joined_input_seg" ]; then

            mrcalc --force "$lh" "$rh" -add "$joined_input_seg"
        fi
        input_seg_path="$joined_input_seg"
        # Remove the T1+segmentation id string, as that is confusing and we only
        # plan on doing one segmentation configuration.
        seg_name="$(basename "$seg_f" .mgz)"
        seg_name=${seg_name//"-T1-${THAL_AMG_SEG_ID}"/}
    else
        seg_name="$(basename "$seg_f" .mgz)"
        input_seg_path="${FREESURFER_SAMPLE_DIR}/mri/${seg_f}"
    fi

    seg_dir="${ANAT_OUT_DIR}/freesurfer_segmentations/${seg_name}"
    mkdir --parents "$seg_dir"
    output_seg="${seg_dir}/${seg_name}.nii.gz"

    if [ ! -s "$output_seg" ] || [ "$input_seg_path" -nt "$output_seg" ]; then
        tmp_seg="${seg_dir}/.tmp.${seg_name}.nii.gz"
        mrgrid --force -info "$input_seg_path" regrid \
            -template "$template_vol" \
            -interp nearest \
            -strides 1,2,3 \
            -datatype int32 \
            "$tmp_seg"

        mkdir --parents "$seg_dir/roi_masks"
        "${SCRIPT_DIR}/split_freesurfer_segmentations_to_masks.py" \
            --fs_seg="$tmp_seg" \
            --lut="$LUT" \
            --output_dir="$seg_dir/roi_masks"
        mv "$tmp_seg" "$output_seg"
    else
        echo "****** $SUBJ_ID | Already converted and split $seg_name into masks******"
    fi
done

final_lesion_mask="${ANAT_OUT_DIR}/ms_lesion_mask.nii.gz"
source_mask="${ANAT_OUT_DIR}/freesurfer_segmentations/aseg/roi_masks/0${FS_LESION_LABEL}_Lesion.nii.gz"

cp --archive --update "$source_mask" "$final_lesion_mask"

# 12. Copy and mask all modality images that were bias-corrected into the
# final output directory.
vol_files=(
    "$t1_denoised"
    "$t2_denoised_reg_t1"
    "$flair_denoised_reg_t1"
)
dest_files=(
    "${ANAT_OUT_DIR}/t1w_brain.nii.gz"
    "${ANAT_OUT_DIR}/t2w_brain.nii.gz"
    "${ANAT_OUT_DIR}/flair_brain.nii.gz"
)
for ((i = 0; i < ${#vol_files[@]}; i++)); do

    vol_f="${vol_files[$i]}"
    dest_f="${dest_files[$i]}"

    if [ ! -s "$dest_f" ] || [ "$vol_f" -nt "$dest_f" ] || [ "$brain_mask" -nt "$dest_f" ]; then
        mrcalc \
            "$vol_f" \
            "$brain_mask" \
            -mult \
            - |
            mrconvert --force \
                - \
                -strides "$template_vol" \
                "$dest_f"
    else
        echo "****** $SUBJ_ID | Already masked volume $(basename "$dest_f" .nii.gz) ******"
    fi

done

# Finally, copy this script and software details to the subject directory.
code_dir="$(realpath "${SUBJECTS_DIR}/..")"/code/structural_images/
mkdir --parents "$code_dir"
cp --archive --update "${SCRIPT_DIR}"/* "$code_dir"
