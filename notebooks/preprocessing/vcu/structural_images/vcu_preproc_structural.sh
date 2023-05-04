#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
t1="$2"
t2="$3"
flair="$4"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
N_PROCS="$(nproc)"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS

LGA_KAPPA="0.075"
LGA_PROB_THRESH="0.25"

export SUBJECTS_DIR="$(pwd)/${SUBJ_ID}/freesurfer"
PRE_FS_DIR="${SUBJECTS_DIR}/pre_freesurfer"
mkdir --parents "$PRE_FS_DIR"
# NU_CORRECT_DIR="${SUBJECTS_DIR}/nu_debiased"
# mkdir --parents "$NU_CORRECT_DIR"
# REG_DIR="${SUBJECTS_DIR}/reg_t1"
# mkdir --parents "$REG_DIR"
SPM_LST_DIR="${SUBJECTS_DIR}/spm-lst_lga"
mkdir --parents "$SPM_LST_DIR"
SPM_ROOT_DIR="/opt/spm"

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

    "${ANTSHOME}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$t1" \
        --shrink-factor 3 \
        --output [ "$t1_debiased","${DEBIAS_DIR}/t1_bias_field.nii.gz" ]

    "${ANTSHOME}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$t2" \
        --shrink-factor 3 \
        --output [ "$t2_debiased","${DEBIAS_DIR}/t2_bias_field.nii.gz" ]

    "${ANTSHOME}/N4BiasFieldCorrection" --verbose 1 \
        -d 3 \
        --input-image "$flair" \
        --shrink-factor 3 \
        --output [ "$flair_debiased","${DEBIAS_DIR}/flair_bias_field.nii.gz" ]
else
    echo "******Already completed image de-biasing******"
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
    "${ANTSHOME}/antsRegistration" --verbose 1 \
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
    echo "******Already completed T2 -> T1 registration******"
fi

# Register FLAIR to T1
flair_reg_t1="${REG_DIR}/flair_reg_t1.nii.gz"
flair_reg_t1_tf="${REG_DIR}/flair_reg_t1_0GenericAffine.mat"
if [ ! -s "$flair_reg_t1" ] ||
    [ ! -s "$flair_reg_t1_tf" ] ||
    [ "$t1_debiased" -nt "$flair_reg_t1" ] ||
    [ "$flair_debiased" -nt "$flair_reg_t1" ]; then
    "${ANTSHOME}/antsRegistration" --verbose 1 \
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
    echo "******Already completed FLAIR -> T1 registration******"
fi

# 1. Run recon-all with the original T1 and T2, up to the creation of aseg.auto.no_CC 
# for the initial GM labels and nu-corrected T1 volume.
pre_lesion_seg="${SUBJECTS_DIR}/mri/_prelesion.aseg.presurf.mgz"
if [ ! -s "$pre_lesion_seg" ] ||
    [ "$t1" -nt "$pre_lesion_seg" ] ||
    [ "$t2" -nt "$pre_lesion_seg" ]; then
    recon-all -autorecon1 -gcareg -canorm -careg -calabel \
        -subjid $SUBJ_ID \
        -i "$t1" \
        -T2 "$t2" \
        -T2pial \
        -wsatlas \
        -gcut \
        -3T \
        -time -openmp $N_PROCS -parallel -threads $N_PROCS

    cp "${SUBJECTS_DIR}/mri/aseg.presurf.mgz" "$pre_lesion_seg"
fi

# 3. Run SPM lesion segmentation (LGA) to get a complimentary lesion segmentation mask.
# Export samseg output images for use in SPM.

lst_lesion_prob_map="$SPM_LST_DIR/ples_lga_${LGA_KAPPA}_rmflair_bias_corrected.nii"
if [ ! -s "$lst_lesion_prob_map" ] ||
    [ "$t1_debiased" -nt "$lst_lesion_prob_map" ] ||
    [ "$flair_reg_t1" -nt "$lst_lesion_prob_map" ]; then

    # Create temporary .nii variants of the debiased, registered t1 and flair images.
    lst_t1="${SPM_LST_DIR}/t1.nii"
    lst_flair="${SPM_LST_DIR}/flair.nii"
    mrconvert -quiet -force "$t1_debiased" "$lst_t1"
    mrconvert -quiet -force "$flair_reg_t1" "$lst_flair"

    matlab -nojvm \
        -sd "$SCRIPT_DIR" \
        -batch \
        "spm_lst_lesion_seg('$SPM_ROOT_DIR', '$lst_t1', '$lst_flair', $LGA_KAPPA, 1, 100)"

    rm "$lst_t1" "$lst_flair"
else
    echo "******Already completed SPM LGA******"
fi
lst_lesion_mask="$SPM_LST_DIR/lga_kappa-${LGA_KAPPA}_p-thresh-${LGA_PROB_THRESH}_lesion_mask.nii.gz"
mrthreshold -quiet -force "$lst_lesion_prob_map" -abs $LGA_PROB_THRESH -datatype uint8 \
    "$lst_lesion_mask"

# 4. Merge samseg and LGA lesion segmentations.
# Use mrtrix 5ttgen command to map the samseg/freesurfer labels to a gray matter mask.
# We can't necessarily trust the white matter masks here, due to false negatives in the
# lesion segmentation, but the GM labels should be unaffected.

# 5. Initialize the recon-all subject directory with only the first few steps of
# autorecon1.

# 6. Copy over the necessary files from samseg to recon-all needed to maintain
# consistency between the two.
# - talairach transformation: samseg.talairach.lta -> talairach.lta
# - talairach template: template.m3z -> talairach.m3z
# - samseg segmentation file, with only 99, 258, 259, 165 labels remaining.
#   -> aseg.manedit.mgz

# 7. Continue recon-all autorecon1 and autorecon2

# 8. Merge samseg lesion segmentation with recon-all segmentation.

# 9. Re-run autorecon2 and run autorecon3 with `-autorecon2-noaseg -autorecon3`

# 10. Run brainstem and thalamic nuclei segmentations.
# segmentBS.sh $SUBJ_ID
# segmentHA_T1.sh $SUBJ_ID

# 11. Copy & extract freesurfer segmentations into the final output directory.

# 12. Copy and mask all modality images that were bias-corrected by samseg into the
# final output directory.
