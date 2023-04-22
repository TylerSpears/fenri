#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
T1="$2"
T2="$3"
FLAIR="$4"

N_PROCS="$(nproc)"
SAMSEG_DIR="$(pwd)/samseg/"
SUBJECTS_DIR="$(pwd)/freesurfer/"
EXPERTS_FILE="${SUBJECTS_DIR}/freesurfer/expert_options.txt"

# 1. Rigid register (unprocessed!) FLAIR and T2 to T1 with ANTS
mkdir --parents "$SAMSEG_DIR"
SAMSEG_REG_DIR="${SAMSEG_DIR}/reg_t1"
mkdir --parents "$SAMSEG_REG_DIR"
# Full registration command with bspline interpolation.
# Register T2 to T1
"${ANTSHOME}/antsRegistration" --verbose 1 \
    --dimensionality 3 \
    --float 0 \
    --collapse-output-transforms 1 \
    --output "[ ${SAMSEG_REG_DIR}/t2_reg_t1_,${SAMSEG_REG_DIR}/t2_reg_t1.nii.gz,${SAMSEG_REG_DIR}/t1_reg_t2.nii.gz ]" \
    --interpolation BSpline \
    --use-histogram-matching 0 \
    --winsorize-image-intensities "[ 0.005,0.995 ]" \
    --initial-moving-transform "[ $T1,$T2,1 ]" \
    --transform "Rigid[ 0.1 ]" \
    --metric "MI[ $T1,$T2,1,32,Regular,0.25 ]" \
    --convergence "[ 1000x500x250x100,1e-6,10 ]" \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox
T2_REG_T1="${SAMSEG_REG_DIR}/t2_reg_t1.nii.gz"

# Register FLAIR to T1
"${ANTSHOME}/antsRegistration" --verbose 1 \
    --dimensionality 3 \
    --float 0 \
    --collapse-output-transforms 1 \
    --output "[ ${SAMSEG_REG_DIR}/flair_reg_t1_,${SAMSEG_REG_DIR}/flair_reg_t1.nii.gz,${SAMSEG_REG_DIR}/t1_reg_flair.nii.gz ]" \
    --interpolation BSpline \
    --use-histogram-matching 0 \
    --winsorize-image-intensities "[ 0.005,0.995 ]" \
    --initial-moving-transform "[ $T1,$FLAIR,1 ]" \
    --transform "Rigid[ 0.1 ]" \
    --metric "MI[ $T1,$FLAIR,1,32,Regular,0.25 ]" \
    --convergence "[ 1000x500x250x100,1e-6,10 ]" \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox
FLAIR_REG_T1="${SAMSEG_REG_DIR}/flair_reg_t1.nii.gz"

# 2. Run `run_samseg` with lesion segmentation, using FLAIR and T2 as extra contrasts
mkdir --parents "${SAMSEG_DIR}/results"
run_samseg \
        --input "$T1" "$T2_REG_T1" "$FLAIR_REG_T1" \
        --mode t1_ t2_ flair_ \
        --pallidum_separate \
        --lesion \
        --lesion_mask_pattern 0 1 1 \
        --output "${SAMSEG_DIR}/results" \
        --threads $N_PROCS

# 3. Run recon-all as before (with the T2 instead of FLAIR, as the T2 has slightly higher spatial resolution)
# recon-all -all \
#         -subjid $SUBJ_ID \
#         -i "$T1" \
#         -T2 "$T2" \
#         -subcortseg \
#         -expert "expert_options.txt" \
#         -3T \
#         -time -openmp 5 -parallel -threads 5

# 4. Replace (merge?) 'aseg.presurf.mgz' with the samseg-generated 'seg.mgz'

# 5. Run `recon-all -autorecon2-noaseg -subjid [subj_id]` to re-run the autorecon2 steps

# 6. Run `recon-all -autorecon-pial ...` to re-create pial surface estimation, atlas segmentations, etc.
#       This essentially just re-runs autorecon3

# 7. Process T2 image with the new aseg.mgz to preserve lesions.

# Register & correct T2 image to match T1 & FLAIR volumes.
# BASE_DIR="${SUBJECTS_DIR}/${SUBJ_ID}"
# mri_convert --no_scale 1 \
#         "$FLAIR" \
#         "${BASE_DIR}/mri/orig/T2raw.mgz"
# bbregister --s $SUBJ_ID \
#         --mov "${BASE_DIR}/mri/orig/T2raw.mgz" \
#         --lta "${BASE_DIR}/mri/transforms/T2raw.auto.lta" \
#         --init-coreg --T2 --gm-proj-abs 2 --wm-proj-abs 1 --no-coreg-ref-mask
# cp "${BASE_DIR}/mri/transforms/T2raw.auto.lta" \
#         "${BASE_DIR}/mri/transforms/T2raw.lta"
# mri_convert -odt float \
#         --apply_transform "${BASE_DIR}/mri/transforms/T2raw.lta" \
#         --resample_type cubic \
#         --no_scale 1 \
#         --reslice_like "${BASE_DIR}/mri/brainmask.mgz" \
#         "${BASE_DIR}/mri/orig/T2raw.mgz" \
#         "${BASE_DIR}/mri/T2.prenorm.mgz"
# mri_normalize -seed 1234 -sigma 0.5 -nonmax_suppress 0 -min_dist 1 \
#         -aseg "${BASE_DIR}/mri/aseg.presurf.mgz" \
#         -surface "${BASE_DIR}/surf/lh.white" identity.nofile \
#         -surface "${BASE_DIR}/surf/rh.white" identity.nofile \
#         `#Add options from expert_options.txt file that would usually go into recon-all.` \
#         -gentle \
#         -nosnr \
#         -n 1 \
#         "${BASE_DIR}/mri/T2.prenorm.mgz" \
#         "${BASE_DIR}/mri/T2.norm.mgz"
# mri_mask -transfer 255 \
#         -keep_mask_deletion_edits \
#         "${BASE_DIR}/mri/T2.norm.mgz" \
#         "${BASE_DIR}/mri/brain.finalsurfs.mgz" \
#         "${BASE_DIR}/mri/T2.mgz"

# Run extra segmentation of brainstem, amygdala, and thalamus regions.
segmentBS.sh $SUBJ_ID
segmentHA_T1.sh $SUBJ_ID
