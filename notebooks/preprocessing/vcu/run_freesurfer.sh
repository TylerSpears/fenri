#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
T1="$2"
T2="$3"
FLAIR="$4"

SUBJECTS_DIR="$(pwd)"
recon-all -all \
        -subjid $SUBJ_ID \
        -i "$T1" \
        -T2 "$T2" \
        -subcortseg \
        -expert "expert_options.txt" \
        -3T \
        -time -openmp 5 -parallel -threads 5

# New MS-included pipeline:
# 1. Rigid register (unprocessed!) FLAIR and T2 to T1 with ANTS
# 2. Run `run_samseg` with lesion segmentation, using FLAIR and T2 as extra contrasts
# 3. Run recon-all as before (with the T2 instead of FLAIR, as the T2 has slightly higher spatial resolution)
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
