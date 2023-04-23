#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
T1="$2"
T2="$3"
FLAIR="$4"

N_PROCS="$(nproc)"
export SUBJECTS_DIR="$(pwd)/${SUBJ_ID}/freesurfer"
EXPERTS_FILE="${SUBJECTS_DIR}/expert_options.txt"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

# According to the `samseg` command, the following steps will run samseg and incorporate
# its output into freesurfer's recon-all.
# 1. Register the input volumes, run the actual `run_samseg` suite, compute segstats,
# and prepare the subject directory for recon-all. This can all be done with only
# the `samseg` command, and certain flags.

# Have to 'cheat' the lesion-mask-pattern that samseg expects, but is incompatible with
# run_samseg.
samseg \
    --s $SUBJ_ID \
    --sd "$SUBJECTS_DIR" \
    --t1w "$T1" \
    --t2w "$T2" \
    --flair "$FLAIR" \
    --refmode t1w \
    --fill \
    --normalization2 \
    --pallidum-separate \
    --lesion \
    --lesion-mask-pattern 0 "1 1"  \
    --threads $N_PROCS

# 2. Run recon-all with the -autorecon2-samseg and -autorecon3 flags.
# The -nosegmentation, -nofill, and -nonormalization2 flags are set in recon-all because
# samseg takes their place, as given in the '--fill' and '--normalization2' flags in
# samseg.
recon-all -autorecon2-samseg -autorecon3 \
        -subjid $SUBJ_ID \
        -nosegmentation \
        -nofill \
        -nonormalization2 \
        -T2pial \
        -wsless \
        -norm1-n 1 \
        -norm2-n 1 \
        -norm3diters 1 \
        -expert "$EXPERTS_FILE" \
        -3T \
        -time -openmp $N_PROCS -parallel -threads $N_PROCS

# Run extra segmentation of brainstem, amygdala, and thalamus regions.
segmentBS.sh $SUBJ_ID
segmentHA_T1.sh $SUBJ_ID

######### Scratch & experimental commands

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
