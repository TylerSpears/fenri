#!/bin/bash
set -eou pipefail

SUBJECTS_DIR="$(pwd)/subj_dir" recon-all \
        -subjid sub-004 \
        -i "/home/tas6hh/Projects/_sandbox/liu_laser_pain_study/sub-004/data/sub-004_ses-001_t1w-mprage.nii.gz" \
        -T2 "/home/tas6hh/Projects/_sandbox/liu_laser_pain_study/sub-004/data/sub-004_ses-001_t2w.nii.gz" \
        -expert "$(pwd)/expert_options.txt" \
        -all -3T -mprage \
        -time -openmp 13 -parallel -threads 13

SUBJECTS_DIR="$(pwd)/subj_dir" mri_mask \
    "$SUBJECTS_DIR/sub-004/mri/T2.norm.mgz" \
    "$SUBJECTS_DIR/sub-004/mri/brainmask.mgz" \
    "$SUBJECTS_DIR/sub-004/mri/T2.masked.mgz"
