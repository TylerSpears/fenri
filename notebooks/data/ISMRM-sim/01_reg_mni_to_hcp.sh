#!/usr/bin/bash

set -eou pipefail

LPS_MNI_T1W_F="$1"
LPS_MNI_MASK_F="$2"
MNI_PREFIX="$3"
SUBJ_T1W_F="$4"
SUBJ_MASK_F="$5"
SUBJ_PREFIX="$6"
OUTPUT_DIR="$7"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

mkdir --parents "$OUTPUT_DIR"

ants_prefix="${OUTPUT_DIR}/${MNI_PREFIX}_reg_${SUBJ_PREFIX}_"

# Convert subject T1w and mask into LPS, to simplify the assumptions made in ITK/ants
lps_subj_t1w_f="${OUTPUT_DIR}/lps_${SUBJ_PREFIX}_t1w.nii.gz"
mrconvert "$SUBJ_T1W_F" -strides "$LPS_MNI_T1W_F" "$lps_subj_t1w_f"
lps_subj_mask_f="${OUTPUT_DIR}/lps_${SUBJ_PREFIX}_brain_mask.nii.gz"
mrconvert "$SUBJ_MASK_F" -strides "$LPS_MNI_T1W_F" "$lps_subj_mask_f"

# Register MNI to the HCP subject.
# Try the full-resolution registration
"${ANTSPATH}/antsRegistrationSyN.sh" -d 3 \
    -m "$LPS_MNI_T1W_F" \
    -f "$lps_subj_t1w_f" \
    -x "$lps_subj_mask_f","$LPS_MNI_MASK_F" \
    -t s \
    -j 1 \
    -p d \
    -o "$ants_prefix" \
    -z 1
