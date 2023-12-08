#!/usr/bin/bash

set -eou pipefail

# There's a ton of command line parameters, but there's not a great way of doing it
# otherwise without assuming filenames...
# An example loop may look like:
# ```bash
# while IFS="" read -r subj || [ -n "$subj" ]; do \
    # printf '==========================================%s\n' "$subj"
    # ismrm_reg_dir="../ismrm-2015_reg_mni-152"
    # hcp_reg_dir="${subj}/01_mni_reg_hcp"
    # mkdir -p $subj/02_warp_creation
    # ANTSPATH="/opt/ants/ants-2.4.3/bin/" ../02_create_ismrm_to_hcp_warps.sh \
        # ../lps_ismrm-2015_t1w.nii.gz  \
        # "${ismrm_reg_dir}/ismrm-2015_reg_mni-152_0GenericAffine.mat" \
        # "${ismrm_reg_dir}/ismrm-2015_reg_mni-152_1Warp.nii.gz" \
        # "${ismrm_reg_dir}/ismrm-2015_reg_mni-152_1InverseWarp.nii.gz" \
        # "ismrm-2015" \
        # "${hcp_reg_dir}/lps_hcp-${subj}_t1w.nii.gz" \
        # "${hcp_reg_dir}/mni-152_reg_hcp-${subj}_0GenericAffine.mat" \
        # "${hcp_reg_dir}/mni-152_reg_hcp-${subj}_1Warp.nii.gz" \
        # "${hcp_reg_dir}/mni-152_reg_hcp-${subj}_1InverseWarp.nii.gz" \
        # "hcp-${subj}" \
        # "${subj}/02_warp_creation" || break
# done < HCP_sub_ids.txt
# ```

ISMRM_TEMPLATE_F="$1"
ISMRM2MNI_AFFINE_F="$2"
ISMRM2MNI_WARP_F="$3"
ISMRM2MNI_INV_WARP_F="$4"
ISMRM_PREFIX="$5"
HCP_TEMPLATE_F="$6"
MNI2HCP_AFFINE_F="$7"
MNI2HCP_WARP_F="$8"
MNI2HCP_INV_WARP_F="$9"
HCP_SUBJ_PREFIX="${10}"
OUTPUT_DIR="${11}"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

mkdir --parents "$OUTPUT_DIR"

# We need both push-forward and pullback warps, for warping tracks and images,
# respectively.
# See <https://community.mrtrix.org/t/registration-using-transformations-generated-from-other-packages/2259>
# for details on composing ants transforms for mrtrix.

# Push-forward warp
push_warp_prefix="${ISMRM_PREFIX}_to_${HCP_SUBJ_PREFIX}_push-forward_warp"
warpinit "$HCP_TEMPLATE_F" "${OUTPUT_DIR}/${push_warp_prefix}_identity[].nii.gz" \
    -force -nthreads $N_PROCS
for i in {0..2}; do
    "${ANTSPATH}/antsApplyTransforms" -v -d 3 -e 0 \
        --input "${OUTPUT_DIR}/${push_warp_prefix}_identity${i}.nii.gz" \
        --output "${OUTPUT_DIR}/mrtrix_${push_warp_prefix}${i}.nii.gz" \
        --reference-image "$ISMRM_TEMPLATE_F" \
        --interpolation Linear \
        --transform [ "$ISMRM2MNI_AFFINE_F",1 ] \
        --transform "$ISMRM2MNI_INV_WARP_F" \
        --transform [ "$MNI2HCP_AFFINE_F",1 ] \
        --transform "$MNI2HCP_INV_WARP_F" \
        --default-value 2147483647
done
warpcorrect "${OUTPUT_DIR}/mrtrix_${push_warp_prefix}[].nii.gz" \
    "${OUTPUT_DIR}/mrtrix_${push_warp_prefix}_corrected.nii.gz" \
    -marker 2147483647 -force -nthreads $N_PROCS

# Pullback warp
pull_warp_prefix="${ISMRM_PREFIX}_to_${HCP_SUBJ_PREFIX}_pullback_warp"
warpinit "$ISMRM_TEMPLATE_F" "${OUTPUT_DIR}/${pull_warp_prefix}_identity[].nii.gz" \
    -force -nthreads $N_PROCS
for i in {0..2}; do
    "${ANTSPATH}/antsApplyTransforms" -v -d 3 -e 0 \
        --input "${OUTPUT_DIR}/${pull_warp_prefix}_identity${i}.nii.gz" \
        --output "${OUTPUT_DIR}/mrtrix_${pull_warp_prefix}${i}.nii.gz" \
        --reference-image "$HCP_TEMPLATE_F" \
        --interpolation Linear \
        --transform "$MNI2HCP_WARP_F" \
        --transform "$MNI2HCP_AFFINE_F" \
        --transform "$ISMRM2MNI_WARP_F" \
        --transform "$ISMRM2MNI_AFFINE_F" \
        --default-value 2147483647
done
warpcorrect "${OUTPUT_DIR}/mrtrix_${pull_warp_prefix}[].nii.gz" \
    "${OUTPUT_DIR}/mrtrix_${pull_warp_prefix}_corrected.nii.gz" \
    -marker 2147483647 -force -nthreads $N_PROCS
