#!/bin/bash

# Quick script for pulling dmris from HCP amazon s3 instance.
# ! This will only work on my dali.cpe.virginia.edu! Locations and the rclone remote name
# are worthless everywhere else.
set -uo pipefail

hcp_dataset="HCP_1200"
subj_id_file="/home/tas6hh/Projects/pitn/notebooks/data/extra_hcp_subj_ids.txt"
exec 4<"$subj_id_file"
#for f in [0-9]*
while read -u4 line
do
    subj_id="$line"
    echo Subject "$subj_id"
    target_d="${subj_id}/T1w"
    echo Target "$target_d"
    # T1w
    source_f="hcp-openaccess/${hcp_dataset}/${subj_id}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
    echo src "$source_f"
    rclone copy -P --no-traverse hcp_s3:"$source_f" "$target_d" && test -e "${target_d}/T1w_acpc_dc_restore_brain.nii.gz"
    download_success=$?
    echo Success? "$download_success"

    # T2w
    source_f="hcp-openaccess/${hcp_dataset}/${subj_id}/T1w/T2w_acpc_dc_restore_brain.nii.gz"
    echo src "$source_f"
    rclone copy -P --no-traverse hcp_s3:"$source_f" "$target_d" && test -e "${target_d}/T2w_acpc_dc_restore_brain.nii.gz"
    download_success=$?
    echo Success? "$download_success"

    # Structural mask
    source_f="hcp-openaccess/${hcp_dataset}/${subj_id}/T1w/brainmask_fs.nii.gz"
    echo src "$source_f"
    rclone copy -P --no-traverse hcp_s3:"$source_f" "$target_d" && test -e "${target_d}/brainmask_fs.nii.gz"
    download_success=$?
    echo Success? "$download_success"

    # diffusion
    target_d="${subj_id}/T1w/Diffusion"
    source_dir="hcp-openaccess/${hcp_dataset}/${subj_id}/T1w/Diffusion"
    echo src "$source_dir"
    for diff_file in 'data.nii.gz' 'bvals' 'bvecs' 'nodif_brain_mask.nii.gz'
    do
        rclone copy -P --no-traverse hcp_s3:"${source_dir}/${diff_file}" "${target_d}" && test -e "${target_d}/${diff_file}"
        #rclone lsf hcp_s3:"${source_dir}/${diff_file}"
        download_success=$?
        echo Success? "$download_success"
    done
    echo "======"
done
