#!/bin/bash

# Quick script for pulling The correct T1 images from HCP amazon s3 instance.
# ! This will only work on my dali.cpe.virginia.edu! Locations and the rclone remote name
# are worthless everywhere else.

for f in [0-9]*
do
        subj_id="$f"
        echo Subject "$subj_id"
        target_d="${subj_id}/T1w"
        echo Target "$target_d"
        if [ ! -e "${target_d}/T1w_acpc_dc_restore_brain.nii.gz" ]
        then
                hcp_dataset="HCP"
                source_f="hcp-openaccess/${hcp_dataset}/${subj_id}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
                echo Source "$source_f"
                rclone copy -P --no-traverse hcp_s3:"$source_f" "$target_d" && test -e "${target_d}/T1w_acpc_dc_restore_brain.nii.gz"
                download_success=$?
                echo Success? "$download_success"
                if [ $download_success -gt 0 ]
                then
                        echo "no"
                        hcp_dataset="HCP_1200"
                        source_f="hcp-openaccess/$hcp_dataset/$subj_id/T1w/T1w_acpc_dc_restore_brain.nii.gz"

                        echo Source "$source_f"
                        rclone copy -P --no-traverse hcp_s3:"$source_f" "$target_d"
                fi
        fi
        echo "======"

done
