#!/bin/bash

set -eou pipefail

RCLONE_REMOTE="hcp_s3"

target_dir="/data/srv/data/pitn/hcp"
filter_file="hcp_rclone_filter.txt"
# ids_file="HCP_unique_ids.txt"
# ids_file="HCP_split_04-05_new_ids.txt"
ids_file="HCP_7T_subsample_idx.txt"

while IFS= read -r line || [[ -n "$line" ]]; do
    subj_id=$line
    echo "$subj_id"
    mkdir --parents "$target_dir/$subj_id/T1w/"
    rclone copy --progress --filter-from "$filter_file" \
        $RCLONE_REMOTE:hcp-openaccess/HCP_1200/"$subj_id"/T1w "$target_dir/$subj_id/T1w/"
done < "$ids_file"
