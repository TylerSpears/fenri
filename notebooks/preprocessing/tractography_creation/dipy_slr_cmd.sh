#!/usr/bin/bash

set -eou pipefail

dipy_slr --num_threads 18 \
        /data/srv/data/pitn/atlas/Atlas_30_Bundles/whole_brain/whole_brain_MNI.trk \
        ../ifod2-msmt-act-gmwmi-seed_whole-brain-tractography_sift-filter_10000000-tracks.trk \
        --x0 "affine" \
        --rm_small_clusters 100 \
        --progressive `#Refers to progressive reg.: transl.->rigid->...` \
        --greater_than 50 \
        --less_than 300 \
        --nb_pts 200 \
        --out_dir "atlas-space/" \
        --out_moved "ifod2-sift-filter-10000000_MNI-space.trk" \
        --out_affine "ifod2-sift-filter-10000000_subj-to-MNI-space_affine.txt" \
        --out_stat_centroids "static_centroids.trk" \
        --out_moving_centroids "moving_centroids.trk" \
        --out_moved_centroids "moved_centroids.trk" \
        --mix_names --log_level "INFO" --log_file "dipy_slr_log.txt"
