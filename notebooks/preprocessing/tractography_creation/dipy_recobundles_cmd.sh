#!/usr/bin/bash

set -eou pipefail

dipy_recobundles \
        atlas-space/whole_brain_MNI_ifod2-msmt-act-gmwmi-seed_whole-brain-tractography_sift-filter_10000000-tracks__ifod2-sift-filter-10000000_MNI-space.trk \
        "/data/srv/data/pitn/atlas/Atlas_30_Bundles/bundles/*.trk" \
        --greater_than 50 \
        --slr_matrix "huge" \
        --clust_thr 15 \
        --model_clust_thr 2.5 \
        --reduction_thr 15 \
        --reduction_distance "mdf" \
        --reduction_thr 15 \
        --slr_metric "symmetric" \
        --slr_transform "similarity" \
        --refine \
        --r_reduction_thr 12 \
        --r_pruning_thr 6 \
        --out_dir "atlas-space/bundles" \
        --mix_names --log_level "INFO" --log_file "dipy_recobundles_log.txt"
