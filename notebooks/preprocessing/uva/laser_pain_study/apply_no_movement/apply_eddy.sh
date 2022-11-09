#!/bin/bash

set -eou pipefail

SUBJ_ID="sub-002"
SESSION_ID="ses-001"
# DATA_DIR="/mnt/storage/data/pitn/uva/liu_laser_pain_study/dry-run/$SUBJ_ID/$SESSION_ID"
OUT_DIR="/srv/tmp/data/pitn/uva/liu_laser_pain_study/derivatives"
TOPUP_DIR="$OUT_DIR/$SUBJ_ID/topup"
FULL_EDDY_DIR="$OUT_DIR/$SUBJ_ID/eddy"
EDDY_APPLY_NO_MOVE_DIR="$OUT_DIR/$SUBJ_ID/eddy_apply_no_move"
B0_BASENAME="${SUBJ_ID}_${SESSION_ID}_b0"
COMBINE_BASENAME="${SUBJ_ID}_${SESSION_ID}_AP-PA_DWI"

# Bring over necessary files from the full eddy run.
RAW_PARAMS_DIR="$EDDY_APPLY_NO_MOVE_DIR/raw_params"
mkdir -p "$RAW_PARAMS_DIR"

rsync -au "$FULL_EDDY_DIR/eddy_full_run.eddy_parameters" \
    "$RAW_PARAMS_DIR/eddy_full_run.eddy_parameters"
rsync -au "$FULL_EDDY_DIR/eddy_full_run.eddy_mbs_first_order_fields.nii" \
    "$RAW_PARAMS_DIR/eddy_full_run.eddy_mbs_first_order_fields.nii"
rsync -au "$FULL_EDDY_DIR/eddy_full_run.eddy_movement_over_time" \
    "$RAW_PARAMS_DIR/eddy_full_run.eddy_movement_over_time"
rsync -au "$FULL_EDDY_DIR/${COMBINE_BASENAME}_eddy_out.log" \
    "$RAW_PARAMS_DIR/${COMBINE_BASENAME}_eddy_out.log"
rsync -au \
    "$FULL_EDDY_DIR/eddy_full_run.eddy_post_eddy_shell_alignment_parameters" \
    "$RAW_PARAMS_DIR/eddy_full_run.eddy_post_eddy_shell_alignment_parameters"

# Convert params with python
./pre_eddy_params.py \
    --in-param-dir="$RAW_PARAMS_DIR" \
    --motion-ec-suffix="eddy_parameters" \
    --mbs-suffix="eddy_mbs_first_order_fields.nii" \
    --s2v-suffix="eddy_movement_over_time" \
    --eddy-log-suffix="out.log" \
    --input-dwi="$FULL_EDDY_DIR/eddy_full_run.eddy_outlier_free_data.nii" \
    --peas-suffix="eddy_post_eddy_shell_alignment_parameters" \
    --acq-params="$TOPUP_DIR/acqparams.txt" \
    --slspec="$FULL_EDDY_DIR/eddy_slspec.txt" \
    --bvals="$FULL_EDDY_DIR/${COMBINE_BASENAME}.bval" \
    --out-dir="$EDDY_APPLY_NO_MOVE_DIR"

# Call "no iteration" eddy with augmented input parameters.
docker run \
    --rm --gpus=all -it --ipc=host \
    --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
    --volume /mnt/storage/data:/mnt/storage/data \
    --volume /srv/tmp:/srv/tmp \
    tylerspears/fsl:cuda10.2 \
    eddy_cuda \
    --imain="$FULL_EDDY_DIR/eddy_full_run.eddy_outlier_free_data.nii" \
    --field="$TOPUP_DIR/${B0_BASENAME}_topup_field" \
    --mask="$FULL_EDDY_DIR/${B0_BASENAME}_bet_mask.nii" \
    --acqp="$TOPUP_DIR/acqparams.txt" \
    --slspec="$FULL_EDDY_DIR/eddy_slspec.txt" \
    --bvals="$FULL_EDDY_DIR/${COMBINE_BASENAME}.bval" \
    --bvecs="$FULL_EDDY_DIR/${COMBINE_BASENAME}.bvec" \
    --index="$FULL_EDDY_DIR/eddy_index.txt" \
    --flm=quadratic `#First model for eddy curr. distortion estimation.` \
    --estimate_move_by_susceptibility \
    --niter=0 `#Number of eddy iterations` \
    --s2v_niter=0 \
    --mbs_niter=0 \
    --init="$EDDY_APPLY_NO_MOVE_DIR/pre_computed.eddy_parameters.txt" \
    --init_s2v="$EDDY_APPLY_NO_MOVE_DIR/pre_computed.s2v_movement_parameters.txt" \
    --init_mbs="$EDDY_APPLY_NO_MOVE_DIR/pre_computed.eddy_mbs_first_order_fields.nii" \
    --repol \
    --hypar="$(cat $EDDY_APPLY_NO_MOVE_DIR/pre_computed.gp_hyperparams.txt)" \
    --dont_peas \
    --dont_sep_offs_move \
    --initrand=2985 \
    --dfields `#Write total displacement fields` \
    --fields `#Write EC fields as images`  \
    --data_is_shelled --history --very_verbose \
    --out="$EDDY_APPLY_NO_MOVE_DIR/eddy_apply_no_move" 2>&1 | tee "$EDDY_APPLY_NO_MOVE_DIR/${COMBINE_BASENAME}_eddy_out.log"

sync
mkdir -p "$EDDY_APPLY_NO_MOVE_DIR/displacement_fields/"
mv -f "$EDDY_APPLY_NO_MOVE_DIR"/eddy_apply_no_move.eddy_displacement_fields.* "$EDDY_APPLY_NO_MOVE_DIR/displacement_fields/"
    # --s2v_lambda=2  `#Regularization of within-vol motion correction; set slightly higher due to higher-order motion parameters.` \
    # --mporder=8 `#Order of within-vol motion correction equation.`  \
    # `#--slm=linear Suggested when there are <60 directions or sphere is not sampled properly.`\
    # --ol_nstd=999 `#Functionally disables outlier removal/detection.` \
