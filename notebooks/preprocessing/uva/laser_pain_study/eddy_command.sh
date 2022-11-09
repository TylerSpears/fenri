#!/bin/bash

set -eou pipefail

SUBJ_ID="sub-001"
SESSION_ID="ses-002"
DATA_DIR="/mnt/storage/data/pitn/uva/liu_laser_pain_study/dry-run/$SUBJ_ID/$SESSION_ID"
OUT_DIR="/srv/tmp/data/pitn/uva/liu_laser_pain_study/derivatives"
TOPUP_DIR="$OUT_DIR/$SUBJ_ID/topup"
EDDY_DIR="$OUT_DIR/$SUBJ_ID/eddy"

# Reciprocal of 'BandwidthPerPixelPhaseEncode'
TOTAL_READOUT_TIME=0.0965997
# AP_BASENAME=$(basename "$(/usr/bin/find $TOPUP_DIR -maxdepth 1 -type f -iname '*dmri*AP*.nii.gz')")
AP_BASENAME="DRY-RUN_002_20211209081012_3_dMRI_SMS_98-directions_AP"
# PA_BASENAME=$(basename "$(/usr/bin/find $TOPUP_DIR -maxdepth 1 -type f -iname '*dmri*PA*.nii.gz')")
PA_BASENAME="DRY-RUN_002_20211209081012_4_dMRI_SMS_98-directions_PA"

# b-value threshold to be considered a "b0."
B0_THRESH=100
N_B0_SELECTIONS=3

### Select b0 volumes from both AP and PA.
B0_BASENAME="${SUBJ_ID}_${SESSION_ID}_b0"
# Create acquisition parameters text file.
rm --force "$TOPUP_DIR/acqparams.txt"
touch "$TOPUP_DIR/acqparams.txt"
# Extract AP b0s
echo "Extracting b0s from AP"
./extract_b0s.py \
    $B0_THRESH \
    $N_B0_SELECTIONS \
    "$DATA_DIR/$AP_BASENAME.nii.gz" \
    "$DATA_DIR/$AP_BASENAME.bvec" \
    "$DATA_DIR/$AP_BASENAME.bval" \
    "$TOPUP_DIR/${B0_BASENAME}_AP"

# write acquisitions
n_ap_b0s=$(docker run \
    --rm \
    --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
    --volume /mnt/storage/data:/mnt/storage/data \
    --volume /srv/tmp:/srv/tmp \
    tylerspears/fsl:cuda10.2 \
    fslnvols "$TOPUP_DIR/${B0_BASENAME}_AP.nii.gz")

for ((i = 0; i < n_ap_b0s; i++)); do
    echo "0 1 0 $TOTAL_READOUT_TIME" >>"${TOPUP_DIR}/acqparams.txt"
done

# Extract PA b0s
echo "Extracting b0s from PA"
./extract_b0s.py \
    $B0_THRESH \
    $N_B0_SELECTIONS \
    "$DATA_DIR/$PA_BASENAME.nii.gz" \
    "$DATA_DIR/$PA_BASENAME.bvec" \
    "$DATA_DIR/$PA_BASENAME.bval" \
    "$TOPUP_DIR/${B0_BASENAME}_PA"

# write acquisitions
n_pa_b0s=$(docker run \
    --rm \
    --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
    --volume /mnt/storage/data:/mnt/storage/data \
    --volume /srv/tmp:/srv/tmp \
    tylerspears/fsl:cuda10.2 \
    fslnvols "$TOPUP_DIR/${B0_BASENAME}_PA.nii.gz")
for ((i = 0; i < n_pa_b0s; i++)); do
    echo "0 -1 0 $TOTAL_READOUT_TIME" >>"$TOPUP_DIR/acqparams.txt"
done

# Merge AP-PA b0 volumes
echo "Merging AP-PA b0 volumes."
docker run \
    --rm -it \
    --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
    --volume /mnt/storage/data:/mnt/storage/data \
    --volume /srv/tmp:/srv/tmp \
    tylerspears/fsl:cuda10.2 \
    fslmerge \
    -t \
    "$TOPUP_DIR/${B0_BASENAME}_AP-PA.nii" \
    "$TOPUP_DIR/${B0_BASENAME}_AP.nii.gz" \
    "$TOPUP_DIR/${B0_BASENAME}_PA.nii.gz"

### Apply topup
# only run topup if one of the output files isn't present.
if [ ! -s "$TOPUP_DIR/${B0_BASENAME}_topup_fieldcoef.nii" ] || [ ! -s "$TOPUP_DIR/${B0_BASENAME}_topup_movpar.txt" ]; then
    echo "Running topup"

    # Apply topup
    docker run \
        --rm --gpus=all --ipc=host -it \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        topup \
        --imain="$TOPUP_DIR/${B0_BASENAME}_AP-PA" \
        --datain="$TOPUP_DIR/acqparams.txt" \
        --config="$TOPUP_DIR/topup_config.cnf" \
        --out="$TOPUP_DIR/${B0_BASENAME}_topup" \
        --fout="$TOPUP_DIR/${B0_BASENAME}_topup_field" \
        --iout="$TOPUP_DIR/${B0_BASENAME}_topup_corrected" \
        --verbose 2>&1 | tee "$TOPUP_DIR/${B0_BASENAME}_topup_out.log"
fi

### Extract the brain mask
# Only run BET if the mask doesn't exist, or the topup output is more recent than the
# generated mask.
if [ ! -s "$EDDY_DIR/${B0_BASENAME}_bet_mask.nii" ] || [ "$TOPUP_DIR/${B0_BASENAME}_topup_corrected.nii" -nt "$EDDY_DIR/${B0_BASENAME}_bet_mask.nii" ]; then
    echo "Computing mean b0 volume."
    # Generate binary mask of mean b0 images from topup correction.
    docker run \
        --rm -it \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        fslmaths \
        "$TOPUP_DIR/${B0_BASENAME}_topup_corrected.nii" \
        -Tmean \
        "$EDDY_DIR/${B0_BASENAME}_topup_corrected_mean.nii.gz"

    echo "Running BET"
    docker run \
        --rm --gpus=all --ipc=host -it \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        bet \
        "$EDDY_DIR/${B0_BASENAME}_topup_corrected_mean.nii.gz" \
        "$EDDY_DIR/${B0_BASENAME}_bet" -R \
        -m -v 2>&1 | tee "$EDDY_DIR/${B0_BASENAME}_bet_out.log"
fi


#### Run eddy correction
# Combine AP and PA DWIs
COMBINE_BASENAME="${SUBJ_ID}_${SESSION_ID}_AP-PA_DWI"
docker run \
    --rm -it \
    --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
    --volume /mnt/storage/data:/mnt/storage/data \
    --volume /srv/tmp:/srv/tmp \
    tylerspears/fsl:cuda10.2 \
    fslmerge \
    -t \
    "$EDDY_DIR/$COMBINE_BASENAME.nii" \
    "$DATA_DIR/$AP_BASENAME.nii.gz" \
    "$DATA_DIR/$PA_BASENAME.nii.gz"

# Combine bvals
paste -d " " "$DATA_DIR/$AP_BASENAME.bval" "$DATA_DIR/$PA_BASENAME.bval" > "$EDDY_DIR/$COMBINE_BASENAME.bval"
# Combine bvecs
paste -d " " "$DATA_DIR/$AP_BASENAME.bvec" "$DATA_DIR/$PA_BASENAME.bvec" > "$EDDY_DIR/$COMBINE_BASENAME.bvec"

# Create index file for indexing into acquisition parameters.
rm --force "$EDDY_DIR/eddy_index.txt"
touch "$EDDY_DIR/eddy_index.txt"
n_ap_dwis=$(
    docker run \
        --rm \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        fslnvols "$DATA_DIR/$AP_BASENAME.nii.gz" \
)
for ((i = 0; i < n_ap_dwis; i++)); do
    echo "1" >>"$EDDY_DIR/eddy_index.txt"
done

n_pa_dwis=$(
    docker run \
        --rm \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        fslnvols "$DATA_DIR/$PA_BASENAME.nii.gz" \
)
# Make sure the index corresponds to the acquisition parameter *after* all the
# AP acquisition parameters are noted.
for ((i=0; i < n_pa_dwis; i++)); do
    echo "$(( n_ap_b0s + 1 ))" >>"$EDDY_DIR/eddy_index.txt"
done

# Generate slspec.txt for eddy.
# Assume that the slice timings are the same for each AP-PA.
./calculate_slspec.py "$DATA_DIR/${AP_BASENAME}.json" "$EDDY_DIR/eddy_slspec.txt"

# Run eddy
if [ "$EDDY_DIR/${B0_BASENAME}_bet_mask.nii" -nt "$EDDY_DIR/eddy_full_run.nii" ]; then
    echo "Running eddy"

    docker run \
        --rm --gpus=all -it --ipc=host \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        eddy_cuda \
        --imain="$EDDY_DIR/$COMBINE_BASENAME.nii" \
        --mask="$EDDY_DIR/${B0_BASENAME}_bet_mask.nii" \
        --acqp="$TOPUP_DIR/acqparams.txt" \
        --index="$EDDY_DIR/eddy_index.txt" \
        --bvecs="$EDDY_DIR/$COMBINE_BASENAME.bvec" \
        --bvals="$EDDY_DIR/$COMBINE_BASENAME.bval" \
        --topup="$TOPUP_DIR/${B0_BASENAME}_topup" \
        --repol `#Replace outliers with GP predictions.` \
        --rep_noise `#Add noise to outlier replacements` \
        --flm=quadratic `#First model for eddy curr. distortion estimation.` \
        --slm=linear `#Suggested when there are <60 directions or sphere is not sampled properly.`\
        --mporder=8 `#Order of within-vol motion correction equation.`  \
        --s2v_lambda=2  `#Regularization of within-vol motion correction; set slightly higher due to higher-order motion parameters.` \
        --slspec="$EDDY_DIR/eddy_slspec.txt" `#Slice timing/group specification for SMS/Multi-Band acquisition.` \
        --ol_type=both `#Incorporates slice timing groups and within-group movements into outlier correction.` \
        --niter=10 `#Number of eddy iterations` \
        --fwhm=10,8,4,2,1,0,0,0,0,0 `#Filter Width...H?..in mm. Corresponds to a Gaussian filtering step before starting an eddy iteration, must have length=niter` \
        --estimate_move_by_susceptibility \
        --cnr_maps \
        --fields `#Write EC fields as images`  \
        --dfields `#Write total displacement fields` \
        --initrand=2985 `#Necessary for reproducing in the application of eddy parameters.` \
        --history --very_verbose \
        --out="$EDDY_DIR/eddy_full_run" 2>&1 | tee "$EDDY_DIR/${COMBINE_BASENAME}_eddy_out.log"

    # Run eddy quality check for reports.
    echo "Running eddy quality check"
    docker run \
        --rm \
        --volume /home/tas6hh/Projects/pitn/:/home/tas6hh/Projects/pitn \
        --volume /mnt/storage/data:/mnt/storage/data \
        --volume /srv/tmp:/srv/tmp \
        tylerspears/fsl:cuda10.2 \
        eddy_quad \
        "$EDDY_DIR/eddy_full_run" \
        --mask="$EDDY_DIR/${B0_BASENAME}_bet_mask.nii" \
        -par="$TOPUP_DIR/acqparams.txt" \
        -idx="$EDDY_DIR/eddy_index.txt" \
        --bvals="$EDDY_DIR/$COMBINE_BASENAME.bval" \
        --bvecs="$EDDY_DIR/$COMBINE_BASENAME.bvec" \
        --field="$TOPUP_DIR/${B0_BASENAME}_topup_field" \
        --slspec="$EDDY_DIR/eddy_slspec.txt" \
        --output-dir="$EDDY_DIR/eddy_quad_quality_control"
fi

# --data_is_shelled \
# return 1 2>/dev/null
# exit 1
