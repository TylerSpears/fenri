#!/bin/bash
set -e

# Must export all variables for substitution later.
export FREESURFER_SUBJ_ID="$1"
echo "Starting $FREESURFER_SUBJ_ID"

export SUBJECT_ID="`expr match "$FREESURFER_SUBJ_ID" '\(OAS[0-9]*\)'`"
export SESSION_ID="`expr match "$FREESURFER_SUBJ_ID" '.*\(d[0-9]*\)'`"

export SUBJECTS_DATA_DIR="/srv/tmp/pitn/oasis3"
export SUBJECTS_DIR="${SUBJECTS_DATA_DIR}/workspace"

export SUBJ_SOURCE_DIR="${SUBJECTS_DATA_DIR}/sub-${SUBJECT_ID}/ses-${SESSION_ID}"
export SUBJ_FREESURFER_SOURCE_DIR="${SUBJECTS_DATA_DIR}/derivatives/freesurfer/${FREESURFER_SUBJ_ID}"
export SUBJ_FREESURFER_TARGET_DIR="${SUBJECTS_DIR}/${FREESURFER_SUBJ_ID}"

# Remove this freesurfer subject directory, if it exists. Otherwise, freesurfer will
# fail.
sudo rm -rf $SUBJ_FREESURFER_TARGET_DIR

export T1_INPUT="`/usr/bin/find "${SUBJ_SOURCE_DIR}" -type f -iname "**_T1w.nii.gz" | head -1`"
export T2_INPUT="`/usr/bin/find "${SUBJ_SOURCE_DIR}" -type f -iname "**_T2w.nii.gz" | head -1`"

export THREADS=4

# Set up docker image
export IMAGE="vnmd/freesurfer_6.0.0:20210917"

# Encapsulate the command as a string to feed to the docker container's 'bash -c'.
docker_cmd=$(cat <<-END
`# Step 1 - Build initial images + T2`
echo `whoami` &&
recon-all
    -autorecon1
    -threads $THREADS -openmp $THREADS
    -s $FREESURFER_SUBJ_ID
    -i "$T1_INPUT"
    -T2 "$T2_INPUT"
&&
cd "$SUBJ_FREESURFER_TARGET_DIR"
&&
`# Step 2 - Copy all surface and label files into the newly-created subject directory`
cp -n -R "${SUBJ_FREESURFER_SOURCE_DIR}/surf" "${SUBJ_FREESURFER_SOURCE_DIR}/label"
    $SUBJ_FREESURFER_TARGET_DIR
&&
`# Copy aseg.auto.mgz to be used for T2 normalization.`
cp -n -R "${SUBJ_FREESURFER_SOURCE_DIR}/mri/aseg.auto.mgz"
    "${SUBJ_FREESURFER_TARGET_DIR}/mri"
&&
`# Step 3 - Register T2 to previous T1`
bbregister --s $FREESURFER_SUBJ_ID --mov mri/orig/T2raw.mgz
    --lta mri/transforms/T2raw.lta
    --init-rr --T2
&&
`# Step 4 - Normalize T2`
mri_convert -odt float
    -at mri/transforms/T2raw.lta -rt cubic -ns 1 -rl
    mri/orig.mgz mri/orig/T2raw.mgz mri/T2.prenorm.mgz
&&
mri_normalize -sigma 0.5 -nonmax_suppress 0 -min_dist 1
    -aseg mri/aseg.auto.mgz
    -surface surf/rh.white identity.nofile -surface surf/lh.white identity.nofile
    mri/T2.prenorm.mgz mri/T2.norm.mgz
&&
`# Step 5 - Mask T2 with T1s calculated mask`
mri_mask mri/T2.norm.mgz mri/brainmask.mgz mri/T2.mgz
END
)

#docker_cmd=$(cat <<-END
#echo "why"
#&& cd "$SUBJ_FREESURFER_TARGET_DIR"
#END
#\)
# Perform variable substitution on the command string.
docker_cmd=$(echo $docker_cmd | envsubst)
# Reflect on your life's choices that led you to do such a terrible thing.

# Run the actual command with docker.
docker run --rm -t --ipc host \
    --user root \
    --workdir $SUBJECTS_DIR \
    --env SUBJECTS_DIR=$SUBJECTS_DIR \
    --volume "/mnt/storage/data/pitn:/mnt/storage/data/pitn" \
    --volume "/srv/tmp/pitn:/srv/tmp/pitn" \
    --volume "/home/tas6hh/Projects:/home/neuro/Projects" \
    --mount type=bind,source="/srv/tmp/pitn/license.txt",target="/opt/freesurfer-6.0.0/license.txt" \
    "$IMAGE" \
    /usr/bin/env bash -c "$docker_cmd"
