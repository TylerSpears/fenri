#!/bin/bash
#IMAGE="vnmd/freesurfer_7.1.1:20210421"
IMAGE="vnmd/freesurfer_6.0.0:20210917"
#docker pull "$IMAGE"
docker run --rm -it --ipc host \
    --user root \
    --env SUBJECTS_DIR="/srv/tmp/pitn/oasis3/workspace" \
    --volume "/mnt/storage/data/pitn:/mnt/storage/data/pitn" \
    --volume "/srv/tmp/pitn:/srv/tmp/pitn" \
    --volume "/home/tas6hh/Projects:/home/neuro/Projects" \
    --mount type=bind,source="/srv/tmp/pitn/license.txt",target="/opt/freesurfer-6.0.0/license.txt" \
    "$IMAGE"
