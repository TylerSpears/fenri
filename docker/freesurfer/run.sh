#!/bin/bash
IMAGE="vnmd/freesurfer_7.1.1:20210421"
#docker pull "$IMAGE"
docker run --rm -it --ipc host \
    --user root \
    --volume "/mnt/storage/data/pitn:/mnt/storage/data/pitn" \
    --volume "/srv/tmp/pitn:/srv/tmp/pitn" \
    --volume "/home/tas6hh/Projects:/home/neuro/Projects" \
    "$IMAGE"
