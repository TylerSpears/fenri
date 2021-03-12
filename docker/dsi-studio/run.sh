#!/bin/sh

TMP_DATA_DIR="/srv/tmp"
DATA_DIR="/srv/data"

x11docker \
    --hostipc \
    --clipboard \
    --user=RETAIN \
    --hostdisplay \
    --group-add video --group-add render \
    --runasroot "ldconfig" \
    -- \
    --rm \
    --ipc=host \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    --env NVIDIA_VISIBLE_DEVICES="all" \
    --env NVIDIA_DRIVER_CAPABILITIES="graphics,utility,compute" \
    -- \
    mysi:latest
