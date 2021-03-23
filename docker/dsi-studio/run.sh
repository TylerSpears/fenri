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
    --user $UID:$GROUPS \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    -- \
    dsi-studio:latest
