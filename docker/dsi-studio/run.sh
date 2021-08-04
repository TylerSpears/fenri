#!/bin/bash

TMP_DATA_DIR="${TMP_DATA_DIR:-/srv/tmp}"
DATA_DIR="${DATA_DIR:-/srv/data}"
DSI_CONFIG_DIR="${DSI_CONFIG_DIR:-$HOME/.config/dsi-studio}"
DSI_CACHE_DIR="${DSI_CACHE_DIR:-$HOME/.cache/dsi-studio}"

mkdir --parents $DSI_CONFIG_DIR $DSI_CACHE_DIR

x11docker \
    --hostipc \
    --clipboard \
    --user=RETAIN \
    --gpu \
    --iglx \
    --hostdisplay \
    --group-add video --group-add render \
    --runasroot "ldconfig" \
    --user $UID:$GROUPS \
    -- \
    --rm \
    --ipc=host \
    --volume "$DSI_CONFIG_DIR":/home/guest/.config \
    --volume "$DSI_CACHE_DIR":/home/guest/.cache \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    -- \
    dsi-studio:latest
