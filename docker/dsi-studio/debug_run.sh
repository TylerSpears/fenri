#!/bin/bash

TMP_DATA_DIR="${TMP_DATA_DIR:-/srv/tmp}"
DATA_DIR="${DATA_DIR:-/srv/data}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/.local/share/dsi-studio}"
DSI_CONFIG_DIR="${DSI_CONFIG_DIR:-$HOME/.config/dsi-studio}"
DSI_CACHE_DIR="${DSI_CACHE_DIR:-$HOME/.cache/dsi-studio}"

mkdir --parents $DSI_CONFIG_DIR $DSI_CACHE_DIR $LOCAL_DIR

case $1 in
    nvidia)
        DSI_RUNTIME="nvidia"
        IMG_TAG="nvidia"
        ;;
    *)
        DSI_RUNTIME="runc"
        IMG_TAG="cpu"
        ;;
esac


x11docker \
    --showenv \
    --interactive \
    --shell=/bin/bash \
    --sudouser=nopasswd \
    --no-entrypoint \
    --hostipc \
    --clipboard \
    --gpu \
    --runtime=$DSI_RUNTIME \
    --hostdisplay \
    --group-add video --group-add render \
    --runasroot "ldconfig" \
    -- \
    --rm \
    -it \
    --ipc=host \
    --volume "$DSI_CONFIG_DIR":/home/guest/.config \
    --volume "$DSI_CACHE_DIR":/home/guest/.cache \
    --volume "$LOCAL_DIR":/home/guest/.local/share/dsi-studio \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    -- \
    tylerspears/dsi-studio:$IMG_TAG /bin/bash
