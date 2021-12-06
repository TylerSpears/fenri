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
    --gpu \
    --iglx \
    --hostdbus \
    --network \
    --hostipc \
    --clipboard \
    --shell=/bin/bash \
    --no-entrypoint \
    --runtime=$DSI_RUNTIME \
    --hostdisplay \
    --runasroot "ldconfig" \
    --user $UID:$GROUPS \
    --newprivileges=yes \
    --sudouser=nopasswd \
    --group-add video --group-add render --group-add sudo \
    --workdir=/home/guest \
    --env HOME="/home/guest" \
    -- \
    --rm \
    -it \
    --volume "$DSI_CONFIG_DIR":/home/guest/.config \
    --volume "$DSI_CACHE_DIR":/home/guest/.cache \
    --volume "$LOCAL_DIR":/home/guest/.local/share/dsi-studio \
    --volume "$TMP_DATA_DIR":/srv/tmp \
    --volume "$DATA_DIR":/srv/data \
    -- \
    tylerspears/dsi-studio:$IMG_TAG bash

