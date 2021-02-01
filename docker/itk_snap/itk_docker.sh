
x11docker \
        --gpu \
        --home \
        --clipboard \
        --interactive \
        --pull=ask \
        --runtime nvidia \
        --hostipc \
        -- \
        --env QT_GRAPHICSSYSTEM="native" \
        --env DISPLAY=unix$DISPLAY \
        --env NVIDIA_VISIBLE_DEVICES="all" \
        --env NVIDIA_DRIVER_CAPABILITIES="graphics,utility,compute" \
        --gpus all \
        -- \
        dorianps/itksnap@sha256:17aaead20d9f56778b662e8cf549453df6340e9d3cf1402abcda75d7ee6c7685

# docker run --rm -it \
#        --privileged \
#        --env QT_GRAPHICSSYSTEM="native" \
#        --env DISPLAY=unix$DISPLAY \
#        --env NVIDIA_VISIBLE_DEVICES="all" \
#        --env NVIDIA_DRIVER_CAPABILITIES="graphics,utility,compute" \
#        --volume /tmp/.X11-unix:/tmp/.X11-unix \
#        --volume $XAUTHORITY:$XAUTHORITY \
#        --ipc host \
#        --gpus all \
#        dorianps/itksnap@sha256:17aaead20d9f56778b662e8cf549453df6340e9d3cf1402abcda75d7ee6c7685
