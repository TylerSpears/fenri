
x11docker \
    --hostipc \
    --clipboard \
    --user=RETAIN \
    --hostdisplay \
    --group-add video --group-add render \
    --gpu \
    --runtime=nvidia \
    -- \
    --rm \
    --gpus=all \
    --ipc=host \
    --volume /srv/data:/srv/data \
    --volume /srv/tmp:/srv/tmp \
    -- \
    mytk
