
x11docker \
    --hostipc \
    --clipboard \
    --user=RETAIN \
    --hostdisplay \
    --group-add video --group-add render \
    -- \
    --rm \
    --ipc=host \
    --volume /srv/tmp:/srv/tmp \
    --volume /media/tyler/data:/srv/data \
    -- \
    mytk:cpu
