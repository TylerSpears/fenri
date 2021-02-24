
docker run \
    --rm \
    --ipc=host \
    --volume /srv/tmp:/srv/tmp \
    --volume /media/tyler/data:/srv/data \
    fsl:6.0.4 "$@"


#x11docker \
#    --hostipc \
#    --interactive \
#    --clipboard \
#    --user=RETAIN \
#    --hostdisplay \
#    --group-add video --group-add render \
#    -- \
#    --rm \
#    --interactive --tty \
#    --ipc=host \
#    --volume /srv/tmp:/srv/tmp \
#    --volume /media/tyler/data:/srv/data \
#    -- \
#    fsl:6.0.4 "$@"

