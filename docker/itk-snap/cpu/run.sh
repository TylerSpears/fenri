ITK_HOME_DIR="$HOME/.local/share/x11docker/itk-snap"
mkdir -p $ITK_HOME_DIR

x11docker \
    --hostipc \
    --runasroot='ldconfig' \
    --hostdbus \
    --clipboard \
    --user=RETAIN \
    --hostdisplay \
    --group-add video --group-add render \
    -- \
    --rm \
    --device=/dev/dri:/dev/dri \
    --net=host \
    --volume "$ITK_HOME_DIR":"/home/itksnap/" \
    --volume /srv/data:/srv/data \
    --volume /srv/tmp:/srv/tmp \
    -- \
    tylerspears/itk-snap:cpu
