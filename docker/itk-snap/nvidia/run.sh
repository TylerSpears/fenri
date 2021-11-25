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
    --gpu \
    --runtime=nvidia \
    -- \
    --rm \
    --net=host \
    --volume "$ITK_HOME_DIR":"/home/itksnap/" \
    --volume /srv/data:/srv/data \
    --volume /srv/tmp:/srv/tmp \
    -- \
    tylerspears/itk-snap:nvidia
