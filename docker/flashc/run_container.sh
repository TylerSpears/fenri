#!/bin/sh
echo "$1"
if test -z "$1"
then
        docker run \
                --rm --gpus=all --ipc=shareable \
                flashc:v0
else
        docker run \
                --rm --gpus=all --ipc=shareable \
                flashc:v0 $1
fi
