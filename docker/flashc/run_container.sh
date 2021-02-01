#!/bin/sh
echo "$1"
if test -z "$1"
                # --rm 
then
        docker run \
                --gpus=all --ipc=shareable \
                flashc:v0
else
        docker run \
                --gpus=all --ipc=shareable \
                flashc:v0 $1
fi