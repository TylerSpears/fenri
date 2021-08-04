#!/bin/bash

# Allow positional parameter for building either CPU or nvidia variants.
case $1 in
    nvidia)
        BASE_IMG="${BASE_IMG:-nvidia/opengl:1.2-glvnd-devel-ubuntu20.04}"
        BUILD_TAG="nvidia"
        ;;
    *)
        BASE_IMG="${BASE_IMG:-ubuntu/20.04}"
        BUILD_TAG="cpu"
        ;;
esac

docker buildx build  \
    -t tylerspears/dsi-studio:$BUILD_TAG \
    --build-arg BASE=$BASE_IMG \
    --file ./Dockerfile \
    .
