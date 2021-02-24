#!/bin/bash

tag=6.0.4
#DOCKER_BUILDKIT=1 docker build -t fsl:$tag .
docker build -t fsl:$tag .
#docker tag brainlife/fsl brainlife/fsl:$tag
#docker push brainlife/fsl:$tag
