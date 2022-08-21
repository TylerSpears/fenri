#!/bin/bash

set -euo pipefail

docker buildx build \
    -t tylerspears/fsl-cuda10.2:6.0.5 \
    -f ./fsl-cuda10.2.Dockerfile \
    .
