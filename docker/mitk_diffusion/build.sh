#!/usr/bin/bash

set -eou pipefail

docker buildx build \
        -t tylerspears/mitk-diffusion \
        -f Dockerfile \
        .
