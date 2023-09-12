#!/usr/bin/bash

set -eou pipefail

docker buildx build \
        -t tylerspears/mitk-diffusion:intel-build \
        -f intel-build.Dockerfile \
        .
