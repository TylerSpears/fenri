# Dockerfile for adding opengl to an Ubuntu-based image with pre-built drivers.
# This is targeted towards CPU and AMD GPU rendering, but should also be compatible
# with nvidia hardware rendering.
# See this PPA for more information:
# https://launchpad.net/~oibaf/+archive/ubuntu/graphics-drivers

ARG BASE=ubuntu:20.04
ARG _USER=${USER}

FROM ${BASE}

USER root
RUN DEBIAN_FRONTEND=noninteractive \
    add-apt-repository ppa:oibaf/graphics-drivers && \
    apt-get update && \
    apt-get install --target-release oibaf/graphics-drivers mesa && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER ${USER}
