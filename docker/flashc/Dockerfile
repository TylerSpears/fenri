# FROM nvidia/cuda@sha256:2b02390a7988c592578466083a1353fa8bb815167f05c44282014c60044352c6
# nvidia/cuda:9.2-devel-ubuntu18.04
FROM nvidia/cuda@sha256:1ac22dab63629449eaadf3615994c88e8ac556137bd2af4e14c6c7d8c8daf86f
# Example build command.
# DOCKER_BUILDKIT=1 docker build -t flashc:v0 .
# libinsighttoolkit4-dev \
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo git \
    swig libfftw3-dev libboost-dev cmake \ 
    python2.7-dev python-numpy \
    && rm -rf /var/lib/apt/lists/*

ARG USER=guest
RUN adduser --disabled-password --gecos '' ${USER} \
    && adduser ${USER} sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/${USER}

######################################
# Setup for the ldconfig workaround.
# Only applicable if you are using nvidia-docker2 on Debian Testing or Experimental
# (which is roughly Debian 11, at the time of writing).
# If you have a different system, you should be able to comment out this block.
SHELL ["/bin/bash", "-c"]
ENV ENTRYPOINT_WRAPPER="/home/${USER}/entrypoint.sh"
# Create an executable script that wraps the "CMD" executable(s).
RUN touch "${ENTRYPOINT_WRAPPER}" \
    && chmod +rx ${ENTRYPOINT_WRAPPER} \
    && echo \
    $'#!/bin/bash \n\
    set -e\n\
    sudo --non-interactive -u root ldconfig \n\
    exec "$@"\n' \
    >> ${ENTRYPOINT_WRAPPER}
ENTRYPOINT ["./entrypoint.sh"]
######################################

# Default CMD, can be replaced with `docker run [options] flashc:v0 "otherexec --params"`
CMD ["nvidia-smi"]
USER ${USER}

# Set up PyCa
RUN git clone https://bitbucket.org/scicompanat/pyca.git pyca \
    && mkdir -p pyca/pyca-bin

WORKDIR /home/${USER}/pyca/pyca-bin

# CUDA architecture version, dependent on the physical GPU you have installed.
# I'm not entirely sure how the these work with backwards-compatibility, and it should
# really be handled automatically. See
# <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>
# for a table of architecture to version mappings.
# The ideal way to handle this would be to use CMake's `FindCUDA` function, which 
# detects this version automatically. But, that would require more than I care to do...
ARG CUDA_ARCH_VERSION="52"

# https://github.com/opencv/opencv/issues/6577#issuecomment-226685773
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://askubuntu.com/questions/1157589/nvcc-fatal-value-sm-20-is-not-defined-for-option-gpu-architecture
RUN cmake \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    -DCUDA_ARCH_VERSION=${CUDA_ARCH_VERSION} \
    ../

RUN \
    make \
    && make install


WORKDIR /home/${USER}
