# nvidia/cuda:9.2-devel-ubuntu18.04
FROM nvidia/cuda@sha256:1ac22dab63629449eaadf3615994c88e8ac556137bd2af4e14c6c7d8c8daf86f
# Example build command.
# DOCKER_BUILDKIT=1 docker build -t flashc:v0 .

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    sudo git \
    swig libfftw3-dev libboost-dev cmake \
    libinsighttoolkit4-dev \
    python2.7-dev python-numpy python-scipy \
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
SHELL ["/bin/sh", "-c"]
######################################
# Default CMD, can be replaced with `docker run [options] flashc:v0 "otherexec --params"`
CMD ["nvidia-smi"]
USER ${USER}

# Set up PyCa
RUN git clone https://bitbucket.org/scicompanat/pyca.git pyca
# Apply patch from an unresolved pull request:
# <https://bitbucket.org/scicompanat/pyca/pull-requests/18>.
# This is necessary to compile with newer versions of gcc.
# See <https://stackoverflow.com/a/2949855/13225248> for the commands used.
WORKDIR /home/${USER}/pyca
RUN git remote add gcc_patch https://bitbucket.org/m-pilia/pyca.git \
    && git fetch gcc_patch \
    && git merge gcc_patch/master

# Prepare for compilation.
RUN mkdir pyca-bin
WORKDIR /home/${USER}/pyca/pyca-bin

# CUDA architecture version, dependent on the physical GPU you have installed.
# I'm not entirely sure how the these work with backwards-compatibility, and it should
# really be handled automatically. See
# <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>
# for a table of architecture to version mappings.
# The ideal way to handle this would be to use CMake's `FindCUDA` function, which
# detects this version automatically. But, that would require more work than I care to do...
ARG CUDA_ARCH_VERSION="52"
#ENV PYTHONPATH="~/python/site-packages:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV RUNTIME_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
ENV BUILD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs"
ENV LD_LIBRARY_PATH=${BUILD_LD_LIBRARY_PATH}

USER root
# Taken from <https://github.com/tensorflow/tensorflow/pull/44732>.
# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ( \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    || true \
    ) \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig
USER ${USER}

# https://github.com/opencv/opencv/issues/6577#issuecomment-226685773
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://askubuntu.com/questions/1157589/nvcc-fatal-value-sm-20-is-not-defined-for-option-gpu-architecture
RUN cmake \
    # -DCMAKE_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    -DCUDA_ARCH_VERSION=${CUDA_ARCH_VERSION} \
    -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS};--pre-include ${HOME}/pyca/preinc.h" \
    ../ \
    && make -j4

USER root
RUN make install
USER ${USER}
# Create empty response file to avoid survey questions, and clean up build packages.
RUN mkdir -p /home/${USER}/.PyCA && echo "" >> /home/${USER}/.PyCA/reportedStatistics.txt

# Run unit tests.
WORKDIR /home/${USER}/pyca/Testing
## Build-time DEBUG
# RUN sudo apt update && sudo apt install -y --no-install-recommends locate tree
# RUN echo ${LD_LIBRARY_PATH} \
#     && echo ${PATH} \
#     && sudo ldconfig \
#     && ls -al --human-readable /usr/local/cuda/lib64/stubs \
#     && (locate -c libcuda.so || true) \
#     && sudo tree -a -P "libcuda.so*" --prune / \
#     && cat /etc/ld.so.conf.d/nvidia.conf \
#     && exit 1
##
# Minimal error producing command.
# Until this bug is resolved (likely in `nvidia-docker`), this is just staying for
# future reference.

# RUN sudo sed -i "s/mname = '.'.join((pkg, '_Core')).lstrip('.')/mname = '.'.join((pkg, '_Core')).lstrip('.')\n        print (pkg, mname, __name__)/" /usr/local/lib/python2.7/dist-packages/PyCA/Core.py \
#     && head --lines 30 /usr/local/lib/python2.7/dist-packages/PyCA/Core.py| cat -n - \
#     && sudo ldconfig \
#     && printenv \
#     && python -c "import PyCA._Core; print PyCA._Core; exit()"
# RUN python -m unittest discover -v -p '*UnitTest.py'


WORKDIR /home/${USER}

#### Finished with PyCA compilation
#### Starting flashc compilation

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libarmadillo-dev libmpich-dev \
    && rm -rf /var/lib/apt/lists/*
USER ${USER}

RUN mkdir -p /home/${USER}/flashc \
    && git clone https://bitbucket.org/FlashC/flashc.git flashc

WORKDIR /home/${USER}/flashc
RUN cmake \
    -DPyCA_DIR="${HOME}/pyca/pyca-bin" \
    .
RUN make -j4

# Build cleanup.
RUN sudo rm /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=${RUNTIME_LD_LIBRARY_PATH}
WORKDIR /home/${USER}
