FROM ubuntu:20.04

#https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux

ARG USER=guest

RUN apt update && \
    apt install -y --no-install-recommends \
    wget \
    ca-certificates \
    sudo \
    python \
    dc \
    jq \
    python-numpy \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

#install fsl, but get rid of src
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    python fslinstaller.py --downloadonly --quiet -d /usr/local/fsl -V 6.0.4 && \
    python fslinstaller.py -d /usr/local/fsl -V 6.0.4 && \
    rm -rf /usr/local/fsl/src

ENV FSLDIR=/usr/local/fsl
ENV PATH=$PATH:$FSLDIR/bin
ENV LD_LIBRARY_PATH=$FSLDIR/lib
ENV FSLOUTPUTTYPE=NIFTI_GZ

RUN adduser --disabled-password --gecos '' ${USER} \
    && adduser ${USER} sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && echo 'source ${FSLDIR}/etc/fslconf/fsl.sh' >> ${HOME}/.bashrc

USER ${USER}
WORKDIR /home/${USER}

SHELL ["/bin/bash", "-c"]
