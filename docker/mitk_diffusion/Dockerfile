FROM ubuntu:20.04
ARG MITK_DIFFUSION_URL=https://www.mitk.org/download/diffusion/nightly/MITK-Diffusion_ubuntu-20.04_2023.08.21_a754b053_32d7d08a_NoPython.tar.gz

RUN apt-get update --yes && apt-get install -yq --no-install-recommends \
        wget \
        libglu1-mesa \
        libfreetype6 \
        libtiff5 \
        libgomp1 && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir --parents /opt/MITK-Diffusion
WORKDIR /opt/MITK-Diffusion
RUN wget "$MITK_DIFFUSION_URL" \
        --no-check-certificate \
        --output-document MITK-Diffusion_bin.tar.gz && \
        tar xf MITK-Diffusion_bin.tar.gz --directory . && \
        rm -rf MITK-Diffusion_bin.tar.gz && \
        mv MITK-Diffusion-[0-9][0-9][0-9][0-9].[0-9]*/* . && \
        rmdir MITK-Diffusion-[0-9][0-9][0-9][0-9].[0-9]*

ENV PATH="/opt/MITK-Diffusion:${PATH}"
