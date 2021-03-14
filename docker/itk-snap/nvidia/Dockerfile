# Example build command:
# `DOCKER_BUILDKIT=1 docker build -t mytk .`

# Switched to Qt5, but the Dockerfile structure is based upon the itksnap dockerfile at:
#dorianps/itksnap@sha256:17aaead20d9f56778b662e8cf549453df630e9d3cf1402abcda75d7ee6c7685

# Pre-built docker images with Qt5, which is non-trivial to contain in an image.
FROM theshadowx/qt5@sha256:18a1f0fb7b013a5f34ce015841d72a74baed54058532889f2d71d97517eaff9e

# Pre-made NVIDIA image for GPU-accelerated opengl
# <https://hub.docker.com/r/nvidia/opengl>.
# Requires `nvidia-docker2` (<https://github.com/NVIDIA/libnvidia-container>).
FROM nvidia/opengl@sha256:e572ae8c3a47547e2cc93635e22c2fb1d9586bde130bca617b8c1f3d8851ae63
ENV LD_LIBRARY_PATH=/opt/itksnap/lib/:${LD_LIBRARY_PATH}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

# Download ITK-SNAP.
# Install ca-certificates package to safely verify SourceForge's signature.
RUN apt-get update -y \
    && apt-get install -y \
    wget \
    ca-certificates \
    && wget -O itksnap.tar.gz \
    "https://sourceforge.net/projects/itk-snap/files/itk-snap/3.8.0/itksnap-3.8.0-20190612-Linux-x86_64.tar.gz/download" \
    && tar -zxf itksnap.tar.gz -C /opt/ \
    && mv /opt/itksnap-*/ /opt/itksnap/ \
    && rm itksnap.tar.gz \
    && useradd -m -s /bin/bash itksnap
ENV PATH="/opt/itksnap/bin/:${PATH}"

# Download libpng12 library. This library is not found in the default ubuntu repos, so
# it must be installed manually.
# See <https://www.linuxuprising.com/2018/05/fix-libpng12-0-missing-in-ubuntu-1804.html>.
# If this image is upgraded to Ubuntu 19 or higher, then this can also be installed from
# a PPA: <https://launchpad.net/%7Elinuxuprising/+archive/ubuntu/libpng12>.
RUN wget -O libpng12.deb \
    "https://launchpad.net/~ubuntu-security/+archive/ubuntu/ppa/+build/15108504/+files/libpng12-0_1.2.54-1ubuntu1.1_amd64.deb"

# Install remaining packages.
RUN apt-get install -y --no-install-recommends \
    mesa-utils \
    libglu1 \
    libcurl4-openssl-dev \
    libsm6 \
    libxt6 \
    libfreetype6 \
    libxrender1 \
    libfontconfig1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    && dpkg -i libpng12.deb \
    && rm libpng12.deb \
    && apt-get purge -y wget ca-certificates \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

# Set to avoid error by XKB. See
# <https://stackoverflow.com/questions/56353346/xkbcommon-error-failed-to-add-default-include-path>
ENV QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb

USER itksnap
COPY --chown=itksnap:itksnap UserPreferences.xml /home/itksnap/.itksnap.org/ITK-SNAP/
CMD ["itksnap"]

# OpenGL debugging commands. Uncomment to build an image that runs glmark2.
#USER root
#RUN apt update && apt install -y --no-install-recommends \
#glmark2 \
#&& rm -rf /var/lib/apt/lists/*
#USER itksnap
#CMD ["glmark2"]
