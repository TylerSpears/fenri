#
# Taken from
# <medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1>

ARG BASE
FROM ${BASE}

# Dependencies for glvnd and X11.
# RUN apt-get update \
#   && apt-get install -y -qq --no-install-recommends \
#     libglvnd0 \
#     libgl1 \
#     libglx0 \
#     libegl1 \
#     libxext6 \
#     libx11-6 \
#   && rm -rf /var/lib/apt/lists/*
# # Env vars for the nvidia-container-runtime.


# Taken from
# <https://raw.githubusercontent.com/frankyeh/DSI-Studio/master/Dockerfile>

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        xvfb \
        build-essential \
        autoconf \
        libtool \
        pkg-config \
        libfontconfig1 \
        libfreetype6 \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libgomp1 \
        libice6 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        wget \
        libboost-all-dev \
        zlib1g \
        zlib1g-dev \
        unzip \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        freeglut3-dev \
        mesa-utils \
        g++ \
        gcc \
        make \
        zlib1g-dev \
        git \
        software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# # Get newer qt5
# RUN add-apt-repository ppa:beineri/opt-qt-5.12.2-bionic \
#     && apt-get update \
#     && apt install -y --no-install-recommends \
#     freetds-common libclang1-5.0 libllvm5.0 libodbc1 libsdl2-2.0-0 libsndio6.1 \
#     libsybdb5 libxcb-xinerama0 qt5123d qt512base qt512canvas3d \
#     qt512connectivity qt512declarative qt512graphicaleffects \
#     qt512imageformats qt512location qt512multimedia qt512scxml qt512svg \
#     qt512wayland qt512x11extras qt512xmlpatterns qt512charts-no-lgpl \
#     && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install DSI Studio
RUN mkdir /opt/dsi-studio \
    && cd /opt/dsi-studio \
    && git clone https://github.com/frankyeh/DSI-Studio.git \
    && mv DSI-Studio src \
    && git clone https://github.com/frankyeh/TIPL.git \
    && mv TIPL src/tipl \
    && mkdir build && cd build \
    && cd /opt/dsi-studio \
    && curl -sSL 'https://pitt.box.com/shared/static/n3ln8k7bakr7y5p0ekm3g4s0ucdmqkia.zip' --output "dsi-studio.zip"\
    && unzip dsi-studio.zip \
    && rm dsi-studio.zip \
    && cd dsi_studio_64 \
    && rm *.dll \
    && rm *.exe \
    && rm -rf iconengines \
    && rm -rf imageformats \
    && rm -rf platforms \
    && rm -rf styles \
    && mv ../build/dsi_studio . \
    && rm -rf /opt/dsi-studio/src /opt/dsi-studio/build

ENV PATH="$PATH:/opt/dsi-studio/dsi_studio_64"

ENTRYPOINT ["dsi_studio"]
