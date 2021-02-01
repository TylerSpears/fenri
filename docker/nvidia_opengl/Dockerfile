FROM ubuntu:16.04 
#Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  libxext-dev \
  libx11-dev \ 
  x11proto-gl-dev \
    #libglvnd0 \
    libgl1 \
    #libglx0 \
    #libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    ninja-build meson \
  && git clone https://gitlab.freedesktop.org/glvnd/libglvnd /opt/libglvnd \
  && cd /opt/libglvnd \
  && meson builddir \
  && ninja -C builddir \
  && apt-get remove -y -qq ninja-build meson \
  && rm -rf /var/lib/apt/lists/*


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

