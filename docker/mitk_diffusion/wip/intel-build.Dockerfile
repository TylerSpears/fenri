FROM intel/oneapi-hpckit:2023.2.1-devel-ubuntu22.04
ARG OVERRIDE_MITK_ITK_MODULE=true

ENV CC=icx
ENV CXX=icpx
ENV CMAKE_C_COMPILER=icx
ENV CMAKE_CXX_COMPILER=icpx
ENV CMAKE_Fortran_COMPILER=ifx

#RUN apt-get update && apt-get install -yq --no-install-recommends \
#        packages-here \
#        && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update --yes && apt-get install -yq --no-install-recommends \
        cmake \
        cmake-curses-gui \
        vim-tiny \
        doxygen \
        libfreetype6-dev \
        libtiff5-dev \
        graphviz \
        libxt-dev \
        libglu1-mesa-dev \
        libxcomposite1 \
        libxcursor1 \
        libxdamage-dev \
        libxi-dev \
        libxkbcommon-x11-0 \
        mesa-common-dev \
        && apt-get upgrade --yes \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ITK.cmake.patch.diff /tmp/ITK.cmake.patch.diff
RUN git clone https://phabricator.mitk.org/source/mitk.git /opt/MITK && \
       git clone https://github.com/MIC-DKFZ/MITK-Diffusion.git /opt/MITK-Diffusion && \
       if [ "$OVERRIDE_MITK_ITK_MODULE" = "true" ]; then /usr/bin/patch "/opt/MITK/CMakeExternals/ITK.cmake" < "/tmp/ITK.cmake.patch.diff"; fi

WORKDIR /opt
RUN mkdir --parents MITK-superbuild && mkdir --parents /opt/MITK-install
WORKDIR /opt/MITK-superbuild

RUN cmake \
        -DCMAKE_C_FLAGS="-Wno-error=implicit-function-declaration" \
        -DCMAKE_INSTALL_PREFIX="/opt/MITK-install" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DMITK_EXTENSION_DIRS=/opt/MITK-Diffusion/ \
        -DMITK_BUILD_CONFIGURATION=DiffusionCmdApps_NoPython \
        -DMITK_ADDITIONAL_CXX_FLAGS="-fno-finite-math-only -Wno-error=unused-but-set-variable" \
        -DBUILD_DiffusionFiberfoxCmdApps=ON \
        -DBUILD_DiffusionFiberProcessingCmdApps=OFF \
        -DBUILD_DiffusionFiberQuantificationCmdApps=OFF \
        -DBUILD_DiffusionQuantificationCmdApps=OFF \
        -DBUILD_DiffusionTractographyCmdApps=OFF \
        -DMITK_USE_OpenMP=ON \
        -DITK_USE_MKL=ON \
        -DMITK_USE_DCMQI=ON \
        -DMITK_USE_DCMTK=ON \
        -DMITK_BUILD_APP_Workbench=OFF \
        -DMITK_USE_Poco=OFF \
        -DMITK_USE_BLUEBERRY=OFF \
        -DMITK_USE_CTK=OFF \
        -DMITK_USE_Qt5=OFF \
        -DMITK_USE_SUPERBUILD=ON \
        ../MITK

# RUN cmake \
#         -DMITK_USE_Poco=OFF \
#         -DCMAKE_C_FLAGS="-Wno-error=implicit-function-declaration" \
#         -DMITK_EXTENSION_DIRS=/opt/MITK-Diffusion/ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DMITK_BUILD_CONFIGURATION=DiffusionCmdApps_NoPython \
#         -DBUILD_DiffusionFiberfoxCmdApps=ON \
#         -DBUILD_APP_Diffusion=ON \
#         -DMITK_USE_OpenMP=ON \
#         -DMITK_BUILD_SHARED_LIBS=ON \
#         -DMITK_USE_DCMQI=ON \
#         -DMITK_USE_DCMTK=ON \
#         -DMITK_BUILD_DiffusionPythonCmdApps=OFF \
#         -DMITK_USE_BLUEBERRY=OFF \
#         -DMITK_USE_CTK=OFF \
#         -DMITK_USE_Qt5=OFF \
#         -DMITK_USE_QT_HELP=OFF \
#         -DMITK_USE_Python3=OFF \
#         -DMITK_BUILD_APP_CoreApp=OFF \
#         -DMITK_BUILD_ALL_APPS=OFF \
#         -DMITK_BUILD_ALL_PLUGINS=OFF \
#         -DMITK_BUILD_APP_Workbench=OFF \
#         -DMITK_DOXYGEN_GENERATE_QCH_FILE=OFF \
#         -DBLUEBERRY_USE_QT_HELP=OFF \
#         -DMITK_BUILD_EXAMPLES=OFF \
#         ../MITK

# ARG MAKE_JOBS=1
#RUN make -j $MAKE_JOBS
