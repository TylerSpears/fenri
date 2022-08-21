# Based on parts of the Mrtrix3 system Dockerfile
# <https://github.com/MRtrix3/mrtrix3>
FROM nvidia/cuda:10.2-runtime-ubuntu18.04

# Core system capabilities required
RUN apt update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt install -y \
    bc \
    build-essential \
    curl \
    dc \
    git \
    libopenblas-dev \
    nano \
    python2.7 \
    python3 \
    tar \
    tcsh \
    tzdata \
    unzip \
    wget

# FSL installer appears to now be ready for use with version 6
# eddy is also now included in FSL6
RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    chmod 775 fslinstaller.py && \
    python3 fslinstaller.py -D -d /opt/fsl -V 6.0.5.2 && \
    rm -f /fslinstaller.py
RUN which immv || ( echo "FSLPython not properly configured; re-running" && rm -rf /opt/fsl/fslpython && /opt/fsl/etc/fslconf/fslpython_install.sh -f /opt/fsl || ( cat /tmp/fslpython*/fslpython_miniconda_installer.log && exit 1 ) )
# RUN wget -qO- "https://www.nitrc.org/frs/download.php/5994/ROBEXv12.linux64.tar.gz//?i_agree=1&download_now=1" | \
#     tar zx -C /opt

# apt cleanup to recover as much space as possible
RUN apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup envvars
ENV OS=Linux \
    FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    FSLTCLSH=/opt/fsl/bin/fsltclsh \
    FSLWISH=/opt/fsl/bin/fslwish \
    LD_LIBRARY_PATH=/opt/fsl/lib:$LD_LIBRARY_PATH \
    PATH=/opt/fsl/bin:/opt/ROBEX:$PATH

# Remove older cuda builds for eddy, they are unnecessary bloat.
# The `.` after the digit prevents the blob from matching versions in the double-digits
# ("10.X" or higher).
RUN rm "${FSLDIR}"/bin/eddy_cuda[5-9].*
