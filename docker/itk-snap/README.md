# Dockerfile for containerizing ITK-SNAP

**WARNING** This docker image is *not* focused on security. If you are running untrusted code, you do so at your own risk.

The `itk_docker.sh` file is just a convenience script for holding all `x11docker` execution parameters. The length can become annoying.

## Requirements

* docker, version 20.10.2 or above
* Access to buildkit docker building backend
* nvidia-container2 <https://github.com/NVIDIA/libnvidia-container>
* x11docker <https://github.com/mviereck/x11docker>

## Citations and Notes

* ITK-SNAP project page: <https://sourceforge.net/projects/itk-snap/files/itk-snap/>
* `UserPreferences.xml` file taken directly from <https://github.com/dorianps/docker/blob/master/itksnap/UserPreferences.xml>
