#Use the new docker buildkit, which allows us to copy files out after completion
DOCKER_BUILDKIT=1 docker build -t dsi-studio:latest --file ./Dockerfile .

