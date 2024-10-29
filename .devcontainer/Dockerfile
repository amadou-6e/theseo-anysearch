FROM mcr.microsoft.com/devcontainers/cpp:1-debian-12

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Install OpenGL and development dependencies
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libglm-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    xorg \
    xvfb \
    mesa-utils \
    x11-xserver-utils \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN echo '#!/bin/bash\n\
    Xvfb :99 -screen 0 1024x768x24 &\n\
    export DISPLAY=:99\n\
    sleep 1\n\
    exec "$@"' > /usr/local/bin/start-xvfb \
    && chmod +x /usr/local/bin/start-xvfb

# Optionally install cmake from source
COPY ./reinstall-cmake.sh /tmp/
RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh