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

COPY ./start-xvfb.sh /usr/local/bin/start-xvfb.sh
RUN chmod +x /usr/local/bin/start-xvfb.sh && \
    echo '. /usr/local/bin/start-xvfb.sh' >> /root/.bashrc && \
    echo '. /usr/local/bin/start-xvfb.sh' >> /home/vscode/.bashrcS

# Optionally install cmake from source
COPY start-xvfb.sh /usr/local/bin/start-xvfb
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

