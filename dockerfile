# Use Ubuntu 24.04 as base
FROM ubuntu:24.04

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install base packages + Python + venv
RUN apt-get update && \
    bash -c 'for i in {1..5}; do \
        apt-get install -y --no-install-recommends \
        openssh-server tmux git wget curl less locales sudo \
        software-properties-common rsync \
        python3 python3-pip python3-venv \
        && break || sleep 5; \
    done' && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp && chmod 1777 /tmp

# Set up Python virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip & install build tooling
RUN pip install --upgrade pip setuptools wheel maturin

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Working directory inside container
WORKDIR /workspace

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
