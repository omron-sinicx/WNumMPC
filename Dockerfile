FROM nvidia/cuda:12.5.1-devel-ubuntu24.04

# build tools
RUN apt-get -y update
RUN apt-get install -y \
    git \
    cmake

# install python3.11
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev
