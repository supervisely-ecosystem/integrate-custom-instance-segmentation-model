FROM supervisely/detectron2:1.0.5

# Nvidia keys
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# SSH-server for running with VS Code
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo openssh-server
RUN mkdir -p /run/sshd

# Utility libs
RUN apt update && apt install -y sudo wireguard iproute2

RUN python3 -m pip install supervisely==6.72.137