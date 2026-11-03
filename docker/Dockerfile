# Build and run ds4 with CUDA on hosts without a CUDA toolkit (e.g. DGX Spark).
#
# The CUDA version must not exceed what the host driver supports: check the
# "CUDA Version" printed by nvidia-smi on the host.  See docker/ds4-docker.sh.
ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential make python3 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
