#!/bin/sh
# Build and run ds4 inside a CUDA container, with the repository mounted at
# its host path so builds, models and outputs stay on the host.
#
#   docker/ds4-docker.sh image                  build the toolchain image
#   docker/ds4-docker.sh make [target]          default target: cuda-spark
#   docker/ds4-docker.sh run CMD [ARGS...]      e.g. run ./ds4 --cuda -m gguf/x.gguf
#   docker/ds4-docker.sh shell                  interactive shell in the container
#
# Environment:
#   CUDA_VERSION       image CUDA version (default 13.0.2; <= host driver's)
#   DS4_DOCKER_IMAGE   image name (default ds4-cuda:$CUDA_VERSION)
#   DS4_DOCKER_MOUNTS  extra host directories to mount at the same path,
#                      space separated (e.g. where gguf/ symlinks point)
set -eu
CUDA_VERSION=${CUDA_VERSION:-13.0.2}
IMAGE=${DS4_DOCKER_IMAGE:-ds4-cuda:$CUDA_VERSION}
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)

docker_run() {
    mounts="-v $ROOT:$ROOT"
    # Follow a gguf/ symlink so the model directory is visible too.
    if [ -L "$ROOT/gguf" ]; then
        target=$(cd "$ROOT/gguf" && pwd -P)
        mounts="$mounts -v $target:$target"
    fi
    for d in ${DS4_DOCKER_MOUNTS:-}; do mounts="$mounts -v $d:$d"; done
    tty=""
    [ -t 0 ] && [ -t 1 ] && tty="-it"
    # memlock: CUDA pins (host-registers) large weight spans.
    # shellcheck disable=SC2086
    exec docker run --rm $tty --gpus all --ipc=host \
        --ulimit memlock=-1:-1 --ulimit stack=67108864 \
        -u "$(id -u):$(id -g)" -e HOME=/tmp -e TERM="${TERM:-xterm}" \
        $mounts -w "$ROOT" "$IMAGE" "$@"
}

cmd=${1:-}
[ $# -gt 0 ] && shift
case "$cmd" in
image)
    exec docker build --build-arg CUDA_VERSION="$CUDA_VERSION" -t "$IMAGE" "$ROOT/docker"
    ;;
make)
    [ $# -gt 0 ] || set -- cuda-spark
    docker_run make "$@"
    ;;
run)
    [ $# -gt 0 ] || { echo "usage: $0 run CMD [ARGS...]" >&2; exit 2; }
    docker_run "$@"
    ;;
shell)
    docker_run bash
    ;;
*)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
