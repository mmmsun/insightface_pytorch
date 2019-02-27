#!/bin/bash
DATA_DIR=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE=$DIR/..
source $DIR/docker.env

docker rm -f $CONTAINER
nvidia-docker run -it \
    -v $WORKSPACE:/workspace \
    -v $DATA_DIR:/workspace/data \
    --workdir=/workspace \
    --name=$CONTAINER \
    --ipc=host \
    $REPO/$IMAGE:$TAG_BASE bash
