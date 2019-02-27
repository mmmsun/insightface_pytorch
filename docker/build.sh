#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE=$DIR/..
source $DIR/docker.env

CMD="
docker build \
    --build-arg BASE_IMAGE=$REPO/$IMAGE:$TAG_BASE \
    -t $REPO/$IMAGE:$TAG \
    -f $DIR/Dockerfile \
    $WORKSPACE
"

echo $CMD
eval $CMD
