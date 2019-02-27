#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE=$DIR
source $DIR/../docker.env

CMD="
docker build \
    -t $REPO/$IMAGE:$TAG_BASE \
    -f $DIR/Dockerfile \
    $WORKSPACE
"

echo $CMD
eval $CMD
