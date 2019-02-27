#!/bin/bash -xe
devkit_dir=$1
feature_path=$2
model_name=$3
result_dir=$4
size=$5
pushd $devkit_dir
python run_experiment.py  $feature_path/MegaFace_Features_cm  $feature_path/FaceScrub_Features_cm _${model_name}_112x112.bin $result_dir -s $size
popd
