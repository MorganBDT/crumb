#!/usr/bin/env bash

GPU="${1:-0}" # Default 0, include alternative GPU index as 1st argument to this script
RUN="${2:-0}"

PROJ_ROOT=/home/mbt10/crumb/other_models/Remind

export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}/image_classification_experiments

IMAGENET_DIR=/n/groups/kreiman/shared_data/Imagenet2012
BASE_MAX_CLASS=100
MODEL=SqueezeNetClassifyAfterLayer12
CKPT_FILE=SqueezeNetClassifyAfterLayer12_base_init_imagenet100900_run${RUN}.pth
LABEL_ORDER_DIR=./imagenet_files_run${RUN}/ # location of numpy label files

CUDA_VISIBLE_DEVICES=${GPU} python -u train_base_init_network_from_scratch.py \
--lr 0.0005 \
--nonpretrained \
--epochs 100 \
--num_classes 1000 \
--arch ${MODEL} \
--ckpt_file ${CKPT_FILE} \
--data ${IMAGENET_DIR} \
--base_max_class ${BASE_MAX_CLASS} \
--save_dir squeezenet_imagenet_ckpts \
--labels_dir ${LABEL_ORDER_DIR} > logs/${MODEL}_${BASE_MAX_CLASS}_base_init_imagenet100900_run${RUN}.log
