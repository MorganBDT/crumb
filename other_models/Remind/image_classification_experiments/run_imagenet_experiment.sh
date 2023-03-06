#!/usr/bin/env bash

GPU="${1:-0}" # Default 0, include alternative GPU index as 1st argument to this script
RUN="${2:-0}"

PROJ_ROOT=/home/mbt10/crumb/other_models/Remind

export PYTHONPATH=${PROJ_ROOT}

cd ${PROJ_ROOT}/image_classification_experiments

IMAGE_DIR=/n/groups/kreiman/shared_data/Imagenet2012
EXPT_NAME=remind_squeezenet_imagenet100900_run${RUN}

REPLAY_SAMPLES=50
#MAX_BUFFER_SIZE=959665
MAX_BUFFER_SIZE=278246
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CLASSES=100
CLASS_INCREMENT=100
#CLASS_INCREMENT=2
NUM_CLASSES=1000
#NUM_CLASSES=104
BASE_INIT_CKPT=./squeezenet_imagenet_ckpts/best_SqueezeNetClassifyAfterLayer12_base_init_imagenet100900_run${RUN}.pth # base init ckpt file
LABEL_ORDER_DIR=./imagenet_files_run${RUN}/ # location of numpy label files

CUDA_VISIBLE_DEVICES=${GPU} python -u imagenet_experiment.py \
--extract_features_from "model.features.12" \
--base_arch "SqueezeNetClassifyAfterLayer12" \
--classifier "SqueezeNetStartAfterLayer12" \
--spatial_feat_dim 13 \
--classifier_ckpt ${BASE_INIT_CKPT} \
--images_dir ${IMAGE_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${BASE_INIT_CLASSES} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.001 \
--end_lr 0.001 \
--lr_step_size 0 \
--lr_mode constant \
--weight_decay 1e-5 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--label_dir ${LABEL_ORDER_DIR} \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
