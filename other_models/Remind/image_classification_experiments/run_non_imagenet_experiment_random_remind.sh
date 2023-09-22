#!/usr/bin/env bash

DATASET="${1:-"null"}" # E.g. core50, toybox, ilab2mlight, cifar100
SCENARIO="${2:-"null"}" # E.g. class_iid, class_instance
GPU="${3:-0}" # Default 0, include alternative GPU index as 1st argument to this script
RUN="${4:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/n/groups/kreiman/shared_data/core50"
    #DATAROOT="/media/data/Datasets/Core50"
    BASE_MAX_CLASS=2
    NUM_CLASSES=10
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/n/groups/kreiman/shared_data/toybox/images"
    #DATAROOT="/media/data/morgan_data/toybox/images"
    BASE_MAX_CLASS=2
    NUM_CLASSES=12
elif [ "$DATASET" = "ilab2mlight" ]; then
    DATAROOT="/n/groups/kreiman/shared_data/iLab-2M-Light/train_img_distributed"
    #DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    BASE_MAX_CLASS=2
    NUM_CLASSES=14
elif [ "$DATASET" = "cifar100" ]; then
    DATAROOT="/n/groups/kreiman/shared_data/cifar100"
    #DATAROOT="/media/data/morgan_data/cifar100"
    BASE_MAX_CLASS=5
    NUM_CLASSES=100
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DATAROOT="/n/groups/kreiman/shared_data"
    BASE_MAX_CLASS=2
    NUM_CLASSES=24
else
    echo "Invalid dataset name!"
    exit
fi

PROJ_ROOT=/home/mbt10/crumb/other_models/Remind
FILELIST_ROOT=/home/mbt10/crumb/dataloaders

export PYTHONPATH=${PROJ_ROOT}
cd ${PROJ_ROOT}/image_classification_experiments

EXPT_NAME=${DATASET}_${SCENARIO}_random_remind_run${RUN}
CLASS_INCREMENT=${BASE_MAX_CLASS}
#modify everything above; NOT below

REPLAY_SAMPLES=50
MAX_BUFFER_SIZE=417
CODEBOOK_SIZE=256
NUM_CODEBOOKS=32
BASE_INIT_CKPT=./squeezenet_ckpts/best_SqueezeNetClassifyAfterLayer12_base_init_"$DATASET"_"$SCENARIO"_run${RUN}.pth # base init ckpt
SAVE_DIR=./squeezenet_results/"${EXPT_NAME}"/

CUDA_VISIBLE_DEVICES=${GPU} python -u non_imagenet_experiment_random_remind.py \
--extract_features_from "model.features.12" \
--base_arch "SqueezeNetClassifyAfterLayer12" \
--classifier "SqueezeNetStartAfterLayer12" \
--spatial_feat_dim 13 \
--run "${RUN}" \
--batch_size 21 \
--filelist_root ${FILELIST_ROOT} \
--scenario "${SCENARIO}" \
--dataset ${DATASET} \
--images_dir ${DATAROOT} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--streaming_min_class ${BASE_MAX_CLASS} \
--streaming_max_class ${NUM_CLASSES} \
--base_init_classes ${BASE_MAX_CLASS} \
--class_increment ${CLASS_INCREMENT} \
--classifier_ckpt ${BASE_INIT_CKPT} \
--rehearsal_samples ${REPLAY_SAMPLES} \
--start_lr 0.001 \
--end_lr 0.001 \
--lr_step_size 0 \
--lr_mode constant \
--weight_decay 1e-5 \
--use_random_resized_crops \
--use_mixup \
--mixup_alpha .1 \
--num_codebooks ${NUM_CODEBOOKS} \
--codebook_size ${CODEBOOK_SIZE} \
--save_dir ${SAVE_DIR} \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
