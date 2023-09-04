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
elif [ "$DATASET" = "core50+ilab2mlight" ]; then
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
mkdir -p logs

MODEL=SqueezeNetClassifyAfterLayer12
SAVE_DIR="squeezenet_ckpts"
CKPT_FILE=SqueezeNetClassifyAfterLayer12_base_init_"$DATASET"_"$SCENARIO"_run${RUN}.pth

CUDA_VISIBLE_DEVICES=${GPU} python -u non_imagenet_train_base_init_network_from_scratch.py \
--lr 0.001 \
--epochs 10 \
--scenario "$SCENARIO" \
--run "$RUN" \
--dataset "$DATASET" \
--arch "$MODEL" \
--ckpt_file "$CKPT_FILE" \
--save_dir "$SAVE_DIR" \
--images_dir "$DATAROOT" \
--filelist_root "$FILELIST_ROOT" \
--base_max_class "$BASE_MAX_CLASS" \
--num_classes "$NUM_CLASSES" > logs/${MODEL}_${BASE_MAX_CLASS}_base_init_"$DATASET"_"$SCENARIO"_run${RUN}.log
