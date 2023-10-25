# Example usage:
# ./scripts/ablation/crumb_stream_1run_per_pt.sh core50 0 imagenet_pretrain_3-30-2022 everyperm

DATASET="${1:-"core50"}"
GPU="${2:-0}"
PRETRAIN_DIR="${3:-"imagenet_pretrain_MobileNet_normal"}"
SUFFIX="${4:-"MobileNet"}"
PRETRAIN_N_CLASSES="${5:-1000}"
LR=${6:-0.001}
OUTDIR="${DATASET}_${SUFFIX}"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/data/Datasets/Core50"
    MEMORY_SIZE=1430
    BATCH_SIZE=21
elif [ "$DATASET" = "toybox" ]; then
    #DATAROOT="./data/toybox/images"
    DATAROOT="/media/data/morgan_data/toybox/images"
    MEMORY_SIZE=1430
    BATCH_SIZE=21
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    MEMORY_SIZE=1430
    BATCH_SIZE=21
elif [ "$DATASET" = "cifar100" ]; then
    #DATAROOT="./data/cifar100"
    DATAROOT="/media/data/morgan_data/cifar100"
    MEMORY_SIZE=2000
    BATCH_SIZE=128
else
    echo "Invalid dataset name!"
    exit
fi

RUNS=(0 1 2 3 4)
for RUN in "${RUNS[@]}"; do
    mkdir -p "$OUTDIR"/class_iid/Crumb_MobileNet/runs-"$RUN"
    mkdir -p "$OUTDIR"/class_instance/Crumb_MobileNet/runs-"$RUN"
    weights_path=./"$PRETRAIN_DIR"/iid/Crumb_MobileNet_offline/runs-"$RUN"/CRUMB_run"$RUN"

    python -u experiment_aug.py --scenario class_iid      --save_model --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --freeze_feature_extract --model_type squeezenet --model_name MobileNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size $BATCH_SIZE --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_MobileNet/runs-"$RUN"/log.log
    python -u experiment_aug.py --scenario class_instance --save_model --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --freeze_feature_extract --model_type squeezenet --model_name MobileNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size $BATCH_SIZE --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_instance/Crumb_MobileNet/runs-"$RUN"/log.log
done