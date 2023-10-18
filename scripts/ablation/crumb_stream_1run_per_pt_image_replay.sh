# Example usage:
# ./scripts/ablation/crumb_stream_1run_per_pt.sh core50 0 imagenet_pretrain_3-30-2022 everyperm

DATASET="${1:-"core50"}"
GPU="${2:-0}"
PRETRAIN_DIR="${3:-"imagenet_pretrain_distmatch_sparse"}"
SUFFIX="${4:-"pretrained_image_replay"}"
LR=${5:-0.001}
PRETRAIN_N_CLASSES=${6:-1000}
OUTDIR="${DATASET}_${SUFFIX}"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/KLAB37/datasets/Core50"
    MEMORY_SIZE=200
    BATCH_SIZE=21
    N_EPOCH_FIRST_TASK=10
elif [ "$DATASET" = "toybox" ]; then
    #DATAROOT="./data/toybox/images"
    DATAROOT="/media/KLAB37/datasets/toybox/images"
    MEMORY_SIZE=200
    BATCH_SIZE=21
    N_EPOCH_FIRST_TASK=10
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    MEMORY_SIZE=200
    BATCH_SIZE=21
    N_EPOCH_FIRST_TASK=10
elif [ "$DATASET" = "cifar100" ]; then
    #DATAROOT="./data/cifar100"
    DATAROOT="/media/KLAB37/datasets/cifar100"
    MEMORY_SIZE=2000
    BATCH_SIZE=21
    N_EPOCH_FIRST_TASK=10
elif [ "$DATASET" = "imagenet" ]; then
    #DATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
    DATAROOT="/media/KLAB37/datasets/ImageNet2012"
    MEMORY_SIZE=139171
    BATCH_SIZE=128
    N_EPOCH_FIRST_TASK=15
else
    echo "Invalid dataset name!"
    exit
fi

RUNS=(0 1 2 3 4)
for RUN in "${RUNS[@]}"; do
    mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"
    mkdir -p "$OUTDIR"/class_instance/Crumb_SqueezeNet/runs-"$RUN"

    weights_path=./"$PRETRAIN_DIR"/iid/Crumb_SqueezeNet_offline/runs-"$RUN"/CRUMB_run"$RUN"
    python -u experiment_aug.py --scenario class_iid      --storage_type image --save_model --specific_runs $RUN --n_epoch_first_task $N_EPOCH_FIRST_TASK --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size $BATCH_SIZE --n_workers 8 --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log
    python -u experiment_aug.py --scenario class_instance --storage_type image --save_model --specific_runs $RUN --n_epoch_first_task $N_EPOCH_FIRST_TASK --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size $BATCH_SIZE --n_workers 8 --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_instance/Crumb_SqueezeNet/runs-"$RUN"/log.log
done
