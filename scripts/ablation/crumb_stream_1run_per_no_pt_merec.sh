# Example usage:
# ./scripts/ablation/crumb_stream_1run_per_pt.sh core50 0 imagenet_pretrain_3-30-2022 everyperm

DATASET="${1:-"core50"}"
GPU="${2:-0}"
SUFFIX="${3:-"merec_instead_of_crumb"}"
LR=${4:-0.001}
OUTDIR="${DATASET}_${SUFFIX}"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/data/Datasets/Core50"
elif [ "$DATASET" = "toybox" ]; then
    #DATAROOT="./data/toybox/images"
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
elif [ "$DATASET" = "cifar100" ]; then
    #DATAROOT="./data/cifar100"
    DATAROOT="/media/data/morgan_data/cifar100"
else
    echo "Invalid dataset name!"
    exit
fi

# RUNS=(0 1 2 3 4)
RUNS=(0) # TODO
for RUN in "${RUNS[@]}"; do
    mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"
    mkdir -p "$OUTDIR"/class_instance/Crumb_SqueezeNet/runs-"$RUN"

    python -u experiment_aug.py --scenario class_iid      --specific_runs $RUN --n_epoch_first_task 1 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 --lr "$LR" --memory_size 0 --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log
    python -u experiment_aug.py --scenario class_instance --specific_runs $RUN --n_epoch_first_task 1 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks 256 --memblock_length 8 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 --lr "$LR" --memory_size 0 --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_instance/Crumb_SqueezeNet/runs-"$RUN"/log.log
done
