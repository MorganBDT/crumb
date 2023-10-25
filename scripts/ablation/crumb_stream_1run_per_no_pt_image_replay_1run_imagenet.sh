DATASET="${1:-"core50"}"
GPU="${2:-0}"
RUN="${3:-0}"
SUFFIX="${4:-"stock_pretrain_image_replay"}"
LR=${5:-0.001}
N_MEMBLOCKS="${6:-256}"
MEMBLOCK_LENGTH="${7:-8}"
PRETRAIN_N_CLASSES="${8:-1000}"

OUTDIR="${DATASET}_Crumb_outputs${SUFFIX}"

if [ "$DATASET" = "imagenet" ]; then
    #DDATAROOT="./data/imagenet"
    DDATAROOT="/media/KLAB37/datasets/ImageNet2012"
    MEMORY_SIZE=50000
    BATCH_SIZE=128
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${9:-${DDATAROOT}}

mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-$RUN/

python -u experiment_aug.py --scenario class_iid --storage_type image --save_model --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size "$BATCH_SIZE" --n_workers 8 --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log