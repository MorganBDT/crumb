DATASET="${1:-"core50"}"
GPU="${2:-0}"
RUN="${3:-0}"
SUFFIX="${4:-""}"
LR=${5:-0.001}
PRETRAIN_RUN="${6:-0}"
PRETRAIN_DIR="${7:-"imagenet_pretrain_distmatch_sparse"}"
N_MEMBLOCKS="${8:-256}"
MEMBLOCK_LENGTH="${9:-8}"
PRETRAIN_N_CLASSES="${10:-1000}"

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

DATAROOT=${11:-${DDATAROOT}}

mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/

weights_path="$PRETRAIN_DIR"/iid/Crumb_SqueezeNet_offline/runs-"$PRETRAIN_RUN"/CRUMB_run"$PRETRAIN_RUN"
python -u experiment_aug.py --scenario class_iid --save_model --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size "$BATCH_SIZE" --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log