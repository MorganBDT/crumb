# Param #1: dataset name, e.g. imagenet
# Param #2: GPU ID. Default is 0
# Usage example: ./scripts/Crumb.sh imagenet 0  (runs the model on gpu 0)
# Usage example: ./scripts/Crumb.sh imagenet 0 0.001 256 8 0 /media/data/alt_imagenet_dir
DATASET="${1:-"imagenet"}"
GPU="${2:-0}"
RUN="${3:-0}"
lr=${4:-0.001}
N_MEMBLOCKS=${5:-256}
MEMBLOCK_LENGTH=${6:-8}
memory_size=${7:-0} # Meaningless in pretraining (all replay operations skipped)
RUNS_STR=$(echo "${RUN}" | sed 's/ /-/g') # If "RUN" is formatted like "0 1 2", replace spaces with dashes
OUTDIR="${DATASET}_pretrain_MobileNet_normal_run${RUNS_STR}"
if [ "$DATASET" = "core50" ]; then
    DDATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DDATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DDATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
elif [ "$DATASET" = "cifar100" ]; then
    DDATAROOT="/home/rushikesh/P1_Oct/cifar100/cifar100png"
elif [ "$DATASET" = "imagenet" ]; then
    DDATAROOT="/media/data/Datasets/ImageNet2012"
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${8:-${DDATAROOT}}

mkdir -p ${OUTDIR}/iid/Crumb_MobileNet_offline

python -u experiment_aug.py --memory_init_strat normal_MobileNet --offline --pretraining --scenario iid --save_model_every_epoch --acc_topk 1 5 --replay_times 0 --dataset $DATASET --dataroot $DATAROOT --specific_runs $RUN --n_epoch 1 --n_epoch_first_task 10 --lr ${lr} --n_memblocks ${N_MEMBLOCKS} --memblock_length ${MEMBLOCK_LENGTH} --memory_size ${memory_size} --replay_coef 0 --freeze_feature_extract --model_type squeezenet --model_name MobileNet --pretrained --agent_type crumb --agent_name Crumb  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --output_dir ${OUTDIR} | tee ${OUTDIR}/iid/Crumb_MobileNet_offline/log.log