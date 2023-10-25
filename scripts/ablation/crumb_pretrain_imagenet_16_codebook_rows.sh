# Param #1: dataset name, e.g. imagenet
# Param #2: GPU ID. Default is 0
# Usage example: ./scripts/Crumb.sh imagenet 0  (runs the model on gpu 0)
# Usage example: ./scripts/Crumb.sh imagenet 0 0.001 256 8 0 /media/data/alt_imagenet_dir
DATASET="${1:-"imagenet"}"
GPU="${2:-0}"
RUN="${3:-0}"
lr=${4:-0.001}
N_MEMBLOCKS=${5:-16}
MEMBLOCK_LENGTH=${6:-8}
memory_size=${7:-0} # Meaningless in pretraining (all replay operations skipped)
RUNS_STR=$(echo "${RUN}" | sed 's/ /-/g') # If "RUN" is formatted like "0 1 2", replace spaces with dashes
OUTDIR="${DATASET}_pretrain_16_augmem_rows_run${RUNS_STR}"
if [ "$DATASET" = "core50" ]; then
    DDATAROOT="/media/KLAB37/datasets/Core50"
elif [ "$DATASET" = "toybox" ]; then
    DDATAROOT="/media/KLAB37/datasets/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DDATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light"
elif [ "$DATASET" = "cifar100" ]; then
    DDATAROOT="/media/KLAB37/datasets/cifar100"
elif [ "$DATASET" = "imagenet" ]; then
    #DDATAROOT="/media/KLAB37/datasets/ImageNet2012"
    DDATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${8:-${DDATAROOT}}

mkdir -p ${OUTDIR}/iid/Crumb_SqueezeNet_offline

python -u experiment_aug.py --memory_init_strat random_distmatch_sparse --offline --pretraining --scenario iid --save_model --acc_topk 1 5 --replay_times 0 --dataset $DATASET --dataroot $DATAROOT --specific_runs $RUN --n_epoch 1 --n_epoch_first_task 10 --lr ${lr} --n_memblocks ${N_MEMBLOCKS} --memblock_length ${MEMBLOCK_LENGTH} --memory_size ${memory_size} --replay_coef 0 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --output_dir ${OUTDIR} | tee ${OUTDIR}/iid/Crumb_SqueezeNet_offline/log.log