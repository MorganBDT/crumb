# This script pretrains CRUMB on ImageNet or CIFAR100 prior to stream learning.
# Make sure you run this from the top-level repo directory.
# Example usage: ./scripts/crumb_pretrain.sh imagenet 0 0 0.001 256 16 0 "./datasets/imagenet_folder"
# Or, equivalently (if you edit the dataset location in the script): ./scripts/crumb_pretrain.sh

DATASET="${1:-"imagenet"}"
GPU="${2:-0}"
RUN="${3:-0}"
lr=${4:-0.001}
n_memblocks=${5:-256}
memblock_length=${6:-16}
memory_size=${7:-0} # Meaningless in pretraining (all replay operations skipped)

RUNS_STR=$(echo "${RUN}" | sed 's/ /-/g') # If "RUN" is formatted like "0 1 2", replace spaces with dashes
OUTDIR="crumb_${DATASET}_pretrain_run${RUNS_STR}"
if [ "$DATASET" = "cifar100" ]; then
    #DDATAROOT="./data/cifar100"
    DDATAROOT="/media/KLAB37/datasets/cifar100"
    N_EPOCHS=4
elif [ "$DATASET" = "imagenet" ]; then
    #DDATAROOT="./data/imagenet"
    DDATAROOT="/media/KLAB37/datasets/ImageNet2012"
    N_EPOCHS=10
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${8:-${DDATAROOT}}

mkdir -p ${OUTDIR}/iid/Crumb_SqueezeNet_offline

python -u experiment_aug.py --offline --pretraining --scenario iid --save_model --acc_topk 1 5 --replay_times 0 --dataset $DATASET --dataroot $DATAROOT --specific_runs $RUN --n_epoch 1 --n_epoch_first_task "$N_EPOCHS" --lr ${lr} --n_memblocks ${n_memblocks} --memblock_length ${memblock_length} --memory_size ${memory_size} --replay_coef 0 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --output_dir ${OUTDIR} | tee ${OUTDIR}/iid/Crumb_SqueezeNet_offline/log.log