DATASET="${1:-"imagenet"}"
GPU="${2:-0}"
RUN="${3:-0}"
lr=${4:-0.001}
RUNS_STR=$(echo "${RUN}" | sed 's/ /-/g') # If "RUN" is formatted like "0 1 2", replace spaces with dashes
OUTDIR="${DATASET}_pretrain_vanilla_run${RUNS_STR}"
if [ "$DATASET" = "core50" ]; then
    DDATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DDATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    DDATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
elif [ "$DATASET" = "cifar100" ]; then
    DDATAROOT="/home/rushikesh/P1_Oct/cifar100/cifar100png"
elif [ "$DATASET" = "imagenet" ]; then
    DDATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${8:-${DDATAROOT}}

mkdir -p ${OUTDIR}/iid/NormalNN_SqueezeNet_offline

python -u experiment.py --offline --scenario iid --save_model --acc_topk 1 5 --replay_times 0 --dataset $DATASET --dataroot $DATAROOT --specific_runs $RUN --n_epoch 1 --n_epoch_first_task 10 --lr ${lr} --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --output_dir ${OUTDIR} | tee ${OUTDIR}/iid/NormalNN_SqueezeNet_offline/log.log