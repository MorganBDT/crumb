# This script runs the main CRUMB stream learning experiments for 5 runs based off of 5 separate pretraining runs.
# The script first requires 5 pretraining runs to be completed and organized into one folder as follows:
# e.g., running this 5 times: "./scripts/crumb_pretrain.sh imagenet 0 $RUN" with values of (0, 1, 2, 3, 4) for $RUN,
# and then combining the directories using "./utils/combine_run_dirs.sh crumb_imagenet_pretrain"
# Make sure you run this from the top-level repo directory.
# Example usage: ./scripts/crumb_stream.sh toybox 0 "_MyExperiment" 0.001 0 "./my_pretrain_dir" 256 16 1000 "dataset/location"
# Or, equivalently (with default dataset and output directories): ./scripts/crumb_stream.sh toybox

DATASET="${1:-"imagenet"}"
GPU="${2:-0}"
SUFFIX="${3:-""}"
LR=${4:-0.001}
PRETRAIN_DIR="${5:-"crumb_imagenet_pretrain"}"
N_MEMBLOCKS="${6:-256}"
MEMBLOCK_LENGTH="${7:-16}"
PRETRAIN_N_CLASSES="${8:-1000}"

OUTDIR="${DATASET}_Crumb_outputs${SUFFIX}"

if [ "$DATASET" = "core50" ]; then
    #DDATAROOT="./data/core50"
    DDATAROOT="/media/KLAB37/datasets/Core50"
    MEMORY_SIZE=417
    BATCH_SIZE=21
elif [ "$DATASET" = "toybox" ]; then
    #DDATAROOT="./data/toybox/images"
    DDATAROOT="/media/KLAB37/datasets/toybox/images"
    MEMORY_SIZE=417
    BATCH_SIZE=21
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DDATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DDATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    MEMORY_SIZE=417
    BATCH_SIZE=21
elif [ "$DATASET" = "cifar100" ]; then
    #DDATAROOT="./data/cifar100"
    DDATAROOT="/media/KLAB37/datasets/cifar100"
    MEMORY_SIZE=2782
    BATCH_SIZE=128
elif [ "$DATASET" = "imagenet" ]; then
    #DDATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
    DDATAROOT="/media/KLAB37/datasets/ImageNet2012"
    MEMORY_SIZE=278342
    BATCH_SIZE=128
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${9:-${DDATAROOT}}

RUNS=(0 1 2 3 4)
for RUN in "${RUNS[@]}"; do
    mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"
    if [ "$DATASET" = "core50" ] || [ "$DATASET" = "toybox" ] || [ "$DATASET" = "ilab2mlight" ]; then
      mkdir -p "$OUTDIR"/class_instance/Crumb_SqueezeNet/runs-"$RUN"
    fi
    weights_path="$PRETRAIN_DIR"/iid/Crumb_SqueezeNet_offline/runs-"$RUN"/CRUMB_run"$RUN"

    python -u experiment_aug.py --scenario class_iid --acc_topk 1 5 --save_model --keep_best_task1_net --best_net_direct --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size "$BATCH_SIZE" --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log
done
