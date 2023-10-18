# This script runs the main CRUMB stream learning experiments for 10 runs based off of a single pretraining run.
# The script first requires a pretraining run: for example, ./scripts/crumb_pretrain.sh
# Make sure you run this from the top-level repo directory.
# Example usage: ./scripts/crumb_stream.sh toybox 0 "_MyExperiment" 0.001 0 "./my_pretrain_dir" 256 16 1000 "dataset/location"
# Or, equivalently (with default dataset and output directories): ./scripts/crumb_stream.sh toybox

DATASET="${1:-"core50"}"
GPU="${2:-0}"
SUFFIX="${3:-"_adaptive_storage_32"}"
LR=${4:-0.001}
PRETRAIN_RUN="${5:-0}"
PRETRAIN_DIR="${6:-"imagenet_pretrain_32_augmem_feat"}"
N_MEMBLOCKS="${7:-256}"
MEMBLOCK_LENGTH="${8:-32}"
PRETRAIN_N_CLASSES="${9:-1000}"

OUTDIR="${DATASET}_Crumb_outputs${SUFFIX}"

if [ "$DATASET" = "core50" ]; then
    #DDATAROOT="./data/core50"
    DDATAROOT="/media/data/Datasets/Core50"
    MEMORY_SIZE=834
    BATCH_SIZE=21
    N_RUNS=10
elif [ "$DATASET" = "toybox" ]; then
    #DDATAROOT="./data/toybox/images"
    DDATAROOT="/media/data/morgan_data/toybox/images"
    MEMORY_SIZE=834
    BATCH_SIZE=21
    N_RUNS=10
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DDATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DDATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    MEMORY_SIZE=834
    BATCH_SIZE=21
    N_RUNS=10
elif [ "$DATASET" = "cifar100" ]; then
    #DDATAROOT="./data/cifar100"
    DDATAROOT="/media/data/morgan_data/cifar100"
    MEMORY_SIZE=5564
    BATCH_SIZE=128
    N_RUNS=10
elif [ "$DATASET" = "imagenet" ]; then
    #DDATAROOT="./data/imagenet"
    DDATAROOT="/media/KLAB37/datasets/ImageNet2012"
    MEMORY_SIZE=556684
    BATCH_SIZE=128
    N_RUNS=5
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DDATAROOT="/media/data/Datasets"
    MEMORY_SIZE=1668
    BATCH_SIZE=21
    N_RUNS=10
elif [ "$DATASET" = "icubworldtransf" ]; then
    DDATAROOT="/media/KLAB37/datasets/icubworldtransf_sparse"
    MEMORY_SIZE=1668
    BATCH_SIZE=21
    N_RUNS=10
else
    echo "Invalid dataset name!"
    exit
fi

DATAROOT=${10:-${DDATAROOT}}

mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/
if [ "$DATASET" = "core50" ] || [ "$DATASET" = "toybox" ] || [ "$DATASET" = "ilab2mlight" ] || [ "$DATASET" = "ilab2mlight+core50" ] || [ "$DATASET" = "icubworldtransf" ]; then
  mkdir -p "$OUTDIR"/class_instance/Crumb_SqueezeNet/
fi


weights_path="$PRETRAIN_DIR"/iid/AugMem_SqueezeNet_offline/runs-"$PRETRAIN_RUN"/HAMN_run"$PRETRAIN_RUN"
python -u experiment_aug.py --scenario class_iid      --adaptive_storage --save_model --n_runs "$N_RUNS" --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size "$BATCH_SIZE" --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/log.log
python -u experiment_aug.py --scenario class_instance --adaptive_storage --save_model --n_runs "$N_RUNS" --n_epoch_first_task 10 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size "$BATCH_SIZE" --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_instance/Crumb_SqueezeNet/log.log