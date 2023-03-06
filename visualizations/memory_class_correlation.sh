# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
# ./scripts/crumb_stream.sh toybox 1

DATASET="${1:-"core50"}"
GPU="${2:-0}"
RUN="${3:-0}"
OUTDIR="${DATASET}_visualization_class_correlation"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/data/Datasets/Core50"
    NCLASS=10
elif [ "$DATASET" = "toybox" ]; then
    #DATAROOT="./data/toybox/images"
    DATAROOT="/media/data/morgan_data/toybox/images"
    NCLASS=12
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    NCLASS=14
elif [ "$DATASET" = "cifar100" ]; then
    #DATAROOT="./data/cifar100"
    DATAROOT="/media/data/morgan_data/cifar100"
    NCLASS=100
else
    echo "Invalid dataset name!"
    exit
fi

mkdir -p "$OUTDIR"/class_instance/Crumb_SqueezeNet/

weights_path=./data/core50_crumb_outputs_final/class_instance/Crumb_SqueezeNet/CRUMB_run"$RUN"
python -u memory_class_correlation.py --scenario class_instance --visualize --n_runs 1 --n_memblocks 256 --memblock_length 8 --model_type resnet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --batch_size 21 --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --n_classes "$NCLASS" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_instance/Crumb_SqueezeNet/log.log