# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
# ./scripts/crumb_stream.sh toybox 1

DATASET="${1:-"core50"}"
GPU="${2:-0}"
EXP_PATH=${3:-"./${DATASET}_unablated"}
N_MEMBLOCKS=${4:-256}
MEMBLOCK_LENGTH=${5:-16}
cut_layer=${6:-12}
model_name=${7:-"SqueezeNet"}
OUTDIR="./ablation_study/${DATASET}_${ABLATION_NAME}"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/data/Datasets/Core50"
    #DATAROOT="/n/groups/kreiman/shared_data/core50"
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
elif [ "$DATASET" = "imagenet" ]; then
    #DATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
    DATAROOT="/media/KLAB37/datasets/ImageNet2012"
    NCLASS=1000
else
    echo "Invalid dataset name!"
    exit
fi

SCENARIOS=("class_iid")
RUNS=(0)
for scenario in "${SCENARIOS[@]}"; do
  for run in "${RUNS[@]}"; do
    weights_path=$EXP_PATH/$scenario/Crumb_"$model_name"/runs-"$run"/CRUMB_run"$run"
    python -u memory_blocks_usage.py --scenario $scenario --visualize --specific_runs $run --n_memblocks "$N_MEMBLOCKS" --memblock_length "$MEMBLOCK_LENGTH" --crumb_cut_layer $cut_layer --model_type squeezenet --model_name "$model_name" --pretrained --agent_type crumb --agent_name Crumb --batch_size 100 --n_workers 0 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --pretrained_dataset_no_of_classes "$NCLASS" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/"$scenario"/Crumb_"$model_name"/runs-"$run"/shape_bias_log.log
  done
done