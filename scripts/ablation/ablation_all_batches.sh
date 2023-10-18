# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
# ./scripts/crumb_stream.sh toybox 1

DATASET="${1:-"core50"}"
GPU="${2:-0}"
ABLATION_NAME=${3:-""}
N_MEMBLOCKS=${4:-256}
MEMBLOCK_LENGTH=${5:-8}
cut_layer=${6:-12}
model_name=${7:-"SqueezeNet"}
OUTDIR="./ablation_study/${DATASET}_${ABLATION_NAME}"

if [ "$DATASET" = "core50" ]; then
    #DATAROOT="./data/core50"
    DATAROOT="/media/KLAB37/datasets/Core50"
    #DATAROOT="/n/groups/kreiman/shared_data/core50"
    NCLASS=10
    SCENARIOS=("class_iid" "class_instance")
elif [ "$DATASET" = "toybox" ]; then
    #DATAROOT="./data/toybox/images"
    DATAROOT="/media/KLAB37/datasets/toybox/images"
    #DATAROOT="/n/groups/kreiman/shared_data/toybox/images"
    NCLASS=12
    SCENARIOS=("class_iid" "class_instance")
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="./data/iLab-2M-Light/train_img_distributed"
    DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    #DATAROOT="/n/groups/kreiman/shared_data/iLab-2M-Light/train_img_distributed"
    NCLASS=14
    SCENARIOS=("class_iid" "class_instance")
elif [ "$DATASET" = "cifar100" ]; then
    #DATAROOT="./data/cifar100"
    DATAROOT="/media/KLAB37/datasets/cifar100"
    #DATAROOT="/n/groups/kreiman/shared_data/cifar100"
    NCLASS=100
    SCENARIOS=("class_iid")
elif [ "$DATASET" = "imagenet" ]; then
    #DATAROOT="/n/groups/kreiman/shared_data/Imagenet2012"
    DATAROOT="/media/KLAB37/datasets/ImageNet2012"
    NCLASS=1000
    SCENARIOS=("class_iid")
else
    echo "Invalid dataset name!"
    exit
fi

SCENARIOS=("class_iid" "class_instance")
RUNS=(0 1 2 3 4)
for scenario in "${SCENARIOS[@]}"; do
  for run in "${RUNS[@]}"; do
    weights_path=$OUTDIR/$scenario/Crumb_"$model_name"/runs-"$run"/CRUMB_run"$run"
    python -u ablation_batch_accuracy.py --scenario $scenario --visualize --specific_runs $run --n_memblocks $N_MEMBLOCKS --memblock_length $MEMBLOCK_LENGTH --crumb_cut_layer $cut_layer --model_type squeezenet --model_name $model_name --pretrained --agent_type crumb --agent_name Crumb --batch_size 100 --n_workers 0 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --pretrained_dataset_no_of_classes "$NCLASS" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/$scenario/Crumb_"$model_name"/runs-"$run"/batch_acc_log.log
  done
done