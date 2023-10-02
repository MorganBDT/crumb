# USAGE: e.g. to run icub on gpu 3:
# ./crumb_baseline_cope.sh icubworldtransf 3

DATASET="${1:-"core50"}"
GPU_ID="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/data/Datasets/Core50"
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=10
    BATCH_SIZE=21
    CLASSES_PER_TASK=2
    MEMORY_SIZE=2 # PER CLASS
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=12
    BATCH_SIZE=21
    CLASSES_PER_TASK=2
    MEMORY_SIZE=2 # PER CLASS
elif [ "$DATASET" = "ilab2mlight" ]; then
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=14
    BATCH_SIZE=21
    CLASSES_PER_TASK=2
    MEMORY_SIZE=2 # PER CLASS
elif [ "$DATASET" = "icubworldtransf" ]; then
    DATAROOT="/media/KLAB37/datasets/icubworldtransf_sparse"
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=20
    BATCH_SIZE=21
    CLASSES_PER_TASK=2
    MEMORY_SIZE=2 # PER CLASS
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DATAROOT="/media/data/Datasets"
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=24
    BATCH_SIZE=21
    CLASSES_PER_TASK=2
    MEMORY_SIZE=2 # PER CLASS
elif [ "$DATASET" = "cifar100" ]; then
    DATAROOT="/media/data/morgan_data/cifar100"
    PARADIGMS=("class_iid")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    NUM_CLASSES=100
    BATCH_SIZE=128
    CLASSES_PER_TASK=5
    MEMORY_SIZE=1 # PER CLASS
elif [ "$DATASET" = "imagenet" ]; then
    DATAROOT="/media/KLAB37/datasets/ImageNet2012"
    PARADIGMS=("class_iid")
    RUNS=(0 1 2 3 4)
    NUM_CLASSES=1000
    BATCH_SIZE=128
    CLASSES_PER_TASK=100
    MEMORY_SIZE=10 # PER CLASS
else
    echo "Invalid dataset name!"
    exit
fi

for PARADIGM in "${PARADIGMS[@]}"; do
  for RUN in "${RUNS[@]}"; do
      python main.py --filelist_root "./../../dataloaders" --nb_protos $MEMORY_SIZE --nb_cl_fg=$CLASSES_PER_TASK --nb_cl=$CLASSES_PER_TASK --gpu=$GPU_ID --random_seed=1993 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=$DATASET --scenario=$PARADIGM --runs $RUN --num_classes $NUM_CLASSES --epochs 30 --disable_gpu_occupancy --dataroot $DATAROOT --train_batch_size $BATCH_SIZE --eval_batch_size $BATCH_SIZE --test_batch_size $BATCH_SIZE
  done
done