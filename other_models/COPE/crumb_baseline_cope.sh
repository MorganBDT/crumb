# USAGE: e.g. to run icub on gpu 3:
# ./crumb_baseline_cope.sh icubworldtransf 3

DATASET="${1:-"core50"}"
GPU_ID="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=2 # PER CLASS
    BATCH_SIZE=21
elif [ "$DATASET" = "toybox" ]; then
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=2 # PER CLASS
    BATCH_SIZE=21
elif [ "$DATASET" = "ilab2mlight" ]; then
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=2 # PER CLASS
    BATCH_SIZE=21
elif [ "$DATASET" = "icubworldtransf" ]; then
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=2 # PER CLASS
    BATCH_SIZE=21
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    PARADIGMS=("class_iid" "class_instance")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=2 # PER CLASS
    BATCH_SIZE=21
elif [ "$DATASET" = "cifar100" ]; then
    PARADIGMS=("class_iid")
    RUNS=(0 1 2 3 4 5 6 7 8 9)
    MEMORY_SIZE=1 # PER CLASS
    BATCH_SIZE=128
elif [ "$DATASET" = "imagenet" ]; then
    PARADIGMS=("class_iid")
    RUNS=(0 1 2 3 4)
    MEMORY_SIZE=10 # PER CLASS
    BATCH_SIZE=128
else
    echo "Invalid dataset name!"
    exit
fi

for PARADIGM in "${PARADIGMS[@]}"; do
  for RUN in "${RUNS[@]}"; do
     python main.py --dataset $DATASET --run $RUN --paradigm $PARADIGM --batch_size $BATCH_SIZE --n_memories $MEMORY_SIZE --gpu_id $GPU_ID --data_file "$DATASET".pt
  done
done