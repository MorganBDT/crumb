# Param #1: GPU ID. Default is 0
# Param #2: Learning rate. Default is 0.001
# Param #3: Memory size (number of stored examples). Default is 20000
# Param #4: Dataset name (default imagenet)
# Param #5: Dataset location
# Param #6: Name of output directory
# Usage: ./scripts/optimal_imagenet/iCARL_imagenet.sh 1 0.001 10000 imagenet "/path/to/dataset" output_folder

GPU="${1:-0}"
RUN="${2:-0}"
MEMORY_SIZE=${3:-10000}
DATASET=${4:-"imagenet"}
DATAROOT=${5:-"/n/groups/kreiman/shared_data/Imagenet2012_temp_copy"}
OUTDIR=${6:-"${DATASET}_iCARL_outputs_nmi_run${RUN}"}

mkdir -p "$OUTDIR"/class_iid/iCARL_SqueezeNet/

python -u experiment.py --scenario class_iid --acc_topk 1 5 --keep_best_task1_net --pretrained --specific_runs $RUN --n_epoch_first_task 10 --n_epoch 1 --lr 0.0001 --reg_coef 100 --model_type squeezenet --model_name SqueezeNet --agent_type exp_replay --agent_name iCARL --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/iCARL_SqueezeNet/log.log