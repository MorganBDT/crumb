# Param #1: GPU ID. Default is 0
# Param #2: Learning rate. Default is 0.001
# Param #3: Dataset name (default imagenet)
# Param #4: Dataset location
# Param #5: Name of output directory
# Usage: ./scripts/optimal_imagenet/L2_imagenet.sh 1 0.001 imagenet "/path/to/dataset" output_folder

GPU="${1:-0}"
RUN="${2:-0}"
DATASET=${3:-"imagenet"}
DATAROOT=${4:-"/n/groups/kreiman/shared_data/Imagenet2012"}
OUTDIR=${5:-"${DATASET}_L2_outputs_nmi_run${RUN}"}

mkdir -p "$OUTDIR"/class_iid/L2_SqueezeNet/

python -u experiment.py --scenario class_iid --acc_topk 1 5 --keep_best_task1_net --pretrained --specific_runs $RUN --n_epoch_first_task 20 --n_epoch 1 --lr 0.001 --reg_coef 100 --model_type squeezenet --model_name SqueezeNet --agent_type regularization --agent_name L2 --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/L2_SqueezeNet/log.log