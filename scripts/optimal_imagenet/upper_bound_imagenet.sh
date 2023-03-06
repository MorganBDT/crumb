# Param #1: GPU ID. Default is 0
# Param #2: Learning rate. Default is 0.001
# Param #3: Dataset name (default imagenet)
# Param #4: Dataset location
# Param #5: Name of output directory
# Usage: ./scripts/optimal_imagenet/upper_bound_imagenet.sh 1 0.001 imagenet "/path/to/dataset" output_folder

GPU="${1:-0}"
RUN="${2:-0}"
DATASET=${3:-"imagenet"}
DATAROOT=${4:-"/n/groups/kreiman/shared_data/Imagenet2012"}
OUTDIR=${5:-"${DATASET}_upper_bound_outputs_nmi_run${RUN}"}

mkdir -p "$OUTDIR"/class_iid/NormalNN_SqueezeNet_offline/

python -u experiment.py --scenario class_iid --offline --acc_topk 1 5 --keep_best_task1_net --pretrained --keep_best_net_all_tasks --specific_runs $RUN --n_epoch_first_task 15 --n_epoch 15 --lr 0.001 --model_type squeezenet --model_name SqueezeNet --agent_type default --agent_name NormalNN --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/NormalNN_SqueezeNet_offline/log.log