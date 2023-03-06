# Usage: ./scripts/optimal_imagenet/crumb_imagenet100900.sh 1 0 "my_experiment" my/pretrain/directory 256 16 ...

GPU="${1:-0}"
RUN="${2:-0}"
SUFFIX="${3:-""}"
PRETRAIN_DIR="${4:-"imagenet_pretrain_task0_lr0.0005"}"
AUGMEM_SLOTS="${5:-256}"
AUGMEM_FEAT="${6:-16}"
MEMORY_SIZE=${7:-278342}
LR=${8:-0.001}
#DATAROOT=${9:-"/n/groups/kreiman/shared_data/Imagenet2012"}
DATAROOT=${9:-"/media/KLAB37/datasets/ImageNet2012"}
DATASET=${10:-"imagenet"}
OUTDIR=${11:-"imagenet100900_crumb_${SUFFIX}_run${RUN}"}

PRETRAIN_N_CLASSES=1000

mkdir -p "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"
weights_path=./"$PRETRAIN_DIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/CRUMB_run"$RUN"

python -u experiment_aug.py --scenario class_iid --freeze_feature_extract  --continuing --task_range 1 9 --acc_topk 1 5 --save_model_every_epoch --keep_best_task1_net --best_net_direct --specific_runs $RUN --n_epoch_first_task 1 --n_epoch 1 --replay_times 1 --replay_coef 5 --n_memblocks "$AUGMEM_SLOTS" --memblock_length "$AUGMEM_FEAT" --pretrained_dataset_no_of_classes "$PRETRAIN_N_CLASSES" --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type crumb --agent_name Crumb --momentum 0.9 --weight_decay 0.0001 --batch_size 128 --n_workers 8 --pretrained_weights --model_weights "$weights_path" --memory_weights "$weights_path" --lr "$LR" --memory_size "$MEMORY_SIZE" --gpuid "$GPU" --dataset "$DATASET" --dataroot "$DATAROOT"  --output_dir "$OUTDIR" | tee "$OUTDIR"/class_iid/Crumb_SqueezeNet/runs-"$RUN"/log.log