# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs_lower_bound_freeze_lr0_0001"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/KLAB37/datasets/Core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/KLAB37/datasets/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DATAROOT="/media/KLAB37/datasets/"
elif [ "$DATASET" = "icubworldtransf" ]; then
    DATAROOT="/media/KLAB37/datasets/icubworldtransf_sparse"
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p ${OUTDIR}/iid/NormalNN_SqueezeNet/
mkdir -p ${OUTDIR}/class_iid/NormalNN_SqueezeNet/
#mkdir -p ${OUTDIR}/instance/NormalNN_SqueezeNet/
mkdir -p ${OUTDIR}/class_instance/NormalNN_SqueezeNet/

python -u experiment.py --scenario class_iid      --n_runs 10 --n_epoch_first_task 10 --n_epoch 1 --lr 0.0001 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type default --agent_name NormalNN --gpuid $GPU --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/NormalNN_SqueezeNet/log.log
python -u experiment.py --scenario class_instance --n_runs 10 --n_epoch_first_task 10 --n_epoch 1 --lr 0.0001 --freeze_feature_extract --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type default --agent_name NormalNN --gpuid $GPU --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/NormalNN_SqueezeNet/log.log