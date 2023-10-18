# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/KLAB37/datasets/Core50"
    MEMORY_SIZE=15
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/KLAB37/datasets/toybox/images"
    MEMORY_SIZE=15
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
    MEMORY_SIZE=15
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DATAROOT="/media/KLAB37/datasets/"
    MEMORY_SIZE=30
elif [ "$DATASET" = "icubworldtransf" ]; then
    DATAROOT="/media/KLAB37/datasets/icubworldtransf_sparse"
    MEMORY_SIZE=30
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p ${OUTDIR}/iid/GEM_SqueezeNet/
mkdir -p ${OUTDIR}/class_iid/GEM_SqueezeNet/
#mkdir -p ${OUTDIR}/instance/GEM_SqueezeNet/
mkdir -p ${OUTDIR}/class_instance/GEM_SqueezeNet/


#python -u experiment.py --scenario iid                --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name GEM  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/iid/GEM_SqueezeNet/log.log  &
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_iid    --lr 0.0001      --n_runs 10 --n_epoch_first_task 10 --memory_size $MEMORY_SIZE  --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name GEM  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/GEM_SqueezeNet/log.log           #&
#python -u experiment.py --scenario instance           --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name GEM  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/instance/GEM_SqueezeNet/log.log            #&
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_instance   --lr 0.0001  --n_runs 10 --n_epoch_first_task 10 --memory_size $MEMORY_SIZE  --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name GEM  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/GEM_SqueezeNet/log.log
