# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_GEM_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/KLAB37/datasets/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/KLAB37/datasets/ilab2M/iLab-2M-Light/train_img_distributed"
else
    echo "Invalid dataset name!"
    exit
fi

#mkdir -p ${OUTDIR}/iid/GEM_SqueezeNet/
mkdir -p ${OUTDIR}/class_iid/GEM_SqueezeNet/
#mkdir -p ${OUTDIR}/instance/GEM_SqueezeNet/
mkdir -p ${OUTDIR}/class_instance/GEM_SqueezeNet/

python -u experiment.py --scenario class_iid --dataset $DATASET --dataroot $DATAROOT  --n_runs 10 --memory_size 15 --model_type squeezenet --model_name SqueezeNet --pretrained --output_dir $OUTDIR --agent_type exp_replay --lr 0.0001 --agent_name GEM  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/GEM_SqueezeNet/log.log

python -u experiment.py --scenario class_instance --dataset $DATASET --dataroot $DATAROOT --lr 0.0001  --n_runs 10 --memory_size 15 --model_type squeezenet --model_name SqueezeNet --output_dir $OUTDIR --pretrained --agent_type exp_replay --agent_name GEM  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/GEM_SqueezeNet/log.log
