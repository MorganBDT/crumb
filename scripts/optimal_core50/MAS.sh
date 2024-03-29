# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50"
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

#mkdir -p ${OUTDIR}/iid/MAS_SqueezeNet/
mkdir -p ${OUTDIR}/class_iid/MAS_SqueezeNet/
#mkdir -p ${OUTDIR}/instance/MAS_SqueezeNet/
mkdir -p ${OUTDIR}/class_instance/MAS_SqueezeNet/


#python -u experiment.py --scenario iid                --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type regularization --agent_name MAS  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/iid/MAS_SqueezeNet/log.log                 &
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_iid          --n_runs 10 --n_epoch_first_task 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type regularization --agent_name MAS  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/MAS_SqueezeNet/log.log           #&
#python -u experiment.py --scenario instance           --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type regularization --agent_name MAS  --gpuid 0 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 20 | tee ${OUTDIR}/instance/MAS_SqueezeNet/log.log            &
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_instance     --n_runs 10 --n_epoch_first_task 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type regularization --agent_name MAS  --gpuid $GPU --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/MAS_SqueezeNet/log.log
