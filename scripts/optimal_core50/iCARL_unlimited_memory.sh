# Param #1: database name, e.g. core50, toybox, ilab2mlight. Default is core50
# Param #2: GPU ID. Default is 0
DATASET="${1:-"core50"}"
OUTDIR="${DATASET}_outputs_unlimited_memory"
GPU="${2:-0}"

if [ "$DATASET" = "core50" ]; then
    DATAROOT="/media/data/Datasets/Core50"
elif [ "$DATASET" = "toybox" ]; then
    DATAROOT="/media/data/morgan_data/toybox/images"
elif [ "$DATASET" = "ilab2mlight" ]; then
    #DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light"
    DATAROOT="/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed"
elif [ "$DATASET" = "ilab2mlight+core50" ]; then
    DATAROOT="/media/data/Datasets"
elif [ "$DATASET" = "icubworldtransf" ]; then
    DATAROOT="/media/KLAB37/datasets/icubworldtransf_sparse"
else
    echo "Invalid dataset name!"
    exit
fi
MEMORY_SIZE=100000000

#mkdir -p ${OUTDIR}/iid/iCARL_SqueezeNet/
mkdir -p ${OUTDIR}/class_iid/iCARL_SqueezeNet/
#mkdir -p ${OUTDIR}/instance/iCARL_SqueezeNet/
mkdir -p ${OUTDIR}/class_instance/iCARL_SqueezeNet/

#python -u experiment.py --scenario iid                --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name iCARL  --gpuid 2 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/iid/iCARL_SqueezeNet/log.log  #&
#python -u experiment.py --scenario instance           --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name iCARL  --gpuid 2 --lr 0.001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 30 | tee ${OUTDIR}/instance/iCARL_SqueezeNet/log.log    #&

python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_iid  --n_runs 10 --n_epoch_first_task 10 --memory_size $MEMORY_SIZE --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name iCARL  --gpuid $GPU --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_iid/iCARL_SqueezeNet/log.log   #&
        
python -u experiment.py --dataset $DATASET --dataroot $DATAROOT  --output_dir $OUTDIR --scenario class_instance --n_epoch_first_task 10 --memory_size $MEMORY_SIZE --n_runs 10 --model_type squeezenet --model_name SqueezeNet --pretrained --agent_type exp_replay --agent_name iCARL  --gpuid $GPU --lr 0.0001 --momentum 0.9 --weight_decay 0.0001 --batch_size 21 --n_workers 8 | tee ${OUTDIR}/class_instance/iCARL_SqueezeNet/log.log
