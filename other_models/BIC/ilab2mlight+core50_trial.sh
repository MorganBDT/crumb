MY_PYTHON="python"

echo "Training class instance on ilab2mlight+core50"
$MY_PYTHON main.py  --dataset ilab2mlight+core50 --epoch 15 --lr 0.0001  --total_cls 24 --numrun 10 --paradigm "class_instance"

echo "Training class iid on ilab2mlight+core50"
$MY_PYTHON main.py  --dataset ilab2mlight+core50 --epoch 15 --lr 0.0001  --total_cls 24 --numrun 10 --paradigm "class_iid"



