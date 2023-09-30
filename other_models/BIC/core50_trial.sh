MY_PYTHON="python"

echo "Training class instance on CORe50"
$MY_PYTHON main.py  --dataset core50 --epoch 15 --lr 0.0001  --total_cls 10 --numrun 10 --paradigm "class_instance"

echo "Training class iid on CORe50"
$MY_PYTHON main.py  --dataset core50 --epoch 15 --lr 0.0001  --total_cls 10 --numrun 10 --paradigm "class_iid"



