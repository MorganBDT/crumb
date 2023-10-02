MY_PYTHON="python"

echo "Training class instance on CORe50"
$MY_PYTHON main.py  --dataset core50 --max_size 15 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 10 --numrun 10 --paradigm "class_instance"

echo "Training class iid on CORe50"
$MY_PYTHON main.py  --dataset core50 --max_size 15 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 10 --numrun 10 --paradigm "class_iid"



