MY_PYTHON="python"

echo "Training class instance on icubworldtransf"
$MY_PYTHON main.py  --dataset icubworldtransf --max_size 30 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 20 --numrun 10 --paradigm "class_instance"

echo "Training class iid on icubworldtransf"
$MY_PYTHON main.py  --dataset icubworldtransf --max_size 30 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 20 --numrun 10 --paradigm "class_iid"



