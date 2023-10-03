MY_PYTHON="python"

$MY_PYTHON main.py  --dataset ilab --max_size 15 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 14 --numrun 10 --paradigm "class_instance"

$MY_PYTHON main.py  --dataset ilab --max_size 15 --epoch 15 --batch_size 21 --lr 0.0001  --total_cls 14 --numrun 10 --paradigm "class_iid"



