# cifar100 task setup
# n_instance parameter: 1 session per object (training session), 1 object per class (not divided into objects), 5 classes per incremental task, thus 5 instances per task
# task_size_iid parameter: 1 object per class, 1 session per object, 500 images per session. 5 classes per incremental task
# Or, more simply: 5 classes per task, 500 training images per class.

# Please note: the following line generates a single task with 100 classes, to be used for pretraining only
python dataloaders/task_setup.py --dataset cifar100 --scenario iid --n_class 100 --task_size_iid 50000 --test_sess 1 --offline

python dataloaders/task_setup.py --dataset cifar100 --scenario class_iid --n_class 5 --task_size_iid 2500 --n_instance 5 --test_sess 1
python dataloaders/task_setup.py --dataset cifar100 --scenario class_iid --n_class 5 --task_size_iid 2500 --n_instance 5 --test_sess 1 --offline