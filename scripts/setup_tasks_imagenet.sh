# Imagenet task setup

# Please note: the following line generates a single task with 1000 classes, to be used for pretraining only
python dataloaders/task_setup.py --dataset imagenet --scenario iid --n_class 1000 --task_size_iid 1281167 --test_sess 1 --offline

# This generates a data ordering like that used in the REMIND paper (Hayes et al. 2020).
# 100 classes are randomly selected for the first task, and these are used for pretraining.
python dataloaders/task_setup.py --dataset imagenet --scenario class_iid --n_class 100 --n_instance 100 --test_sess 1
python dataloaders/task_setup.py --dataset imagenet --scenario class_iid --n_class 100 --n_instance 100 --test_sess 1 --offline

