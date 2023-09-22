# Core50+ilab2mlight task setup
# Note that task_size_iid is not consistent between core50 and ilab, but it is only needed for the "iid" setting, which we don't currently use anyway.
# Note also that n_instance is not consistent between core50 and ilab, but is only needed for the "instance" setting, which we don't currently use

python dataloaders/task_setup.py --dataset core50+ilab2mlight --scenario class_iid --test_sess 3 7 10 15 19       --classes_per_dataset 10 14  --offline
python dataloaders/task_setup.py --dataset core50+ilab2mlight --scenario class_instance --test_sess 3 7 10 15 19  --classes_per_dataset 10 14  --offline

python dataloaders/task_setup.py --dataset core50+ilab2mlight --scenario class_iid --test_sess 3 7 10 15 19      --classes_per_dataset 10 14
python dataloaders/task_setup.py --dataset core50+ilab2mlight --scenario class_instance --test_sess 3 7 10 15 19 --classes_per_dataset 10 14