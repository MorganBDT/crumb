import pandas as pd
import numpy as np
import os
import sys
import math
import argparse
import copy

#core50 dataset components
#10 classes; 
#5 objects per class; 
#50 objects in total
#11 sessions per object (8 sessions for train, 3 for test)
#instance is defined as (object, session)

#Toybox dataset components
#12 classes
#30 objects per class
#360 objects in total
#10 sessions per object (12 in original dataset, "present" and "absent" transformations discarded)
TOYBOX_FRAMES_PER_SESSION = 15

#iLab-2M-Light dataset components
# 14 classes ("semi" removed from original 15)
# 28 objects per class
# 392 objects in total
# 8 sessions per object
ILAB2MLIGHT_FRAMES_PER_SESSION = 15


# sample every x frames from core50 dataset
def sample_frames_core50(examples, sample_rate=20, first_frame=10, max_frames=None):

    # for some reason, some image frames are missing before the 10th frame, so we start sampling at the 10th frame
    sampled = examples[examples.im_num.isin(np.arange(first_frame,300,sample_rate))]
    return sampled


# split into training and testing based on choice of session    
def train_test_split(dataset, outdir, test_sess, o2=False):

    # get directory map of dataset (core50, toybox, etc)
    if '+' not in dataset: # if not a combined dataset, like core50+ilab2mlight
        dirmap = pd.read_csv(os.path.join(outdir, dataset + '_dirmap.csv'))
    else:
        dirmap = None

    # sample frames from sessions
    # core50 videos were taken 20fps, so sampling every 20 images is equivalent to sampling one frame per second
    if dataset == 'core50':
        examples = sample_frames_core50(dirmap)
    elif dataset == 'toybox':
        # toybox pre-sampled in toybox_sample.py due to the complexity of this dataset's structure
        examples = dirmap
    elif dataset == 'ilab2mlight':
        # ilab2mlight pre-sampled in ilab2mlight_sample.py due to the complexity of this dataset's structure
        examples = dirmap
    elif dataset == 'cifar100' or dataset == 'imagenet' or dataset == 'imagenet900':
        examples = dirmap
    elif dataset == 'core50+ilab2mlight':
        core50_dirmap = sample_frames_core50(pd.read_csv(os.path.join(outdir, 'core50_dirmap.csv')))
        ilab2mlight_dirmap = pd.read_csv(os.path.join(outdir, 'ilab2mlight_dirmap.csv'))

        # MODIFY iLAB FORMAT TO APPEND TO CORe50
        core50_dirmap = core50_dirmap[["class", "object", "session", "im_num", "im_path"]]
        ilab2mlight_dirmap = ilab2mlight_dirmap[["class", "object", "session", "im_num", "im_path"]]

        # Reindex ilab sessions from 1-8 with test set sessions 4 and 8, to 12-19 with test sessions 15 and 19
        # This allows 11 core50 sessions to be separate.
        ilab2mlight_dirmap["session"] = ilab2mlight_dirmap["session"] + 11

        # Reindex ilab classes to start counting from 11 instead of 1, to allow the 10 CORe50 classes to be separate
        ilab2mlight_dirmap["class"] = ilab2mlight_dirmap["class"] + 10

        ## APPEND DIRECTORY NAMES
        if o2:
            ilab2mlight_dir = os.path.join("iLab-2M-Light", "train_img_distributed")
            core50_dir = os.path.join("core50", "core50_128x128")
        else:
            ilab2mlight_dir = os.path.join("ilab2M", "iLab-2M-Light", "train_img_distributed")
            core50_dir = os.path.join("Core50", "core50_128x128")
        ilab2mlight_dirmap['im_path'] = ilab2mlight_dir + "/" + ilab2mlight_dirmap['im_path'].astype(str)
        core50_dirmap['im_path'] = core50_dir + "/" + core50_dirmap['im_path'].astype(str)

        combined_dirmap = pd.concat([core50_dirmap, ilab2mlight_dirmap], ignore_index=True)

        train_combined = combined_dirmap[~combined_dirmap.session.isin(test_sess)].reset_index(drop=True)
        test_combined = combined_dirmap[combined_dirmap.session.isin(test_sess)].reset_index(drop=True)

        ## Resample to get equal number of images per class (only in training set - keep all test images).

        balanced_splits = []
        for idx, combined_split in enumerate([train_combined, test_combined]):
            # Calculate the number of rows to sample for each class
            sample_size = combined_split['class'].value_counts().min()
            print("Min # examples per class in " + ["train", "test"][idx] + " set of core50+ilab2mlight: ", sample_size)

            # Create an empty DataFrame to store the balanced data
            balanced_dirmap = combined_dirmap.iloc[:0].copy()

            # Iterate through each unique class value
            for class_value in combined_split['class'].unique():
                # Sample the required number of rows for this class
                sampled_rows = combined_split[combined_split['class'] == class_value].sample(sample_size, replace=False, random_state=0)

                # Append the sampled rows to the balanced DataFrame
                balanced_dirmap = pd.concat([balanced_dirmap, sampled_rows])

            balanced_splits.append(balanced_dirmap)

        balanced_all = pd.concat(balanced_splits, ignore_index=True)
        balanced_all = balanced_all.sort_values(by=["class", "object", "session", "im_num"], ignore_index=True)

        examples = balanced_all

    else:
        raise ValueError("Invalid dataset name")
    
    # split based on which sessions are designated train and test sessions
    train = examples[~examples.session.isin(test_sess)].reset_index(drop=True)
    test = examples[examples.session.isin(test_sess)].reset_index(drop=True)
    
    return train, test


def class_shuffle_dataset_transfer(classes_by_dataset):
    """classes_by_dataset should be a list of lists, each list contains the class ids in one dataset."""
    class_ids_shuffled = []
    for dataset_classes in copy.deepcopy(classes_by_dataset):
        np.random.shuffle(dataset_classes)
        class_ids_shuffled.extend(list(dataset_classes))
    return np.array([int(x) for x in class_ids_shuffled], dtype=np.int64)


# setup files containing image paths/labels for each task in the iid scenario
def iid_task_setup(dataset='core50', n_runs = 10, task_size = 1200, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders', o2=False):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess, o2=o2)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        iid_task_filelist(train.copy(), test, run, task_size, offline, outdir, dataset=dataset)
    

# setup files containing image paths/labels for each task in the class iid scenario
# nclass is number of classes per task
def class_iid_task_setup(dataset='core50', n_runs = 10, n_class = 2, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir='dataloaders', o2=False, classes_per_dataset=None, setting="class_iid"):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess, o2=o2)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        class_iid_task_filelist(train.copy(), test, run, n_class, offline, outdir, dataset=dataset, classes_per_dataset=classes_per_dataset, setting=setting)


# setup files containing image paths/labels for each task in the instance setting
# ninstance is number of instances per task
def instance_task_setup(dataset='core50', n_runs = 10, n_instance = 80, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders', o2=False):

    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess, o2=o2)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        instance_task_filelist(train.copy(), test, run, n_instance, sample_rate, offline, outdir, dataset=dataset)
        

# setup files containing image paths/labels for each task in the class scenario
# nclass is number of classes per task
def class_instance_task_setup(dataset='core50', n_runs = 10, n_class = 2, sample_rate = 20, test_sess = [3,7,10], offline = False, outdir = 'dataloaders', o2=False, classes_per_dataset=None, setting="class_instance"):
    
    # split into training and testing
    # default is to use sessions 3,7, & 10 for testing, the rest for training
    train, test = train_test_split(dataset, outdir, test_sess, o2=o2)
    
    # creating tasks for each run
    for run in range(n_runs):
        # passing a copy so the original train df is not modified
        class_instance_task_filelist(train.copy(), test, run, n_class, offline,  outdir, dataset=dataset, classes_per_dataset=classes_per_dataset, setting=setting)


# setup tasks where each task is iid (independently and identically distributed)
def iid_task_filelist(train, test, run, task_size, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/iid/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/iid/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir) 
     
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # shuffling training data, using a different seed for each run
    train_shuffled = train.sample(frac = 1, random_state = run).reset_index(drop = True)
    
    # last task may have fewer examples, if no. examples and task size do not divide properly
    n_task = math.ceil(train.shape[0] / task_size)
    
    # assigning examples to each task
    for b in range(n_task):
        # if offline, we want all previous examples as well
        # also want to shuffle them
        if offline:
            train_task = train_shuffled.loc[: (b+1) * task_size].sample(frac = 1, random_state = run)
        else:
            train_task = train_shuffled.loc[b * task_size: (b+1) * task_size]
        
        # assigning task
        train_task['task'] = b
        
        # appending task to train_final
        train_final = train_final.append(train_task)    
     
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test = test[test['class'].isin(class_to_label.keys())]
    test['label'] = test['class'].map(class_to_label)


    # writing tasks to .txt files
    for b in range(n_task):
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)
    
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.to_csv(run_dir + '/train_all.txt')

        
def class_iid_task_filelist(train, test, run, n_class, offline, outdir, dataset='core50', classes_per_dataset=None, setting="class_iid"):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/' + setting + '/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/' + setting + '/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting the random seed
    np.random.seed(run)
    
    # shuffling the list of classes
    shuffled_classes = train['class'].unique()
    if classes_per_dataset is None:
        np.random.shuffle(shuffled_classes)
    elif setting == "class_iid_transfer":
        shuffled_classes = class_shuffle_dataset_transfer(classes_per_dataset)
    else:
        raise ValueError("Error: what setting is being used, class_iid or class_iid_transfer?")
    
    # number of tasks
    n_task = math.ceil(len(shuffled_classes) / n_class)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterating over classes and assigning examples for each class a task
    for b in range(n_task):
        
        # assigning classes to task
        # shuffline again if offline
        if offline:
            task_classes = shuffled_classes[:(b+1)*n_class]
            np.random.shuffle(task_classes)
        else:  
            task_classes = shuffled_classes[(b*n_class):(b+1)*n_class]
        
        # shuffling task and assigning task number
        train_task = train.loc[train['class'].isin(task_classes)].sample(frac = 1, random_state = run)
        train_task['task'] = b
        
        # appending task to train_final
        train_final = train_final.append(train_task)
        
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test = test[test['class'].isin(class_to_label.keys())]
    test = test[test['class'].isin(class_to_label.keys())]
    test['label'] = test['class'].map(class_to_label)

    for b in range(n_task):
        
        # writing task filelist
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)

            
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')

    
def instance_task_filelist(train, test, run, n_instance, sample_rate, offline, outdir, dataset='core50'):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/instance/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/instance/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting random seed
    np.random.seed(run)
    
    # getting unique instances of objects
    instances = np.unique(train[['object', 'session']].values, axis = 0)
    
    # shuffling
    np.random.shuffle(instances)
    
    # calculating the number of training examples per task
    # MT: n_instance is actually the number of SESSIONS per task
    if dataset == 'core50':
        n_ex = n_instance * (300 / sample_rate)
    elif dataset == 'toybox':
        n_ex = n_instance * TOYBOX_FRAMES_PER_SESSION
    elif dataset == 'ilab2mlight':
        n_ex = n_instance * ILAB2MLIGHT_FRAMES_PER_SESSION
    else:
        raise ValueError("Invalid dataset name, use 'core50', 'toybox', or 'ilab2mlight'")
    
    # last task might have fewer instances due to no. train examples not divinding evenly
    # MT: train.shape[0] is the number of images in the train set
    n_task = math.ceil(train.shape[0] / n_ex)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterate over tasks
    for b in range(n_task):
        
        # initializing
        train_task = pd.DataFrame(columns = train.columns)
        
        # assigning instances to task
        # if offline, getting all seen instances
        # also shuffline if offline
        if offline:
            task_instances = instances[:(b+1)*n_instance]
            np.random.shuffle(task_instances)
        else:  
            task_instances = instances[(b*n_instance):(b+1)*n_instance]
        
        # iterate over instances assigned to that task
        for inst in task_instances:
                    
            # get examples that match that instance, ensure its sorted temporally
            inst_task = train[(train.object == inst[0]) & (train.session == inst[1])].sort_values('im_num')
            
            # assign task
            inst_task['task'] = b
            
            # append to examples for that task
            train_task = train_task.append(inst_task)
            
        # append to train_final
        train_final = train_final.append(train_task)
    
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test = test[test['class'].isin(class_to_label.keys())]
    test['label'] = test['class'].map(class_to_label)
    
    for b in range(n_task):
            
        # write task to file
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)

        
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to .txt
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')      
        
    
def class_instance_task_filelist(train, test, run, n_class, offline, outdir, dataset='core50', classes_per_dataset=None, setting="class_instance"):
    
    # defining directory to dump file
    if offline:
        run_dir = outdir + '/' + dataset + '_task_filelists/' + setting + '/run' + str(run) + '/offline/'
    else:
        run_dir = outdir + '/' + dataset + '_task_filelists/' + setting + '/run' + str(run) + '/stream/'
    
    # creating directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # setting the random seed
    np.random.seed(run)
    
    # shuffling the list of classes
    shuffled_classes = train['class'].unique()
    if classes_per_dataset is None:
        np.random.shuffle(shuffled_classes)
    elif setting == "class_instance_transfer":
        shuffled_classes = class_shuffle_dataset_transfer(classes_per_dataset)
    else:
        raise ValueError("Error: what setting is being used, class_instance or class_instance_transfer?")
    
    # number of tasks
    n_task = math.ceil(len(shuffled_classes) / n_class)
    
    # initializing task column
    train['task'] = -1
    
    # initializing 
    train_final = pd.DataFrame(columns = train.columns)
    
    # iterating over classes and assigning examples for each class a task
    for b in range(n_task):
        
        train_task = pd.DataFrame(columns = train.columns)
        
        # assigning classes to task
        # if offline, getting all previously seen classes and shuffline
        if offline:
            task_classes = shuffled_classes[:(b+1)*n_class]
            np.random.shuffle(task_classes)
        else:  
            task_classes = shuffled_classes[(b*n_class):(b+1)*n_class]
                    
        # get examples pertaining to these classes
        train_class = train[train['class'].isin(task_classes)]
        
        class_instances = np.unique(train_class[['object', 'session', 'class']].values, axis = 0)
        np.random.shuffle(class_instances)
        
        for inst in class_instances:
                        
            # get examples that match that instance, ensure its sorted temporally
            inst_task = train_class[(train_class.object == inst[0]) & (train_class.session == inst[1]) & (train_class["class"] == inst[2])].sort_values('im_num')
            
            # assign task
            inst_task['task'] = b
            
            # append to examples for that task
            train_task = train_task.append(inst_task)
        
        # appending task
        train_final = train_final.append(train_task)
        
    # defining labels based on order of appearance of classes
    classes = train_final['class'].unique()
    class_to_label = {classes[i]:i for i in range(len(classes))}
    train_final['label'] = train_final['class'].map(class_to_label)
    test = test[test['class'].isin(class_to_label.keys())]
    test['label'] = test['class'].map(class_to_label)
    
    for b in range(n_task):
        # writing task filelist
        train_final[train_final.task == b][['im_path', 'label']].to_csv(run_dir + '/train_task_' + str(b).zfill(2) + '_filelist.txt', sep = ' ', index = False, header = False)
        
            
    # write test set to .txt as well
    test[['im_path', 'label']].to_csv(run_dir + '/test_filelist.txt', sep = ' ', index = False, header = False)
    
    # writing the train df to csv
    train_final.reset_index(drop = True, inplace = True)
    train_final.to_csv(run_dir + '/train_all.txt')


def write_task_filelists(args):
    
    if args.scenario == 'iid':
        iid_task_setup(dataset=args.dataset, n_runs = args.n_runs, task_size = args.task_size_iid, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)
        
    elif args.scenario == 'class_iid':
        class_iid_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)
    
    elif args.scenario == 'instance':
        instance_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_instance = args.n_instance, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)
        
    elif args.scenario == 'class_instance':
        class_instance_task_setup(dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)

    elif args.scenario == 'class_iid_transfer':
        class_iid_task_setup(setting=args.scenario, classes_per_dataset=args.classes_per_dataset, dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)

    elif args.scenario == 'class_instance_transfer':
        class_instance_task_setup(setting=args.scenario, classes_per_dataset=args.classes_per_dataset, dataset=args.dataset, n_runs = args.n_runs, n_class = args.n_class, sample_rate = args.sample_rate, test_sess = args.test_sess, offline = args.offline, outdir = args.root, o2=args.o2)

def get_args(argv):
    
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    # scenario/task
    parser.add_argument('--dataset', type=str, default='core50', help="Name of the dataset to use, e.g. 'core50', 'toybox', 'ilab2mlight'")
    parser.add_argument('--scenario', type=str, default='iid', help="How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--n_runs', type=int, default=10, help="Number of times to repeat the experiment with different data orderings")
    parser.add_argument('--task_size_iid', type=int, default=1200, help="Number of examples per task, only applicable for the iid setting")
    parser.add_argument('--n_class', type=int, default=2, help="Number of classes per task, valid for the class_iid and class_instance settings")
    parser.add_argument('--n_instance', type=int, default=80, help="Number of instances to assign to each class, valid for the instance setting") #8 sessions per obj; 5 objs; 2 classes per incremental task; thus 80 instances (note - actually SESSIONS) per task
    parser.add_argument('--sample_rate', type=int, default=20, help="Sample an image every x frames")
    parser.add_argument('--test_sess', nargs="+", default=[3, 7, 10], type=int, help="Which sessions to use for testing")
    parser.add_argument('--offline', default=False, action='store_true', dest='offline')
    parser.add_argument('--o2', default=False, action='store_true', dest='o2', help="If formulating combined datasets for o2 cluster, use this flag")
    parser.add_argument('--classes_per_dataset', type=int, nargs='+', default=[10, 14], help="Number of classes in each dataset (e.g., for core50+ilab2mlight")

    # directories
    parser.add_argument('--root', type=str, default='dataloaders', help="Directory that contains the data")
    
    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():
    
    # get command line arguments
    args = get_args(sys.argv[1:])
    
    # appending path to cwd to root
    args.root = os.path.join(os.getcwd(), args.root)
    
    # ensure that a valid scenario has been passed
    if args.scenario not in ['iid', 'class_iid', 'instance', 'class_instance', 'class_iid_transfer', 'class_instance_transfer']:
        print('Invalid scenario passed, must be one of: iid, class_iid, instance, class_instance, class_iid_transfer, class_instance_transfer')
        return

    classes_per_dataset = []
    count = 0
    for dset_class_count in args.classes_per_dataset:
        classes_per_dataset.append(list(range(count+1, count+dset_class_count+1)))
        count += dset_class_count
    args.classes_per_dataset = classes_per_dataset
    
    write_task_filelists(args)
    

if __name__ == '__main__':
    
    main()
    


    

