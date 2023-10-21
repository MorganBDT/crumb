import os
import sys
import argparse
import random
import time
import subprocess
import numpy as np
import torch
import pandas as pd
from dataloaders import datasets
from torchvision import transforms
import agents
from plotnine import ggplot, aes, geom_bar, geom_segment, arrow, geom_text, geom_errorbar, geom_errorbarh, labs, themes, ggsave, scale_y_continuous


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_out_path(args):
    if args.custom_folder is None:
        if args.offline:
            subdir = args.agent_name + '_' + args.model_name + '_' + 'offline/'
        else:
            subdir = args.agent_name + '_' + args.model_name
    else:
        subdir = args.custom_folder

    if args.specific_runs is not None:
        rundir = "runs"
        for r in args.specific_runs:
            rundir = rundir + "-" + str(r)
        subdir = os.path.join(subdir, rundir)

    total_path = os.path.join(args.output_dir, args.scenario, subdir)

    # make output directory if it doesn't already exist
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    return total_path


def run(args, run):

    # read dataframe containing information for each task
    if args.offline:
        task_df = pd.read_csv(os.path.join('dataloaders', args.dataset + '_task_filelists', args.scenario, 'run' + str(run), 'offline', 'train_all.txt'), index_col = 0)
    else:
        task_df = pd.read_csv(os.path.join('dataloaders', args.dataset + '_task_filelists', args.scenario, 'run' + str(run), 'stream', 'train_all.txt'), index_col = 0)

    # get classes for each task
    active_out_nodes = task_df.groupby('task')['label'].unique().map(list).to_dict()

    # get tasks
    tasks = task_df.task.unique()

    # include classes from previous task in active output nodes for current task
    for i in range(1, len(tasks)):
        active_out_nodes[i].extend(active_out_nodes[i-1])

    # since the same classes might be in multiple tasks, want to consider only the unique elements in each list
    # mostly an aesthetic thing, will not affect results
    for i in range(1, len(tasks)):
        active_out_nodes[i] = list(set(active_out_nodes[i]))

    # agent parameters
    agent_config = {
        'lr': args.lr,
        'n_class': None,
        'acc_topk': args.acc_topk,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'model_name': args.model_name,
        'agent_type': args.agent_type,
        'agent_name': args.agent_name,
        'model_weights': args.model_weights,
        'memory_weights': args.memory_weights,
        'pretrained': args.pretrained,
        'feature_extract': False,
        'freeze_feature_extract': args.freeze_feature_extract,
        'storage_type': args.storage_type,
        'optimizer': args.optimizer,
        'gpuid': args.gpuid,
        'memory_size': args.memory_size,
        'n_workers': args.n_workers,
        'n_memblocks': args.n_memblocks,
        'memblock_length': args.memblock_length,
        'memory_init_strat': args.memory_init_strat,
        'freeze_memory': args.freeze_memory,
        'crumb_cut_layer': args.crumb_cut_layer,
        'use_random_resize_crops': args.use_random_resize_crops,
        'replay_coef': args.replay_coef,
        'replay_times': args.replay_times,
        'augment_replays': args.augment_replays,
        'ntask': len(tasks),
        'visualize': args.visualize,
        'replay_in_1st_task': args.replay_in_1st_task,
        'pretrained_weights': args.pretrained_weights,
        'pretrained_dataset_no_of_classes': args.pretrained_dataset_no_of_classes,
        'pretraining': args.pretraining
        }

    if args.dataset == "core50":
        agent_config["n_class"] = 10
    elif args.dataset == "toybox":
        agent_config["n_class"] = 12
    elif args.dataset == "ilab2mlight":
        agent_config["n_class"] = 14
    elif args.dataset == "cifar100":
        agent_config["n_class"] = 100
    elif args.dataset == "imagenet":
        agent_config["n_class"] = 1000
    else:
        raise ValueError("Invalid dataset name, try 'core50', 'toybox', or 'ilab2mlight' or 'cifar100'")

    # initialize agent
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)

    if args.dataset == 'core50':
        # image transformations
        composed = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # get test data
        test_data = datasets.CORE50(
                    dataroot = args.dataroot, filelist_root = args.filelist_root, scenario = args.scenario, offline = args.offline, run = run, train = False, transform=composed)
    elif args.dataset in ["toybox", "ilab2mlight", "cifar100", "imagenet"]:
        # image transformations
        composed = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # get test data
        test_data = datasets.Generic_Dataset(
            dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario, offline=args.offline,
            run=run, train=False, transform=composed)
    else:
        raise ValueError("Invalid dataset name, try 'core50', 'toybox', 'ilab2mlight', 'cifar100', or 'imagenet'")

    visualizations = make_visualizations(agent, composed, args, run, tasks, active_out_nodes, test_data)

    return visualizations


def make_visualizations(agent, transforms, args, run, tasks, active_out_nodes, test_data):

    print('============BEGINNING TESTING============')

    set_seed(0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

    agent.net.evalModeOn()
    mem_inds_batches = []
    for i, (inputs, target) in enumerate(test_loader):
        if agent.gpu:
            with torch.no_grad():
                inputs = inputs.cuda()
        feature_maps = agent.net.FeatureExtractor(inputs)
        mem_inds = agent.net.crumbr.fbank_to_top1_inds(feature_maps)
        mem_inds_batches.append(mem_inds)

    all_mem_inds = torch.cat(mem_inds_batches, dim=0)

    #all_mem_inds = all_mem_inds[:, :, :, 0]

    flattened = all_mem_inds.view(all_mem_inds.size(0), -1)

    frequencies_per_example = [(flattened == i).sum(dim=1) for i in range(args.n_memblocks)]
    mean_frequencies = torch.stack([frequencies.float().mean() for frequencies in frequencies_per_example])
    print(mean_frequencies)
    std_dev = torch.stack([frequencies.float().std() for frequencies in frequencies_per_example])

    total_count = flattened.numel()
    mean_frequencies = mean_frequencies / total_count
    std_dev = std_dev / total_count

    label_inds = torch.tensor([32, 205, 197, 48, 201])
    colors = ['blue', 'red', '#f5c542', 'blue', 'red']
    label_frequencies = mean_frequencies[label_inds].cpu().numpy()

    sorted_indices = torch.argsort(mean_frequencies, descending=True)
    mean_frequencies = mean_frequencies[sorted_indices]
    std_dev = std_dev[sorted_indices]

    sorted_indices = sorted_indices.cpu()
    # label_positions = (sorted_indices.cpu().numpy()[:, None] == label_inds).nonzero(as_tuple=True)[0]
    label_positions = torch.tensor([torch.where(sorted_indices == x)[0] for x in label_inds]).squeeze().cpu().numpy()

    print("HEIGHT OF HEIGHEST BAR:", mean_frequencies[0])
    mean_frequencies[0] = 0

    # Convert the tensor data to a Pandas DataFrame
    df = pd.DataFrame({
        'memory block index': torch.arange(args.n_memblocks).cpu().numpy(),
        'frequency': mean_frequencies.cpu().numpy(),
        'std_dev': std_dev.cpu().numpy()
    })

    plot = (
            ggplot(df, aes(x='memory block index', y='frequency')) +
            geom_bar(stat='identity', fill='black') +
            #geom_errorbar(aes(ymin='frequency-std_dev', ymax='frequency+std_dev'), width=0.25) +
            geom_segment(aes(x=0, xend=0, y=0, yend=0.00002),
                         arrow=arrow(type='closed', angle=15, length=0.1, ends='last')) +
            geom_text(aes(x=label_positions[0]+12, y=label_frequencies[0], label=label_inds.numpy()[0]), color=colors[0], va='bottom', nudge_x=3, nudge_y=0.000003, size=12) +
            geom_segment(aes(x=label_positions[0]+10, xend=label_positions[0], y=label_frequencies[0] + 0.000003, yend=label_frequencies[0] + 0.0000005), size=1, color=colors[0], arrow=arrow(type='open', angle=20, length=0.1, ends='last')) +

            geom_text(aes(x=label_positions[1], y=label_frequencies[1], label=label_inds.numpy()[1]), color=colors[1], va='bottom', nudge_y=0.000005, size=12) +
            geom_segment(aes(x=label_positions[1], xend=label_positions[1], y=label_frequencies[1] + 0.000005, yend=label_frequencies[1] + 0.0000005), size=1, color=colors[1], arrow=arrow(type='open', angle=20, length=0.1, ends='last')) +

            geom_text(aes(x=label_positions[2]-3, y=label_frequencies[2], label=label_inds.numpy()[2]), color=colors[2], va='bottom', nudge_y=0.000005, size=12) +
            geom_segment(aes(x=label_positions[2], xend=label_positions[2], y=label_frequencies[2] + 0.000005, yend=label_frequencies[2] + 0.0000005), size=1, color=colors[2], arrow=arrow(type='open', angle=20, length=0.1, ends='last')) +
            
            geom_text(aes(x=label_positions[3]+12, y=label_frequencies[3], label=label_inds.numpy()[3]), color=colors[3], va='bottom', nudge_y=0.000003, size=12) +
            geom_segment(aes(x=label_positions[3]+10, xend=label_positions[3], y=label_frequencies[3] + 0.000003, yend=label_frequencies[3] + 0.0000005), size=1, color=colors[3], arrow=arrow(type='open', angle=20, length=0.1, ends='last')) +
            
            geom_text(aes(x=label_positions[4]+3, y=label_frequencies[4], label=label_inds.numpy()[4]), color=colors[4], va='bottom', nudge_y=0.000005, size=12) +
            geom_segment(aes(x=label_positions[4], xend=label_positions[4], y=label_frequencies[4] + 0.000005, yend=label_frequencies[4] + 0.0000005), size=1, color=colors[4], arrow=arrow(type='open', angle=20, length=0.1, ends='last')) +
            
            geom_segment(aes(x=256, xend=256, y=0.000005, yend=0), color='black', linetype='dashed') +
            labs(x='memory block index', y='frequency') +
            # scale_y_continuous(breaks=[0, 0.000005, 0.00001, 0.000015, 0.00002], labels=["0", "", "1e-5", "", "2e-5"]) +
            themes.theme_bw()
    )

    ggsave(plot, filename="./" + args.dataset + "_mem_usage.png", width=6, height=2, dpi=600)

    return None

def get_args(argv):
    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='core50',
                        help="Name of the dataset to use, e.g. 'core50', 'toybox', 'ilab2mlight'")

    # stream vs offline learning
    parser.add_argument('--offline', default=False, action='store_true', dest='offline',
                        help="offline vs online (stream learning) training")

    # Accuracy metric (e.g. top1, top5)
    parser.add_argument('--acc_topk', nargs='+', default=[1], type=int, help='Specify topk metrics (e.g. top-1 accuracy, top-5 accuracy). Use like --acc_topk 1 5')

    # scenario/task
    parser.add_argument('--scenario', type=str, default='iid',
                        help="How to set up tasks, e.g. iid => randomly assign data to each task")
    parser.add_argument('--n_runs', type=int, default=1,
                        help="Number of times to repeat the experiment with different data orderings")
    parser.add_argument('--specific_runs', nargs='+', default=None, help='Do specific runs (data orderings) instead of the first n runs. Overrides the --n_runs parameter. Use like --specific_runs 1 3 5')

    # model hyperparameters/type
    parser.add_argument('--model_type', type=str, default='squeezenet',
                        help="The type (mlp|lenet|vgg|squeezenet) of backbone network")
    parser.add_argument('--model_name', type=str, default='ResNet18', help="The name of actual model for the backbone")
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lr_decrement', type=float, default=0, help="Decrement for linearly decreasing learning rate. Set to 0 by default for constant lr")
    parser.add_argument('--lr_min', type=float, default=0, help="Minimum learning rate, for learning rates that decrease.")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true')
    parser.add_argument('--freeze_memory', default=False, dest='freeze_memory', action='store_true')
    parser.add_argument('--freeze_feature_extract', default=False, dest='freeze_feature_extract', action='store_true')
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--memory_weights', type=str, default=None,
                        help="The path to the file for the memory weights (*.pth).")

    # keep track of validation accuracy
    parser.add_argument('--validate', default=False, action='store_true', dest='validate',
                        help="To keep track of validation accuracy or not")

    # for replay models
    parser.add_argument('--memory_size', type=int, default=1200, help="Number of training examples to keep in memory")
    parser.add_argument('--replay_coef', type=float, default=1,
                        help="The coefficient for replays. Larger means less plasilicity. ")
    parser.add_argument('--replay_times', type=int, default=1, help="The number of times to replay per batch. ")
    parser.add_argument('--replay_in_1st_task', default=False, action='store_true', dest='replay_in_1st_task')
    parser.add_argument('--augment_replays', default=False, action='store_true', dest='augment_replays', help='Do data augmentation (e.g. random horizontal flip) on replay examples')


    # forCRUMB model
    parser.add_argument('--n_memblocks', type=int, default=256, help="Number of memory slots to keep in memory")
    parser.add_argument('--memblock_length', type=int, default=512, help="Feature dim per memory slot to keep in memory")
    parser.add_argument('--memory_init_strat', type=str, default="random_distmatch_sparse", help="Memory initialization strategy to use. Possible values include 'random_everyperm', ''random_standard_normal', and 'zeros' (zeros only useful if loading pretrained augmem). PLEASE NOTE: this may be overridden by loading a pretrainedCRUMB matrix.")
    parser.add_argument('--visualize', default=False, action='store_true', dest='visualize',
                        help="To visualize memory and attentions (only valid for Crumb")
    parser.add_argument('--storage_type', type=str, default="feature",
                        help="Type of replay storage, e.g. 'feature' or 'image'")
    parser.add_argument('--crumb_cut_layer', type=int, default=12,
                        help="Layer after which to cut the network and insert a crumb reconstructor module. Numbering starts at 1")
    parser.add_argument('--use_random_resize_crops', default=False, action='store_true', dest='use_random_resize_crops',
                        help='Apply REMINDs feature-level data augmentation')

    # directories
    # parser.add_argument('--dataroot', type = str, default = 'data/core50', help = "Directory that contains the data")
    parser.add_argument('--dataroot', type=str, default='/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50',
                        help="Directory that contains the data")
    # parser.add_argument('--dataroot', type = str, default = '/home/mengmi/Projects/Proj_CL_NTM/data/core50', help = "Directory that contains the data")
    parser.add_argument('--filelist_root', type=str, default='dataloaders',
                        help="Directory that contains the filelists for each task")
    parser.add_argument('--output_dir', default='outputs',
                        help="Where to store accuracy table")
    parser.add_argument('--custom_folder', default=None, type=str, help="a custom subdirectory to store results")

    # gpu/cpu settings
    parser.add_argument('--gpuid', nargs="+", type=int, default=[-1],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--n_workers', default=1, type=int, help="Number of cpu workers for dataloader")

    parser.add_argument('--pretraining', default=False, action='store_true', dest='pretraining', help='Streamline training process for pretraining (e.g. no storage of replay examples, no redundant tests on 1st/current task')

    # for whole model pretrained on some dataset
    parser.add_argument('--pretrained_weights', default=False, action='store_true', dest='pretrained_weights', help = "To use pretrained weights or not")
    parser.add_argument('--pretrained_dataset_no_of_classes', default=1000, type=int, dest='pretrained_dataset_no_of_classes', help = "Number of classes in the dataset used for pretraining ")

    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():

    start_time = time.time()

    # get command line arguments
    args = get_args(sys.argv[1:])
    print("args are")
    print(args)

    # appending path to cwd to directories
    args.dataroot = os.path.join(os.getcwd(), args.dataroot)
    args.output_dir = os.path.join(os.getcwd(), args.output_dir)

    # ensure that a valid scenario has been passed
    if args.scenario not in ['iid', 'class_iid', 'instance', 'class_instance']:
        print('Invalid scenario passed, must be one of: iid, class_iid, instance, class_instance')
        return

    total_path = get_out_path(args)

    # writing hyperparameters
    git_commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    try:
        git_branch_name = subprocess.check_output(["git branch | grep \"*\" | cut -c 3-"], shell=True).decode('ascii').strip()
    except subprocess.CalledProcessError:
        try:
            print("Having trouble getting branch name with this version of git. Trying a different command...")
            git_branch_name = subprocess.check_output(['git', 'branch', '--show-current']).decode('ascii').strip()
            print("Alternate command for getting git branch name was successful. Continuing...")
        except subprocess.CalledProcessError:
            print("Having trouble getting branch name with this version of git. Trying yet another command...")
            git_branch_name = subprocess.check_output(['git', 'rev-parse', 'abbrev-ref', 'HEAD']).decode('ascii').strip()
            print("Alternate command for getting git branch name was successful. Continuing...")
    print("Current git commit id: " + git_commit_hash + " on branch " + git_branch_name)
    args_dict = vars(args)
    args_dict["git_branch_name"] = git_branch_name
    args_dict["git_commit_hash"] = git_commit_hash
    with open(os.path.join(total_path, 'hyperparams.csv'), 'w') as f:
        for key, val in args_dict.items():
            f.write("{key},{val}\n".format(key=key, val=val))

    if args.specific_runs is None:
        runs = range(args.n_runs)
    else:
        runs = [int(r) for r in args.specific_runs]

    for r in runs:
        print('=============Generating visualizations for run ' + str(r) + '=============')

        # setting seed for reproducibility
        torch.manual_seed(r)
        np.random.seed(r)
        random.seed(r)

        vizualizations = run(args, r)

    print("Total running time: " + str(time.time() - start_time))

if __name__ == '__main__':

    main()
