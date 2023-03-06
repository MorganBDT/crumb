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
from PIL import Image, ImageOps

from utils.metric import accuracy, AverageMeter, Timer


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
        'optimizer': args.optimizer,
        'gpuid': args.gpuid,
        'memory_size': args.memory_size,
        'n_workers': args.n_workers,
        'n_memblocks': args.n_memblocks,
        'memblock_length': args.memblock_length,
        'memory_init_strat': args.memory_init_strat,
        'freeze_memory': args.freeze_memory,
        'replay_coef': args.replay_coef,
        'replay_times': args.replay_times,
        'augment_replays': args.augment_replays,
        'ntask': len(tasks),
        'visualize': args.visualize,
        'replay_in_1st_task': args.replay_in_1st_task,
        'pretrained_weights': args.pretrained_weights,
        'pretrained_dataset_no_of_classes': args.n_classes,
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
        test_data = datasets.CORE50_vis(
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

    #visualizations = make_visualizations(agent, composed, args, run, tasks, active_out_nodes, test_data)
    for class_id in range(10):
        visualizations = make_visualizations(agent, composed, args, run, tasks, active_out_nodes, test_data, spec_class_id=class_id)

    return visualizations


def make_binary_image(data):
    img = Image.new('1', (13, 13))
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(data[j][i])
    return img


def white_to_transparent(rgba_image):
    datas = rgba_image.getdata()
    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((item[0], item[1], item[2], 200))

    rgba_image.putdata(new_data)
    return rgba_image


def make_visualizations(agent, composed_transforms, args, run, tasks, active_out_nodes, test_data, spec_class_id=None):

    if args.offline:
        print('============BEGINNING OFFLINE VISUALIZATIONS============')
    else:
        print('============BEGINNING STREAM VISUALIZATIONS============')

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
    metric_topk = (1,)

    accs_out_avg = {}
    accs_dir_avg = {}
    for k in metric_topk:
        accs_out_avg[k] = AverageMeter()
        accs_dir_avg[k] = AverageMeter()

    batch_timer = Timer()
    batch_timer.tic()

    agent.net.evalModeOn()


    num_slices = int(round(agent.net.compressedChannel / args.memblock_length))
    slice_memory_counts = []
    for _ in range(num_slices):
        slice_memory_counts.append(np.zeros((args.n_memblocks, args.n_classes), dtype=int))

    spec_slice_id = 0
    spec_slot_id = args.spec_slot_id
    if spec_class_id is None:
        spec_class_id = 9
    topk_images = 10
    class_images = []
    class_images_inds = []
    class_images_spec_slot_counts = []

    for i, (inputs, target, paths) in enumerate(test_loader):

        if agent.gpu:
            with torch.no_grad():
                inputs = inputs.cuda()
                target = target.cuda()

        # Forward pass
        extracted = agent.net.FeatureExtractor(inputs)

        mem_inds = agent.net.crumbr.fbank_to_top1_inds(extracted).detach().cpu().numpy()

        # Get images with specified slot id present
        for im_ind in range(target.size(0)):
            if target[im_ind] == spec_class_id:
                unique_slots, slot_counts = np.unique(mem_inds[im_ind, :, :, spec_slice_id], return_counts=True)
                slot_counts_dict = dict(zip(unique_slots, slot_counts))
                if spec_slot_id in slot_counts_dict:
                    spec_slot_count = slot_counts_dict[spec_slot_id]
                else:
                    spec_slot_count = 0
                class_images.append({
                    "image_path": paths[im_ind],
                    "spec_slot_count": spec_slot_count,
                    "mem_inds": mem_inds[im_ind]
                })
                # class_images_spec_slot_counts.append(spec_slot_count)
                # class_images.append(inputs[im_ind].detach().cpu())
                # class_images_inds.append(mem_inds[im_ind])


        for ex_ind, example in enumerate(target.tolist()): # example is the class id, which must be an integer
            for slice_ind in range(num_slices):
                unique_slots, slot_counts = np.unique(mem_inds[ex_ind, :, :, slice_ind], return_counts=True)
                slot_counts_list = np.column_stack((unique_slots, slot_counts))
                for count in slot_counts_list:
                    slice_memory_counts[slice_ind][count[0], example] = slice_memory_counts[slice_ind][count[0], example] + count[1]

        read = agent.net.crumbr.reconstruct_fbank_top1(extracted)
        direct = agent.net.block(extracted)
        output = agent.net.block(read)
        direct = direct.detach().cpu()
        output = output.detach().cpu()

        if agent.gpu:
            target = target.cpu()

        accs_out = accuracy(output, target, metric_topk)
        accs_dir = accuracy(direct, target, metric_topk)

        for k_ind, k in enumerate(metric_topk):
            accs_out_avg[k].update(accs_out[k_ind], inputs.size(0))
            accs_dir_avg[k].update(accs_dir[k_ind], inputs.size(0))

    full_test_accs_mem = [acc_out.avg for acc_out in list(accs_out_avg.values())]
    full_test_accs_direct = [acc_dir.avg for acc_dir in list(accs_dir_avg.values())]

    test_time = batch_timer.toc()

    print(' * Original test Acc: A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(
        test_acc_out=full_test_accs_mem[0], test_acc_direct=full_test_accs_direct[0], time=test_time))

    total_path = get_out_path(args)

    # get images with specified slot most represented
    #sorted_images = [im for _, im in sorted(zip(class_images_spec_slot_counts, class_images), reverse=True)][0:topk_images]
    #_, sorted_images, sorted_images_inds = map(list, zip(*sorted(zip(class_images_spec_slot_counts, class_images, class_images_inds), reverse=True)))[0:topk_images]
    # sorted_images = class_images[0:topk_images]
    # sorted_images_inds = class_images_inds[0:topk_images]
    sorted_class_images = sorted(class_images, key=lambda x: x["spec_slot_count"], reverse=True)[0:topk_images]

    for im_ind, im_dict in enumerate(sorted_class_images):
        print(str(im_ind) + ": " + im_dict["image_path"])
        im_pil = Image.open(im_dict["image_path"])
        im_pil = im_pil.resize((224, 224))
        #im_pil.save(os.path.join(total_path, "slot" + str(spec_slot_id) + "_slice" + str(spec_slice_id) + "_n" + str(im_ind) + "_im.jpeg"))
        binary_array = im_dict["mem_inds"][:, :, spec_slice_id] == spec_slot_id
        binary_im = make_binary_image(binary_array)
        binary_im = binary_im.resize((224, 224))
        #binary_im.save(os.path.join(total_path, "slot" + str(spec_slot_id) + "_slice" + str(spec_slice_id) + "_n" + str(im_ind) + "_slot_inst.jpeg"))

        slot_indicator = white_to_transparent(ImageOps.colorize(binary_im.convert("L"), black="white", white="blue").convert("RGBA"))

        im_pil = im_pil.convert("RGBA")
        im_pil.paste(slot_indicator, (0,0), slot_indicator)
        im_pil = im_pil.convert("RGB")
        im_pil.save(os.path.join(total_path, "slot" + str(spec_slot_id) + "_slice" + str(spec_slice_id) + "_class" + str(spec_class_id) + "_n" + str(im_ind) + "_combined.jpeg"))

    # Make pictures coloring multiple hypotheses
    # class_images has all images from that class
    # for im_ind, im_dict in enumerate(class_images):
    #     im_pil = Image.open(im_dict["image_path"])
    #     im_pil = im_pil.resize((224, 224))
    #     binary_array_buttons_text = (im_dict["mem_inds"][:, :, spec_slice_id] == 201) | (im_dict["mem_inds"][:, :, spec_slice_id] == 205) | (im_dict["mem_inds"][:, :, spec_slice_id] == 211) | (im_dict["mem_inds"][:, :, spec_slice_id] == 217)
    #     binary_array_can_text = (im_dict["mem_inds"][:, :, spec_slice_id] == 197)
    #     binary_array_background = (im_dict["mem_inds"][:, :, spec_slice_id] == 32) | (im_dict["mem_inds"][:, :, spec_slice_id] == 48)
    # 
    #     binary_im_buttons_text = make_binary_image(binary_array_buttons_text).resize((224, 224)).convert("L")
    #     binary_im_can_text = make_binary_image(binary_array_can_text).resize((224, 224)).convert("L")
    #     binary_im_background = make_binary_image(binary_array_background).resize((224, 224)).convert("L")
    # 
    #     buttons_text_indicator = white_to_transparent(ImageOps.colorize(binary_im_buttons_text, black="white", white="red").convert("RGBA"))
    #     can_text_indicator = white_to_transparent(ImageOps.colorize(binary_im_can_text, black="white", white="yellow").convert("RGBA"))
    #     background_indicator = white_to_transparent(ImageOps.colorize(binary_im_background, black="white", white="blue").convert("RGBA"))
    # 
    #     background_indicator.paste(buttons_text_indicator, (0, 0), buttons_text_indicator)
    #     background_indicator.paste(can_text_indicator, (0, 0), can_text_indicator)
    #     combined = background_indicator
    # 
    #     im_pil = im_pil.convert("RGBA")
    #     im_pil.paste(combined, (0, 0), combined)
    #     im_pil = im_pil.convert("RGB")
    #     im_pil.save(os.path.join(total_path, "combined_slice" + str(spec_slice_id) + "_class" + str(spec_class_id) + "_n" + str(im_ind) + "_combined.jpeg"))


    for slice_ind in range(num_slices):
        memory_counts_df = pd.DataFrame(slice_memory_counts[slice_ind], columns=[
            "scissors", "cups", "cans", "remote controls", "mobile phones", "balls", "light bulbs", "markers", "plug adapters", "glasses"
        ])
        memory_counts_df.to_csv(os.path.join(total_path, "class_memory_slot_counts_slice" + str(slice_ind) + ".csv"))


    # Given a slot ID with slice ID and class ID, find how many copies of this are present for each image in the test set of that class.
    # Sort the test set by instances of this slot, then get the top 25.
    # Save the top 25 images (either from stored data or via task_filelists). Create images of the same size showing the locations of the hypotheses and save those.

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

    # forcodebook model
    parser.add_argument('--n_memblocks', type=int, default=100, help="Number of memory slots to keep in memory")
    parser.add_argument('--memblock_length', type=int, default=512, help="Feature dim per memory slot to keep in memory")
    parser.add_argument('--memory_init_strat', type=str, default="random_distmatch_sparse", help="Memory initialization strategy to use. Possible values include 'random_everyperm', ''random_standard_normal', and 'zeros' (zeros only useful if loading pretrained augmem). PLEASE NOTE: this may be overridden by loading a pretrainedcodebook matrix.")
    parser.add_argument('--visualize', default=False, action='store_true', dest='visualize',
                        help="To visualize memory and attentions (only valid for Crumb")

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
    parser.add_argument('--n_classes', default=1000, type=int, dest='n_classes', help = "Number of classes the model is trained on")

    parser.add_argument('--spec_slot_id', type=int, default=0, help="slot ID to visualize")

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
