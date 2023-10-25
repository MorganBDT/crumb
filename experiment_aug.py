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


def run(args, run, task_range=None):

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
        'run': run,
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
        'crumb_cut_layer': args.crumb_cut_layer,
        'memory_size': args.memory_size,
        'n_workers': args.n_workers,
        'n_memblocks': args.n_memblocks,
        'memblock_length': args.memblock_length,
        'memory_init_strat': args.memory_init_strat,
        'freeze_batchnorm': args.freeze_batchnorm,
        'freeze_memory': args.freeze_memory,
        'storage_type': args.storage_type,
        'replay_coef': args.replay_coef,
        'replay_times': args.replay_times,
        'augment_replays': args.augment_replays,
        'use_random_resize_crops': args.use_random_resize_crops,
        'ntask': len(tasks),
        'visualize': args.visualize,
        'replay_in_1st_task': args.replay_in_1st_task,
        'n_epoch_first_task': args.n_epoch_first_task,
        'pretrained_weights': args.pretrained_weights,
        'pretrained_dataset_no_of_classes': args.pretrained_dataset_no_of_classes,
        'pretraining': args.pretraining,
        'continuing': args.continuing,
        'pt_only_codebook_out_loss': args.pt_only_codebook_out_loss,
        'plus_direct_loss': args.plus_direct_loss,
        'direct_loss_only': args.direct_loss_only,
        'full_out_dir': get_out_path(args)
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
    elif args.dataset == "imagenet900":
        agent_config["n_class"] = 900
    elif args.dataset == "ilab2mlight+core50":
        agent_config["n_class"] = 24
    elif args.dataset == "icubworldtransf":
        agent_config["n_class"] = 20
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
    elif args.dataset in ["toybox", "ilab2mlight", "cifar100", "imagenet", "imagenet900", "ilab2mlight+core50", "icubworldtransf"]:
        # image transformations
        composed = transforms.Compose(
            [transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # get test data
        test_data = datasets.Generic_Dataset(
            dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario, offline=args.offline,
            run=run, train=False, transform=composed)
    else:
        raise ValueError("Invalid dataset name, try 'core50', 'toybox', 'ilab2mlight', 'cifar100', 'imagenet', or 'imagenet900'")

    if args.validate:
        # splitting test set into test and validation
        test_size = int(0.75 * len(test_data))
        val_size = len(test_data) - test_size
        test_data, val_data = torch.utils.data.random_split(test_data, [test_size, val_size])
    else:
        val_data = None

    all_accs, mem_stats = train(agent, composed, args, run, tasks, active_out_nodes, test_data, val_data, task_range=task_range)

    return all_accs, mem_stats


def train(agent, transforms, args, run, tasks, active_out_nodes, test_data, val_data, task_range=None):

    if args.offline:
        print('============BEGINNING OFFLINE LEARNING============')
    else:
        print('============BEGINNING STREAM LEARNING============')

    all_accs = {
        "mem": {
            "test_all": {"all_epochs": [], "best_epochs": []},
            "test_1st": {"all_epochs": [], "best_epochs": []},
            "test_current_task": {"all_epochs": [], "best_epochs": []},
            "val_all": {"all_epochs": [], "best_epochs": []}
        },
        "direct": {
            "test_all": {"all_epochs": [], "best_epochs": []},
            "test_1st": {"all_epochs": [], "best_epochs": []},
            "test_current_task": {"all_epochs": [], "best_epochs": []},
            "val_all": {"all_epochs": [], "best_epochs": []}
        }
    }

    mem_stats = []

    ntask = len(tasks)  # number of tasks
    if task_range is None:
        task_range = range(ntask)
    else:
        assert task_range.start >= 0, "Start of task range (first number given to --task_range) must be 0 or above"
        assert task_range.stop <= ntask, "End of task range (second number given to --task_range) must be at least 1 less than number of tasks (counting from 0)"

    if args.continuing:
        print("INITIAL ACCURACY CHECK FOR SANITY:")
        agent.active_out_nodes = active_out_nodes[0]
        test_inds_1st = [i for i in range(len(test_data)) if
                         test_data.labels[i] in active_out_nodes[0]]  # retrive first task
        task_test_data_1st = torch.utils.data.Subset(test_data, test_inds_1st)
        test_loader_1st = torch.utils.data.DataLoader(
            task_test_data_1st, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
        test_accs_mem_1st, test_accs_direct_1st, test_time_1st = agent.validation(test_loader_1st, args.acc_topk)
        print(' * Test Acc (1st task): A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(
                test_acc_out=test_accs_mem_1st[0], test_acc_direct=test_accs_direct_1st[0], time=test_time_1st))

        # If in the specific scenario of continuing to train on subsequent tasks after pretraining on the first task:
        if not args.pretraining and int(args.task_range[0]) == 1:
            # Get replay buffer contents for task 0
            task = 0
            # get training data pertaining to chosen scenario, task, run
            if args.dataset == 'core50':
                train_data = datasets.CORE50(
                    dataroot=args.dataroot, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            elif args.dataset in ["toybox", "ilab2mlight", "cifar100", "imagenet", "ilab2mlight+core50", "icubworldtransf"]:
                train_data = datasets.Generic_Dataset(
                    dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            else:
                raise ValueError(
                    "Invalid dataset name, try 'core50', 'toybox', 'ilab2mlight', 'cifar100', 'imagenet'")
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
            agent.updateStorage_epoch(train_loader, run, task)

            all_accs["direct"]["test_all"]["all_epochs"].append([])
            all_accs["direct"]["test_1st"]["all_epochs"].append([])
            all_accs["direct"]["test_current_task"]["all_epochs"].append([])
            all_accs["direct"]["val_all"]["all_epochs"].append([])
            all_accs["mem"]["test_all"]["all_epochs"].append([])
            all_accs["mem"]["test_1st"]["all_epochs"].append([])
            all_accs["mem"]["test_current_task"]["all_epochs"].append([])
            all_accs["mem"]["val_all"]["all_epochs"].append([])

            all_accs["direct"]["test_1st"]["all_epochs"][task].append(test_accs_direct_1st)
            all_accs["mem"]["test_1st"]["all_epochs"][task].append(test_accs_mem_1st)
            all_accs["direct"]["test_all"]["all_epochs"][task].append(test_accs_direct_1st)
            all_accs["mem"]["test_all"]["all_epochs"][task].append(test_accs_mem_1st)
            all_accs["direct"]["test_current_task"]["all_epochs"][task].append(test_accs_direct_1st)
            all_accs["mem"]["test_current_task"]["all_epochs"][task].append(test_accs_mem_1st)

            all_accs["direct"]["test_1st"]["best_epochs"].append(test_accs_direct_1st)
            all_accs["mem"]["test_1st"]["best_epochs"].append(test_accs_mem_1st)
            all_accs["direct"]["test_all"]["best_epochs"].append(test_accs_direct_1st)
            all_accs["mem"]["test_all"]["best_epochs"].append(test_accs_mem_1st)
            all_accs["direct"]["test_current_task"]["best_epochs"].append(test_accs_direct_1st)
            all_accs["mem"]["test_current_task"]["best_epochs"].append(test_accs_mem_1st)

    # iterate over tasks
    for task in task_range:

        print('=============Training Task ' + str(task) + '=============')

        agent.active_out_nodes = active_out_nodes[task]

        print('Active output nodes for this task: ')
        print(agent.active_out_nodes)

        all_accs["direct"]["test_all"]["all_epochs"].append([])
        all_accs["direct"]["test_1st"]["all_epochs"].append([])
        all_accs["direct"]["test_current_task"]["all_epochs"].append([])
        all_accs["direct"]["val_all"]["all_epochs"].append([])
        all_accs["mem"]["test_all"]["all_epochs"].append([])
        all_accs["mem"]["test_1st"]["all_epochs"].append([])
        all_accs["mem"]["test_current_task"]["all_epochs"].append([])
        all_accs["mem"]["val_all"]["all_epochs"].append([])

        if (args.n_epoch_first_task is not None) and (task == 0):
            n_epoch = args.n_epoch_first_task
        else:
            n_epoch = args.n_epoch
        for epoch in range(n_epoch):

            print('===' + args.agent_name + '; Epoch ' + str(epoch) + '; RUN ' + str(run) + '; TASK ' + str(task))

            # get training data pertaining to chosen scenario, task, run
            if args.dataset == 'core50':
                train_data = datasets.CORE50(
                    dataroot=args.dataroot, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            elif args.dataset in ["toybox", "ilab2mlight", "cifar100", "imagenet", "imagenet900", "ilab2mlight+core50", "icubworldtransf"]:
                train_data = datasets.Generic_Dataset(
                    dataroot=args.dataroot, dataset=args.dataset, filelist_root=args.filelist_root, scenario=args.scenario,
                    offline=args.offline, run=run, batch=task, transform=transforms)
            else:
                raise ValueError("Invalid dataset name, try 'core50', 'toybox', 'ilab2mlight', 'cifar100', 'imagenet', or 'imagenet900'")

            # get train loader
            train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            if args.validate:
                # then test and val data are subsets, not datasets and need to be dealt with accordingly
                # get test data only for the seen classes
                test_inds = [i for i in range(len(test_data)) if test_data.dataset.labels[test_data.indices[i]] in agent.active_out_nodes] # list(range(len(test_data)))
                task_test_data = torch.utils.data.Subset(test_data, test_inds)
                #labels = [task_test_data[i] for i in range(len(task_test_data))]
                test_loader = torch.utils.data.DataLoader(
                            task_test_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)
                val_inds = [i for i in range(len(val_data)) if val_data.dataset.labels[val_data.indices[i]] in agent.active_out_nodes]
                task_val_data = torch.utils.data.Subset(val_data, val_inds)
                val_loader = torch.utils.data.DataLoader(
                        task_val_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            else:
                # get test data only for the seen classes
                test_inds = [i for i in range(len(test_data)) if test_data.labels[i] in agent.active_out_nodes] # list(range(len(test_data)))
                task_test_data = torch.utils.data.Subset(test_data, test_inds)
                test_loader = torch.utils.data.DataLoader(
                            task_test_data, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

                test_inds_1st = [i for i in range(len(test_data)) if test_data.labels[i] in active_out_nodes[0]] # retrive first task
                task_test_data_1st = torch.utils.data.Subset(test_data, test_inds_1st)
                test_loader_1st = torch.utils.data.DataLoader(
                            task_test_data_1st, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

                test_inds_current_task = [i for i in range(len(test_data)) if test_data.labels[i] in active_out_nodes[task] and (task == 0 or test_data.labels[i] not in active_out_nodes[task-1])]  # retrieve the task we just trained on
                task_test_data_current_task = torch.utils.data.Subset(test_data, test_inds_current_task)
                test_loader_current_task = torch.utils.data.DataLoader(
                            task_test_data_current_task, batch_size=args.batch_size, shuffle=False, num_workers = args.n_workers, pin_memory=True)

            # learn
            mem_stats_samples = agent.learn_stream(train_loader, run, task, epoch)

            # Store mem stats in a list of dictionaries, to be converted to dataframe -> csv
            if len(mem_stats_samples) > 0:
                mem_stats_init = {"run": run, "task": task, "epoch": epoch}
                mem_stats.append({**mem_stats_init, **mem_stats_samples[0]})
                mem_stats.append({**mem_stats_init, **mem_stats_samples[1]})
                print("===Codebook stats from start of task " + str(task) + ", epoch " + str(epoch) + "===")
                print(mem_stats_samples[0])
                print("===Codebook stats from end of task " + str(task) + ", epoch " + str(epoch) + "===")
                print(mem_stats_samples[1])

            # validate if applicable
            if args.validate:
                val_accs_mem, val_accs_direct, val_time = agent.validation(val_loader, args.acc_topk)
                print(' * Val Acc: A-out {val_acc_out:.3f}, A-direct {val_acc_direct:.3f}, Time: {time:.2f}'.format(
                    val_acc_out=val_accs_mem[0], val_acc_direct=val_accs_direct[0], time=val_time))
                all_accs["direct"]["val_all"]["all_epochs"][task].append(val_accs_direct)
                all_accs["mem"]["val_all"]["all_epochs"][task].append(val_accs_mem)

            test_accs_mem, test_accs_direct, test_time = agent.validation(test_loader, args.acc_topk)
            print(' * Test Acc: A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(
                test_acc_out=test_accs_mem[0], test_acc_direct=test_accs_direct[0], time=test_time))
            all_accs["direct"]["test_all"]["all_epochs"][task].append(test_accs_direct)
            all_accs["mem"]["test_all"]["all_epochs"][task].append(test_accs_mem)

            if not args.pretraining:
                # For the first task, task1 accuracy and current task accuracy are identical to overall task accuracy.
                # No need to repeat these computations.
                if task == 0:
                    test_accs_mem_1st, test_accs_direct_1st, test_time_1st = test_accs_mem, test_accs_direct, test_time
                    test_accs_mem_current_task, test_accs_direct_current_task, test_time_current_task = test_accs_mem, test_accs_direct, test_time
                else:
                    test_accs_mem_1st, test_accs_direct_1st, test_time_1st = agent.validation(test_loader_1st, args.acc_topk)
                    test_accs_mem_current_task, test_accs_direct_current_task, test_time_current_task = agent.validation(test_loader_current_task, args.acc_topk)

                print(' * Test Acc (1st task): A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(
                        test_acc_out=test_accs_mem_1st[0], test_acc_direct=test_accs_direct_1st[0], time=test_time_1st))
                all_accs["direct"]["test_1st"]["all_epochs"][task].append(test_accs_direct_1st)
                all_accs["mem"]["test_1st"]["all_epochs"][task].append(test_accs_mem_1st)

                print(' * Test Acc (current task): A-out {test_acc_out:.3f}, A-direct {test_acc_direct:.3f}, Time: {time:.2f}'.format(
                        test_acc_out=test_accs_mem_current_task[0], test_acc_direct=test_accs_direct_current_task[0], time=test_time_current_task))
                all_accs["direct"]["test_current_task"]["all_epochs"][task].append(test_accs_direct_current_task)
                all_accs["mem"]["test_current_task"]["all_epochs"][task].append(test_accs_mem_current_task)

            if args.visualize:
                attread_filename = 'visualization/' + args.scenario + '/' + args.scenario + '_run_' + str(run) + '_task_' + str(task) + '_epoch_' + str(epoch)
                agent.visualize_att_read(attread_filename)
                agent.visualize_memory(attread_filename)

            if args.keep_best_net_all_tasks or (args.keep_best_task1_net and task == 0):
                # Save state of model
                torch.save(agent.net.model.state_dict(), os.path.join(get_out_path(args), "model_state_epoch_" + str(epoch) + ".pth"))

            if args.save_model or args.save_model_every_epoch:
                print("Saving model state after epoch " + str(epoch))
                if args.save_model_every_epoch:
                    model_save_path = os.path.join(get_out_path(args), "CRUMB_run" + str(run) + "_task" + str(task) + "_epoch" + str(epoch))
                else:
                    model_save_path = os.path.join(get_out_path(args), "CRUMB_run" + str(run))
                agent.save_model(model_save_path)
                agent.save_memory(model_save_path)

                if task == 0 and epoch == 0:
                    agent.save_memory(os.path.join(get_out_path(args), "CRUMB_run" + str(run) + "_start"), custom_memory=agent.net.initial_memory)

            # Decrement LR (has no effect if args.lr_decrement is 0)
            agent.decrement_lr(args.lr_decrement, args.lr_min)

            # Freeze feature extractor if past a certain epoch
            if args.freeze_feature_extract_after_epoch is not None and epoch >= args.freeze_feature_extract_after_epoch:
                print("FREEZING FEATURE EXTRACTOR AFTER EPOCH: " + str(epoch))
                for param in agent.net.FeatureExtractor.parameters():
                    param.requires_grad = False

            if args.end_pretraining_after_epoch is not None and epoch >= args.end_pretraining_after_epoch:
                agent.config["pretraining"] = False

        if (args.keep_best_net_all_tasks or (args.keep_best_task1_net and task == 0)) and args.n_epoch_first_task > 1:
            if args.best_net_direct:
                comp_test_accs_all_epochs = all_accs["direct"]["test_all"]["all_epochs"]
            else:
                comp_test_accs_all_epochs = all_accs["mem"]["test_all"]["all_epochs"]
            # Reload state of network when it had highest top-1 test accuracy on first task
            comp_top1_accs = [epch[0] for epch in comp_test_accs_all_epochs[task]]
            max_top1_acc = max(comp_top1_accs)
            max_top1_acc_ind = comp_top1_accs.index(max_top1_acc)
            print("Test top-1 accs on task " + str(task) + ": " + str(comp_top1_accs))
            print("Loading model parameters with this max test acc: " + str(max_top1_acc))
            agent.net.model.load_state_dict(torch.load(
                os.path.join(get_out_path(args), "model_state_epoch_" + str(max_top1_acc_ind) + ".pth"))
            )

            reload_test_accs_mem, reload_test_accs_direct, test_time = agent.validation(test_loader, args.acc_topk)
            if args.best_net_direct:
                print(' * Direct test Acc (after reloading best model): {acc:.3f}, Time: {time:.2f}'.format(acc=reload_test_accs_direct[0], time=test_time))
                assert abs(reload_test_accs_direct[0] - max_top1_acc) < 0.1, "Test accuracy of reloaded model does not match original highest test accuracy. Is the model saving and loading its state correctly?"
            else:
                print(' * Mem test Acc (after reloading best model): {acc:.3f}, Time: {time:.2f}'.format(acc=reload_test_accs_mem[0], time=test_time))
                assert abs(reload_test_accs_mem[0] - max_top1_acc) < 0.1, "Test accuracy of reloaded model does not match original highest test accuracy. Is the model saving and loading its state correctly?"

            # Set the test/val accs to be stored for this task to those corresponding to the best-performing network
            test_accs_direct = all_accs["direct"]["test_all"]["all_epochs"][task][max_top1_acc_ind]
            test_accs_mem = all_accs["mem"]["test_all"]["all_epochs"][task][max_top1_acc_ind]
            test_accs_direct_1st = all_accs["direct"]["test_1st"]["all_epochs"][task][max_top1_acc_ind]
            test_accs_mem_1st = all_accs["mem"]["test_1st"]["all_epochs"][task][max_top1_acc_ind]
            test_accs_direct_current_task = all_accs["direct"]["test_current_task"]["all_epochs"][task][max_top1_acc_ind]
            test_acc_mem_current_task = all_accs["mem"]["test_current_task"]["all_epochs"][task][max_top1_acc_ind]
            if args.validate:
                val_accs_direct = all_accs["direct"]["val_all"]["all_epochs"][task][max_top1_acc_ind]
                val_accs_mem = all_accs["mem"]["val_all"]["all_epochs"][task][max_top1_acc_ind]

            # Delete saved network states
            for save_num in range(len(all_accs["direct"]["test_all"]["all_epochs"][task])):
                os.remove(os.path.join(get_out_path(args), "model_state_epoch_" + str(save_num) + ".pth"))


        # after all the epochs, store test_acc
        all_accs["direct"]["test_all"]["best_epochs"].append(test_accs_direct)
        all_accs["mem"]["test_all"]["best_epochs"].append(test_accs_mem)

        if not args.pretraining:
            all_accs["direct"]["test_1st"]["best_epochs"].append(test_accs_direct_1st)
            all_accs["direct"]["test_current_task"]["best_epochs"].append(test_accs_direct_current_task)

            all_accs["mem"]["test_1st"]["best_epochs"].append(test_accs_mem_1st)
            all_accs["mem"]["test_current_task"]["best_epochs"].append(test_accs_mem_current_task)

        # same with val acc
        if val_data is not None:
            all_accs["direct"]["val_all"]["best_epochs"].append(val_accs_direct)
            all_accs["mem"]["val_all"]["best_epochs"].append(val_accs_mem)

    if args.save_model or args.save_model_every_epoch:
        agent.save_model(os.path.join(get_out_path(args), "CRUMB_run"+str(run)))
        agent.save_memory(os.path.join(get_out_path(args), "CRUMB_run"+str(run)))

    return all_accs, mem_stats

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
    parser.add_argument('--specific_runs', nargs='+', type=int, default=None, help='Do specific runs (data orderings) instead of the first n runs. Overrides the --n_runs parameter. Use like --specific_runs 1 3 5')
    parser.add_argument('--task_range', nargs='+', type=int, default=None, help='Specify a range of tasks to train the network on (e.g. for splitting pretraining on the first task and stream learning on subsequent tasks into separate jobs). Use like --task_range <first task> <last task>, e.g. --task_range 0 0 does only the first task, and --task_range 1 9 does the second through 10th tasks (count starts at 0)')

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
    parser.add_argument('--freeze_batchnorm', default=False, dest='freeze_batchnorm', action='store_true')
    parser.add_argument('--crumb_cut_layer', type=int, default=12, help="Layer after which to cut the network and insert a crumb reconstructor module. Numbering starts at 1")
    parser.add_argument('--freeze_memory', default=False, dest='freeze_memory', action='store_true')
    parser.add_argument('--freeze_feature_extract', default=False, dest='freeze_feature_extract', action='store_true')
    parser.add_argument('--freeze_feature_extract_after_epoch', type=int, default=None, help="Freeze the feature extractor after a certain epoch in the first task")
    parser.add_argument('--end_pretraining_after_epoch', type=int, default=None, help="End the 'pretraining' status after a certain epoch in the first task")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--memory_weights', type=str, default=None,
                        help="The path to the file for the memory weights (*.pth).")
    parser.add_argument('--n_epoch', type=int, default=1, help="Number of epochs to train")
    parser.add_argument('--n_epoch_first_task', type=int, default=None, help="Number of epochs to train on the first task (may be different from n_epoch, which is used for the other tasks)")
    parser.add_argument('--keep_best_task1_net', default=False, dest='keep_best_task1_net', action='store_true', help="When training for multiple epochs on task 1, retrieve the network state (among those after each epoch) with best testing accuracy for learning subsequent tasks")
    parser.add_argument('--keep_best_net_all_tasks', default=False, dest='keep_best_net_all_tasks', action='store_true', help="When training for multiple epochs on more than one task: for each task, retrieve the network state (among those after each epoch) with best testing accuracy for learning subsequent tasks")
    parser.add_argument('--best_net_direct', default=False, dest='best_net_direct', action='store_true', help="This param determines what accuracy is used to select the best model weights for the first task. If this flag is included, the 'direct' accuracy is used, without memory bank. Otherwise, the 'out' accuracy is used, which incorporates the memory bank")

    # keep track of validation accuracy
    parser.add_argument('--validate', default=False, action='store_true', dest='validate',
                        help="To keep track of validation accuracy or not")

    # for replay models
    parser.add_argument('--storage_type', type=str, default="feature", help="Type of replay storage, e.g. 'feature' or 'image'")
    parser.add_argument('--memory_size', type=int, default=417, help="Number of training examples to keep in memory")
    parser.add_argument('--replay_coef', type=float, default=5,
                        help="The coefficient for replays. Larger means less plasilicity. ")
    parser.add_argument('--replay_times', type=int, default=1, help="The number of times to replay per batch. ")
    parser.add_argument('--replay_in_1st_task', default=False, action='store_true', dest='replay_in_1st_task')
    parser.add_argument('--augment_replays', default=False, action='store_true', dest='augment_replays', help='Do data augmentation (e.g. random horizontal flip) on replay examples')
    parser.add_argument('--use_random_resize_crops', default=False, action='store_true', dest='use_random_resize_crops', help='Apply REMINDs feature-level data augmentation')

    # For loss function-related ablation experiments
    parser.add_argument('--pt_only_codebook_out_loss', default=False, action='store_true', dest='pt_only_codebook_out_loss', help='ours - direct loss ablation')
    parser.add_argument('--plus_direct_loss', default=False, action='store_true', dest='plus_direct_loss', help='ours + direct loss ablation')
    parser.add_argument('--direct_loss_only', default=False, action='store_true', dest='direct_loss_only', help='direct loss ablation')

    # for CRUMB model
    parser.add_argument('--n_memblocks', type=int, default=256, help="Number of memory blocks to keep in memory")
    parser.add_argument('--memblock_length', type=int, default=16, help="Feature dim per memory block to keep in memory")
    parser.add_argument('--memory_init_strat', type=str, default="random_distmatch_sparse", help="Memory initialization strategy to use. Possible values include 'random_everyperm', ''random_standard_normal', and 'zeros' (zeros only useful if loading pretrained codebook). PLEASE NOTE: this may be overridden by loading a pretrainedCRUMB matrix.")
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

    parser.add_argument('--save_model', default=False, action='store_true', dest='save_model', help='Save the trained model weights and memory bank at end of training')
    parser.add_argument('--save_model_every_epoch', default=False, action='store_true', dest='save_model_every_epoch', help='Save a copy of the trained model and weights after every single epoch')

    parser.add_argument('--pretraining', default=False, action='store_true', dest='pretraining', help='Streamline training process for pretraining (e.g. no storage of replay examples, no redundant tests on 1st/current task')
    parser.add_argument('--continuing', default=False, action='store_true', dest='continuing', help='Continue training a loaded model without randomly re-initializing the final layer')

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
    args.dataroot = os.path.join(os.getcwd(),args.dataroot)
    args.output_dir = os.path.join(os.getcwd(),args.output_dir)

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
            if "," in str(val):
                val = "\"" + str(val) + "\""
            f.write("{key},{val}\n".format(key=key, val=val))

    all_accs_all_runs = []
    mem_stats_all_runs = []

    if args.specific_runs is None:
        runs = range(args.n_runs)
    else:
        runs = [int(r) for r in args.specific_runs]

    if args.task_range is None:
        task_range = None
    else:
        task_limits = [int(t) for t in args.task_range]
        task_range = range(task_limits[0], task_limits[1]+1)

    if args.pretraining:
        task_tests = ["test_all"]
    else:
        task_tests = ["test_all", "test_1st", "test_current_task", "val_all"]

    for r in runs:
        print('=============Stream Learning Run ' + str(r) + '=============')

        # setting seed for reproducibility
        torch.manual_seed(r)
        np.random.seed(r)
        random.seed(r)

        all_accs, mem_stats = run(args, r, task_range=task_range)

        # save accs for all epochs on this run
        p = os.path.join(total_path, "all_epochs")
        if not os.path.exists(p):
            os.makedirs(p)
        for acc_type in ["mem", "direct"]:
            for tasks_tested in task_tests:
                for k_ind, k in enumerate(args.acc_topk):
                    # extract top-k acc from list of lists of lists to create a mere list of lists
                    df = pd.DataFrame([[epochs_accs[k_ind] for epochs_accs in tasks_accs] for tasks_accs in all_accs[acc_type][tasks_tested]["all_epochs"]])
                    df.to_csv(os.path.join(p, "top" + str(k) + "_" + tasks_tested + "_" + acc_type + '_all_epochs_run' + str(r) + ".csv"), index=False, header=False)
        all_accs_all_runs.append(all_accs)
        if len(mem_stats) > 0:
            mem_stats_all_runs.extend(mem_stats)


    # converting list of list of testing accuracies for each run to a dataframe and saving
    for acc_type in ["mem", "direct"]:
        for tasks_tested in task_tests:
            for k_ind, k in enumerate(args.acc_topk):
                # For each run, extract top-k acc from list of lists of lists to create a mere list of lists
                best_accs_across_runs = [[tasks_accs[k_ind] for tasks_accs in all_accs_1run[acc_type][tasks_tested]["best_epochs"]] for all_accs_1run in all_accs_all_runs]
                df = pd.DataFrame(best_accs_across_runs)
                df.to_csv(os.path.join(total_path, "top" + str(k) + "_" + tasks_tested + "_" + acc_type + "_all_runs.csv"), index=False, header=False)

    mem_stats_df = pd.DataFrame(mem_stats_all_runs)
    mem_stats_df.to_csv(os.path.join(total_path, "mem_stats.csv"))

    print("Total running time: " + str(time.time() - start_time))

if __name__ == '__main__':

    main()
