import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math


# This script plots the accuracy, task by task, in a continual learning experiment with multiple sequential tasks.

def get_args(argv):

    # defining arguments that the user can pass into the program
    parser = argparse.ArgumentParser()

    # plotting parameters
    parser.add_argument('--dirs_csv', type=str, default=None, help=
                        "Directory to a CSV with experiment names in first column (column label 'experiment_name') "
                        "and NN training result paths in second column (column label 'dir')")

    parser.add_argument('--plot_dir', type=str, default="plots", help="Path to directory to store plot(s)")

    parser.add_argument('--direct_acc', default=False, action='store_true', dest='direct_acc', help="Plot direct accuracy instead of mem accuracy ('a-out')")

    parser.add_argument('--n_class_per_task', type=int, default=None, help="Number of classes trained on in each task")
    parser.add_argument('--task1', default=False, action='store_true', dest='task1', help="Plot accuracy on task 1 instead of all tasks trained on so far")
    parser.add_argument('--current_task', default=False, action='store_true', dest='current_task', help="Plot accuracy only on the current task instead of all tasks trained on so far")

    parser.add_argument('--task1_acc_cutoff', type=int, default=-1, help="Only use runs with task1 accuracies >= this")

    parser.add_argument('--verbose', default=False, action='store_true', dest='verbose')

    parser.add_argument('--plot_name_prefix', type=str, default="")


    # return parsed arguments
    args = parser.parse_args(argv)
    return args


def main():

    # get command line arguments
    args = get_args(sys.argv[1:])

    if args.direct_acc:
        acc_type = "direct"
    else:
        acc_type = "mem"
    if args.task1:
        test_tasks = "1st"
    elif args.current_task:
        test_tasks = "current_task"
    else:
        test_tasks = "all"

    fig, ax = plt.subplots()
    plot_names = []

    experiments = pd.read_csv(args.dirs_csv, keep_default_na=False)
    experiments = experiments[experiments["dir"] != ""]

    max_tasks = 0

    for idx, row in experiments.iterrows():

        # hyperparams = pd.read_csv(os.path.join(row["dir"], "hyperparams.csv"), index_col=0, header=None, squeeze=True).to_dict()

        result = pd.read_csv(os.path.join(row["dir"], "top1_test_"+test_tasks+"_"+acc_type+"_all_runs.csv"), header=None)

        # Use only runs with task1 acc greater than a threshold
        result = result[result[0] > args.task1_acc_cutoff].copy()

        if args.verbose:
            print(row["experiment_name"])
            print(result)

        x = result.columns
        y = result.mean(axis=0)
        y_err = result.std(axis=0) / math.sqrt(len(result.index))
        name = row["experiment_name"]
        plot_names.append(name)

        ax.errorbar(x, y, y_err, label=name, capsize=2)

        if len(x) > max_tasks:
            max_tasks = len(x)

    if args.n_class_per_task is not None:
        x = list(range(max_tasks))
        y = [100/((task+1)*args.n_class_per_task) for task in range(len(x))]
        ax.plot(x, y, 'k-', label='Chance')

    ax.legend()
    ax.set_ylabel('Accuracy')
    ax.set_yticks([t for t in range(0,100,10)])
    ax.set_xlabel('Task')
    ax.set_xticks([t for t in range(max_tasks)])
    ax.set_xticklabels([t+1 for t in range(max_tasks)]) # Task labelling starts at 1, not 0
    if args.task1:
        ax.set_title("Task 1 accuracy")
    elif args.current_task:
        ax.set_title("Current task accuracy")
    else:
        ax.set_title("Accuracy on all seen classes")

    try:
        f_name = args.plot_name_prefix + "_".join([acc_type, test_tasks]) + "_" + "-".join(plot_names)
        fig.savefig(os.path.join(args.plot_dir, f_name), dpi=300)
    except OSError:
        f_name = args.plot_name_prefix + "_".join([acc_type, test_tasks])
        fig.savefig(os.path.join(args.plot_dir, f_name), dpi=300)


if __name__ == '__main__':

    main()


