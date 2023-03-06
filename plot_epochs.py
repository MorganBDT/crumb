import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import math


# This script plots the accuracy, epoch by epoch, during training on a single task.

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

    parser.add_argument('--verbose', default=False, action='store_true', dest='verbose')

    parser.add_argument('--topk', default=1, type=int, help="1 for top1 acc, 5 for top5 acc")


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
    else:
        test_tasks = "all"

    fig, ax = plt.subplots()
    plot_names = []

    experiments = pd.read_csv(args.dirs_csv, keep_default_na=False)
    experiments = experiments[experiments["dir"] != ""]

    max_epochs = 0

    for idx, row in experiments.iterrows():

        hyperparams = pd.read_csv(os.path.join(row["dir"], "hyperparams.csv"), index_col=0, header=None, squeeze=True).to_dict()
        results_by_run = []

        for run in range(int(hyperparams["n_runs"])):
            try:
                csv_path = os.path.join(row["dir"], "all_epochs", "top"+str(args.topk)+"_test_"+test_tasks+"_"+acc_type+"_all_epochs_run"+str(run)+".csv")
                results_by_run.append(pd.read_csv(csv_path, header=None))
            except FileNotFoundError:
                print("Not found: " + csv_path)
                results_by_run.append(pd.read_csv(os.path.join(row["dir"], "top"+str(args.topk)+"_test_all_epochs_run" + str(run) + ".csv"), header=None))

        result = pd.concat(results_by_run, ignore_index=True)

        if args.verbose:
            print(row["experiment_name"])
            print(result)

        x = result.columns
        y = result.mean(axis=0)
        y_err = result.std(axis=0) / math.sqrt(len(result.index))
        name = row["experiment_name"]
        plot_names.append(name)

        ax.errorbar(x, y, y_err, label=name, capsize=2)

        if len(x) > max_epochs:
            max_epochs = len(x)

    if args.n_class_per_task is not None:
        x = list(range(max_epochs))
        y = [1/args.n_class_per_task for _ in range(max_epochs)]  # vector of all the same value
        ax.plot(x, y, 'k-', label='Chance')

    ax.legend()
    ax.set_ylabel('Accuracy')
    ax.set_yticks([t for t in range(0,100,10)])
    ax.set_xlabel('Epoch')
    #ax.set_xticks([t for t in range(max_epochs)])
    #ax.set_xticklabels([t+1 for t in range(max_epochs)]) # Task labelling starts at 1, not 0
    ax.set_title("Task 1 accuracy")

    f_name = "_".join([acc_type, test_tasks]) + "_" + "-".join(plot_names)

    fig.savefig(os.path.join(args.plot_dir, f_name), dpi=300)


if __name__ == '__main__':

    main()


