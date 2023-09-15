# REQUIRES sshpass. Install with "sudo apt install sshpass"

import os
import sys
import argparse
import subprocess
import pandas as pd


def get_args(argv):
    p = argparse.ArgumentParser()

    p.add_argument('-s', '--ssh', type=str, help="The ssh location you're downloading from, e.g. user@asdfasdf.onion")
    p.add_argument('-p', '--password', type=str)
    p.add_argument('--proj_path', type=str, default="~/projects/crumb")
    p.add_argument('-e', '--exp_dirs', nargs="+")
    p.add_argument('-m', '--exp_dirs_model_names', nargs="+")
    p.add_argument('-b', '--dest_dir', type=str, default="./bin")
    p.add_argument('-c', '--csv_filename', type=str, default="exp_dirs")
    p.add_argument('--no_download', default=False, action="store_true", help="Use directories already in the 'bin' folder")
    p.add_argument('--no_process', default=False, action="store_true", help="Download the directories from the server into the bin, but don't process them any further")

    # return parsed arguments
    args = p.parse_args(argv)
    return args

print(os.getcwd())

args = get_args(sys.argv[1:])

if not os.path.exists("./bin"):
    os.makedirs("./bin")

if not os.path.exists("./plots"):
    os.makedirs("./plots")

if args.dest_dir is not None and not os.path.exists(args.dest_dir):
    os.makedirs(args.dest_dir)

if not args.no_download:
    dl_processes = []
    for exp_dir in args.exp_dirs:
        command = ["sshpass", "-p", args.password, "scp", "-r", os.path.join(args.ssh + ":" + args.proj_path, exp_dir), "./bin"]
        print("Starting download from server with command: " + " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        dl_processes.append(process)

experiments_iid = []
experiments_cliid = []
experiments_clinst = []
for i, exp_dir in enumerate(args.exp_dirs):
    if not args.no_download:
        stdout, stderr = dl_processes[i].communicate()
        print("------------ stdout from download for " + exp_dir + " ------------\n" + str(stdout))
        print("------------ stderr from download for " + exp_dir + " ------------\n" + str(stderr))

    if args.exp_dirs_model_names is None:
        model_name = "SqueezeNet"
    elif len(args.exp_dirs_model_names[i]) == 1:
        model_name = args.exp_dirs_model_names[0]
    else:
        model_name = args.exp_dirs_model_names[i]

    if os.path.exists(os.path.join("./bin", exp_dir, "iid")):
        experiments_iid.append({
            "experiment_name": exp_dir + "_iid",
            "dir": os.path.join(args.dest_dir, exp_dir, "iid", "Crumb_" + model_name + "_offline")
        })
    if os.path.exists(os.path.join("./bin", exp_dir, "class_iid")):
        experiments_cliid.append({
            "experiment_name": exp_dir + "_cliid",
            "dir": os.path.join(args.dest_dir, exp_dir, "class_iid", "Crumb_" + model_name)
        })
    if os.path.exists(os.path.join("./bin", exp_dir, "class_instance")):
        experiments_clinst.append({
            "experiment_name": exp_dir + "_clinst",
            "dir": os.path.join(args.dest_dir, exp_dir, "class_instance", "Crumb_" + model_name)
        })
if not args.no_process:
    plot_commands = []
    if len(experiments_iid) > 0:
        pd.DataFrame(experiments_iid).to_csv(os.path.join("./bin", args.csv_filename + "_iid.csv"), index=False)
        plot_commands.append(("python plot_epochs.py --n_class_per_task 1000 --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_iid.csv") + " --direct_acc").split(" "))
    if len(experiments_cliid) > 0:
        pd.DataFrame(experiments_cliid).to_csv(os.path.join("./bin", args.csv_filename + "_cliid.csv"), index=False)
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix cliid_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_cliid.csv") + " --direct_acc").split(" "))
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix cliid_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_cliid.csv") + " --direct_acc --task1").split(" "))
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix cliid_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_cliid.csv") + " --direct_acc --current_task").split(" "))
    if len(experiments_clinst) > 0:
        pd.DataFrame(experiments_clinst).to_csv(os.path.join("./bin", args.csv_filename + "_clinst.csv"), index=False)
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix clinst_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_clinst.csv") + " --direct_acc").split(" "))
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix clinst_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_clinst.csv") + " --direct_acc --task1").split(" "))
        plot_commands.append(("python plot_cl_experiments.py --n_class_per_task 2 --task1_acc_cutoff 0 --plot_name_prefix clinst_ --dirs_csv " + os.path.join(args.dest_dir, args.csv_filename + "_clinst.csv") + " --direct_acc --current_task").split(" "))

    move_command = "mv " + os.path.join(os.getcwd(), "bin", "*") + " " + args.dest_dir
    print("Moving files from bin with command: " + move_command)
    process = subprocess.Popen(move_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("------------ stdout from moving files ------------\n" + str(stdout))
    print("------------ stderr from moving files ------------\n" + str(stderr))

    processes = []
    for cmd in plot_commands:
        processes.append(subprocess.Popen(cmd, stdout=subprocess.PIPE))
    for process in processes:
        stdout, stderr = process.communicate()
        print("------------ stdout from plotting ------------\n" + str(stdout))
        print("------------ stderr from plotting ------------\n" + str(stderr))