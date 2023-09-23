import os
import sys
import json
import argparse
import pandas as pd


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None, required=True, help="core50,toybox,ilab2mlight,cifar100")
    parser.add_argument('--scenario', type=str, default=None, required=True, help="class_iid or class_instance")
    parser.add_argument('--num_runs', type=int, default=None, required=True, help="Number of runs")
    parser.add_argument('--results_base_folder_name', type=str, default=None, required=False, help="Base folder name (without e.g. _run0)")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])
dataset = args.dataset
scenario = args.scenario
num_runs = args.num_runs

if args.results_base_folder_name is None:
    results_base_folder_name = dataset + "_" + scenario + "_remind"
else:
    results_base_folder_name = args.results_base_folder_name

if dataset == "core50":
    num_classes = 10
    num_tasks = 5
elif dataset == "toybox":
    num_classes = 12
    num_tasks = 6
elif dataset == "ilab2mlight":
    num_classes = 14
    num_tasks = 7
elif dataset == "cifar100":
    num_classes = 100
    num_tasks = 20
elif dataset == "ilab2mlight+core50":
    num_classes = 24
    num_tasks = 12
elif dataset == "icubworldtransf":
    num_classes = 20
    num_tasks = 10
else:
    raise ValueError("Dataset must be one of core50, toybox, ilab2mlight, cifar100, or ilab2mlight+core50")

top1_test_all_direct_all_runs_list = []
top1_test_1st_direct_all_runs_list = []
top5_test_all_direct_all_runs_list = []
top5_test_1st_direct_all_runs_list = []

for r in range(num_runs):

    fpath = os.path.join(
        "squeezenet_results",
        results_base_folder_name + "_run" + str(r),
        "accuracies_min_trained_0_max_trained_" + str(num_classes) + "_" + results_base_folder_name + "_run" + str(r) + "_task_" + str(num_tasks-1) + ".json"
    )
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
    else:
        print(fpath + " not found")

    top1_test_all_direct_all_runs_list.append(data["seen_classes_top1"])
    top1_test_1st_direct_all_runs_list.append(data["base_classes_top1"])
    top5_test_all_direct_all_runs_list.append(data["seen_classes_top5"])
    top5_test_1st_direct_all_runs_list.append(data["base_classes_top5"])

    print("run " + str(r) + ":")
    print("Values in seen_classes_top1: " + str(len(data["seen_classes_top1"])))
    print("Values in base_classes_top1: " + str(len(data["base_classes_top1"])))
    print("Values in seen_classes_top5: " + str(len(data["seen_classes_top5"])))
    print("Values in base_classes_top5: " + str(len(data["base_classes_top5"])))

top1_test_all_direct_all_runs = pd.DataFrame(top1_test_all_direct_all_runs_list)
top1_test_1st_direct_all_runs = pd.DataFrame(top1_test_1st_direct_all_runs_list)
top5_test_all_direct_all_runs = pd.DataFrame(top5_test_all_direct_all_runs_list)
top5_test_1st_direct_all_runs = pd.DataFrame(top5_test_1st_direct_all_runs_list)

if not os.path.exists(os.path.join("squeezenet_results", results_base_folder_name)):
    os.makedirs(os.path.join("squeezenet_results", results_base_folder_name))

top1_test_all_direct_all_runs.to_csv(os.path.join("squeezenet_results",
                                                  results_base_folder_name,
                                                  "top1_test_all_direct_all_runs.csv"), header=False, index=False)
top1_test_1st_direct_all_runs.to_csv(os.path.join("squeezenet_results",
                                                  results_base_folder_name,
                                                  "top1_test_1st_direct_all_runs.csv"), header=False, index=False)
top5_test_all_direct_all_runs.to_csv(os.path.join("squeezenet_results",
                                                  results_base_folder_name,
                                                  "top5_test_all_direct_all_runs.csv"), header=False, index=False)
top5_test_1st_direct_all_runs.to_csv(os.path.join("squeezenet_results",
                                                  results_base_folder_name,
                                                  "top5_test_1st_direct_all_runs.csv"), header=False, index=False)
