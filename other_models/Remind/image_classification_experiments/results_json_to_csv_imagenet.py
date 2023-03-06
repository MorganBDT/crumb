import json
import pandas as pd
import os

results_base_folder_name = "remind_squeezenet_imagenet_origpqreshape"
num_runs = 5
num_tasks = 10

max_classes = 1000

top1_test_all_direct_all_runs_list = []
top1_test_1st_direct_all_runs_list = []
top5_test_all_direct_all_runs_list = []
top5_test_1st_direct_all_runs_list = []

for r in range(num_runs):

    fpath = os.path.join(
        "streaming_experiments",
        results_base_folder_name + "_run" + str(r),
        "accuracies_min_trained_0_max_trained_" + str(max_classes) + ".json"
    )
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
    else:  # specific exception for REMIND run3 on imagenet
        fpath = os.path.join(
            "streaming_experiments",
            results_base_folder_name + "_run" + str(r),
            "accuracies_min_trained_0_max_trained_900.json"
        )
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)

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

if not os.path.exists(os.path.join("streaming_experiments", results_base_folder_name)):
    os.makedirs(os.path.join("streaming_experiments", results_base_folder_name))

top1_test_all_direct_all_runs.to_csv(os.path.join("streaming_experiments",
                                                  results_base_folder_name,
                                                  "top1_test_all_direct_all_runs.csv"), header=False, index=False)
top1_test_1st_direct_all_runs.to_csv(os.path.join("streaming_experiments",
                                                  results_base_folder_name,
                                                  "top1_test_1st_direct_all_runs.csv"), header=False, index=False)
top5_test_all_direct_all_runs.to_csv(os.path.join("streaming_experiments",
                                                  results_base_folder_name,
                                                  "top5_test_all_direct_all_runs.csv"), header=False, index=False)
top5_test_1st_direct_all_runs.to_csv(os.path.join("streaming_experiments",
                                                  results_base_folder_name,
                                                  "top5_test_1st_direct_all_runs.csv"), header=False, index=False)
