import sys
import pandas as pd
import scipy.stats as stats

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "core50"
runs = 5

if dataset == "core50":
    ablations = (
        "unablated",
        "cutlayer_3",
        "pretrained_image_replay",
        "stock_pretrain_image_replay",
        "no_pretrain",
        "cifar100_pretrain_4epochs",
        "MobileNet_normal_init",
        "stdnml",
        "uniform",
        "distmatch_dense",
        "no_replay",
        "half_mem_cap",
        "quarter_mem_cap",
        "direct_loss_in_streaming",
        "no_direct_loss_in_pt",
        "only_direct_loss_in_streaming",
        "2_augmem_rows",
        "4_augmem_rows",
        "8_augmem_rows",
        "16_augmem_rows",
        "32_augmem_rows",
        "64_augmem_rows",
        "128_augmem_rows",
        "512_augmem_rows",
        "4_augmem_feat",
        "16_augmem_feat",
        "32_augmem_feat",
        "16_augmem_feat_buffer_size_400",
        "32_augmem_feat_buffer_size_800",
        "freeze_memory"
    )
else:
    ablations = (
        "unablated",
        "pretrained_image_replay",
        "stock_pretrain_image_replay",
        "freeze_memory"
    )

if dataset == "cifar100" or dataset == "imagenet":
    scenarios = ("class_iid",)
else:
    scenarios = ("class_iid", "class_instance")

all_dfs = {scenario: {} for scenario in scenarios}

print("Paired-sample t-test results (all paired with unablated): ")

for ablation in ablations:
    if "MobileNet" in ablation:
        model = "MobileNet"
    else:
        model = "SqueezeNet"

    for scenario in scenarios:
        run_dfs = []
        for run in range(runs):
            run_dfs.append(pd.read_csv(
                "./ablation_study/" + dataset + "_" + ablation + "/" + scenario + "/Crumb_" + model
                + "/runs-" + str(run) + "/batchwise_accs_run" + str(run) + ".csv"))

        all_dfs[scenario][ablation] = pd.concat(run_dfs)

        if not ablation == "unablated":
            print(ablation + ", " + scenario + ":")
            print(stats.ttest_rel(all_dfs[scenario]["unablated"]["accuracy"], all_dfs[scenario][ablation]["accuracy"]))






