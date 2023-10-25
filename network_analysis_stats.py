import sys
import pandas as pd
import scipy.stats as stats

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "core50"
runs = 5

control_scenario = "unablated"

if dataset == "core50":
    ablations = (
        "unablated",
        "cutlayer_3",
        "pretrained_image_replay",
        "stock_pretrain_image_replay",
        "no_pretrain",
        "cifar100_pretrain",
        "MobileNet_bigbatches",
        "stdnml",
        "uniform",
        "distmatch_dense",
        "no_replay",
        "half_mem_cap",
        "quarter_mem_cap",
        "direct_loss_in_streaming",
        "no_direct_loss_in_pt",
        "only_direct_loss_in_streaming",
        "1_memory_blocks",
        "2_memory_blocks",
        "4_memory_blocks",
        "8_memory_blocks",
        "16_memory_blocks",
        #"32_memory_blocks",
        #"64_memory_blocks",
        #"128_memory_blocks",
        "512_memory_blocks",
        "4dim_memory_blocks",
        "16dim_memory_blocks",
        "32dim_memory_blocks",
        "16dim_memory_blocks_adj",
        "32dim_memory_blocks_adj",
        "freeze_memory",
        "pt_weights_only",
        "pt_memblocks_only",
        "pretrained_image_replay",
        "merec_instead_of_crumb",
        # "memcap_400",
        # "memcap_800",
        # "memcap_1600",
        # "memcap_3200",
        # "memcap_6400",
        # "stock_pretrain_image_replay_memcap_400",
        # "stock_pretrain_image_replay_memcap_800",
        # "stock_pretrain_image_replay_memcap_1600",
        # "stock_pretrain_image_replay_memcap_3200",
        # "stock_pretrain_image_replay_memcap_6400",
    )
    # memcap = 6400
    # control_scenario = "memcap_" + str(memcap)
    # ablations = (
    #     "memcap_" + str(memcap),
    #     "stock_pretrain_image_replay_memcap_" + str(memcap),
    # )
else:
    ablations = (
        "unablated",
        "stock_pretrain_image_replay",
    )

if dataset == "cifar100" or dataset == "imagenet":
    scenarios = ("class_iid",)
else:
    scenarios = ("class_iid", "class_instance")

all_dfs = {scenario: {} for scenario in scenarios}

print("Paired-sample t-test results (all paired with" + control_scenario + "): ")

for ablation in ablations:
    if "MobileNet" in ablation:
        model = "MobileNet"
    else:
        model = "SqueezeNet"

    for scenario in scenarios:
        run_dfs = []
        if ablation in ["1_memory_blocks", "1_memory_block"]:
            runs_to_do = 1
        else:
            runs_to_do = runs
        for run in range(runs_to_do):
            try:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + dataset + "_" + ablation + "/" + scenario + "/Crumb_" + model
                    + "/runs-" + str(run) + "/batchwise_accs_run" + str(run) + ".csv"))
            except FileNotFoundError:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + dataset + "_" + ablation + "/" + scenario + "/AugMem_" + model
                    + "/runs-" + str(run) + "/batchwise_accs_run" + str(run) + ".csv"))

        all_dfs[scenario][ablation] = pd.concat(run_dfs)

        if ablation == control_scenario:  # For 1 memory block
            run0_df_control = run_dfs[0]

        if not ablation == control_scenario:
            print(ablation + ", " + scenario + ":")
            if ablation in ["1_memory_blocks", "1_memory_block"]:
                print(stats.ttest_rel(run0_df_control["accuracy"], all_dfs[scenario][ablation]["accuracy"]))
            else:
                print(stats.ttest_rel(all_dfs[scenario][control_scenario]["accuracy"], all_dfs[scenario][ablation]["accuracy"]))






