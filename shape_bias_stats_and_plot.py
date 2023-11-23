import sys
import argparse
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import copy

def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs")
    parser.add_argument('--scenario', type=str, default="class_instance")
    parser.add_argument('--topk', type=int, default=None, help="e.g. 1 for top1 acc, 5 for top5 acc. If left to 'None', look for files without topk suffix")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

plt.style.use("fivethirtyeight")
plt.rcParams["font.family"] = "Nimbus Sans"

if args.scenario == "class_instance":
    datasets = ["ImageNet-Pretraining", "CORe50", "Toybox", "iLab", "iCub", "iLab+CORe50", "Online-CIFAR100", "Online-ImageNet"]
    dataset_ids = ["imagenet-pretraining", "core50", "toybox", "ilab2mlight", "icubworldtransf", "ilab2mlight+core50", "cifar100", "imagenet"]
    dataset_labels = ["ImageNet\n(Pretraining)", "CORe50", "Toybox", "iLab", "iCub", "iLab+CORe50", "Online\nCIFAR100", "Online\nImageNet"]
elif args.scenario == "class_iid":
    datasets = ["CORe50", "Toybox", "iLab", "iCub", "iLab+CORe50"]
    dataset_ids = ["core50", "toybox", "ilab2mlight", "icubworldtransf", "ilab2mlight+core50"]
    dataset_labels = ["CORe50", "Toybox", "iLab", "iCub", "iLab+CORe50"]

perturbation_names = [
    "Spatial perturbation",
    "Feature perturbation",
    "Style perturbation"
]

crumb_adv_values = {
    "Spatial perturbation": [],
    "Feature perturbation": [],
    "Style perturbation": []
}
errors = {
    "Spatial perturbation": [],
    "Feature perturbation": [],
    "Style perturbation": []
}
sigs = {
    "Spatial perturbation": [],
    "Feature perturbation": [],
    "Style perturbation": []
}

for idx, dataset in enumerate(dataset_ids):

    print("=============================")
    print("RESULTS FOR DATASET:", dataset)
    print("=============================")

    perturbations = [
        "spatial_perturb_acc",
        "feature_perturb_acc",
        "style_perturb_acc"
    ]
    offline_suffix = ""
    pt_infix = ""
    if dataset == "imagenet-pretraining":
        dataset = "imagenet"
        offline_suffix = "_offline"
        pt_infix = "_pretrain"
        scenario = "iid"
        ablations = ("unablated", "direct_loss_only")
    elif dataset in ["imagenet", "cifar100"]:
        scenario = "class_iid"
        ablations = ("unablated", "stock_pretrain_image_replay")
    else:
        scenario = args.scenario
        ablations = ("unablated", "stock_pretrain_image_replay")

    all_dfs = {scenario: {}}

    print("Paired-sample t-test results (all paired with unablated/no perturbation): ")

    for ablation in ablations:
        if "MobileNet" in ablation:
            model = "MobileNet"
        else:
            model = "SqueezeNet"

        if args.topk is None:
            topk_suffix = ""
        else:
            topk_suffix = "_top" + str(args.topk)

        run_dfs = []
        for run in range(args.n_runs):
            try:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + dataset + pt_infix + "_" + ablation + "/" + scenario + "/Crumb_" + model
                    + offline_suffix + "/runs-" + str(run) + "/shape_bias_accs_run" + str(run) + topk_suffix + ".csv"))
            except FileNotFoundError:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + dataset + pt_infix + "_" + ablation + "/" + scenario + "/AugMem_" + model
                    + offline_suffix + "/runs-" + str(run) + "/shape_bias_accs_run" + str(run) + topk_suffix +  ".csv"))

        all_dfs[scenario][ablation] = pd.concat(run_dfs)

        if ablation == "unablated":
            unablated_run_dfs = copy.deepcopy(run_dfs)
        else:  # if ablation != "unablated"
            print('===========================')
            print(ablation + ", " + scenario + ":")
            print("Mean final accuracy for no perturbation:" + str(all_dfs[scenario][ablation]["unablated_acc"].mean()))
            for p, perturbation in enumerate(perturbations):
                if perturbation == "style_perturb_acc" and not dataset == "imagenet":
                    crumb_adv_values[perturbation_names[p]].append(np.nan)
                    errors[perturbation_names[p]].append(np.nan)
                    sigs[perturbation_names[p]].append(False)
                    continue
                print("------------------")
                print("Mean final " + perturbation + ": " + str(all_dfs[scenario][ablation][perturbation].mean()))
                rel_drop_unablated = (all_dfs[scenario]["unablated"]["unablated_acc"] - all_dfs[scenario]["unablated"][perturbation]) / all_dfs[scenario]["unablated"]["unablated_acc"]
                rel_drop_thisablation = (all_dfs[scenario][ablation]["unablated_acc"] - all_dfs[scenario][ablation][perturbation]) / all_dfs[scenario][ablation]["unablated_acc"]
                print("Relative drop in accuracy for unablated, " + perturbation + ": " + str(rel_drop_unablated.mean()))
                print("Relative drop in accuracy for " + ablation + ", " + perturbation + ": " + str(rel_drop_thisablation.mean()))
                crumb_adv = rel_drop_thisablation.mean() - rel_drop_unablated.mean()
                print("CRUMB advantage: " + str(round(crumb_adv, 5)))

                rel_drops_unablated = [(dfs["unablated_acc"] - dfs[perturbation]) / dfs["unablated_acc"] for dfs in unablated_run_dfs]
                rel_drops_thisablation = [(dfs["unablated_acc"] - dfs[perturbation]) / dfs["unablated_acc"] for dfs in run_dfs]

                mean_diffs = [rel_drops_thisablation[i].mean() - rel_drops_unablated[i].mean() for i in range(args.n_runs)]
                # print(mean_diffs)
                sem = stats.tstd(mean_diffs)

                print("CRUMB advantage SEM:", sem)

                # print("paired-samples t-test:")
                # print(stats.ttest_rel(rel_drop_unablated, rel_drop_thisablation))
                print("Wilcoxon signed-rank test:")
                result = stats.wilcoxon(rel_drop_unablated, rel_drop_thisablation)
                print("p =", result.pvalue)
                sig = result.pvalue < 0.05 / 6
                print("corrected p < 0.05?", sig)

                crumb_adv_values[perturbation_names[p]].append(crumb_adv)
                errors[perturbation_names[p]].append(sem)
                sigs[perturbation_names[p]].append(sig)


# Colors and hatches
colors = {
    "Spatial perturbation": "#F7464A",
    "Feature perturbation": "#4D9FF0",
    "Style perturbation": "#C17ACD"
}
hatches = {
    "Spatial perturbation": "/",
    "Feature perturbation": "o",
    "Style perturbation": ""
}

# Plotting
width = 0.25  # bar width
fig, ax = plt.subplots(figsize=(12, 8))

# Create bars with error bars
bars = []
for idx, (cond, vals) in enumerate(crumb_adv_values.items()):
    bar = ax.bar(np.arange(len(datasets)) + idx * width, vals, width, color=colors[cond], yerr=errors[cond], capsize=10,
           hatch=hatches[cond], label=cond, edgecolor='black', linewidth=0.7)
    bars.append(bar)

if args.scenario == "class_instance":
    ax.legend(loc='upper right', fontsize=20) #, ncols=3, borderpad=0.3, columnspacing=0.5)
    ax.set_ylim([-0.225, 0.24])
    low_ast_offset = 0.038
    high_ast_offset = 0.0005
    plt.xticks(rotation=60)
else:
    ax.legend(loc='upper left', fontsize=20)
    ax.set_ylim([-0.34, 0.3])
    low_ast_offset = 0.05
    high_ast_offset = 0.0005
    plt.xticks(rotation=45)
ax.tick_params(axis='both', which='major', labelsize=22)
for i, bar in enumerate(ax.patches):
    bar.set_edgecolor('black')

# Add asterisks for statistically significant bars
for idx, (cond, significance) in enumerate(sigs.items()):
    for i, sig in enumerate(significance):
        if sig:
            if bars[idx][i].get_height() >= 0:  # Bars above zero
                height = bars[idx][i].get_height() + errors[cond][i] + high_ast_offset  # offset by 0.01 for visibility
            else:  # Bars below zero
                height = bars[idx][i].get_height() - errors[cond][i] - low_ast_offset  # offset by 0.05 for visibility below bar
            ax.text(bars[idx][i].get_x() + bars[idx][i].get_width()/2.0, height, '*', ha='center', va='bottom',
                    color='black', fontsize=28, fontweight='bold')

# Formatting
ax.set_xticks(np.arange(len(datasets)) + width)
ax.set_xticklabels(dataset_labels)
ax.set_ylabel("Relative accuracy advantage of CRUMB", fontsize=24)
#ax.set_xlabel("Datasets", fontsize=20)
# ax.set_title("Relative advantage across different datasets", fontsize=20)
ax.axhline(0, color='black', linewidth=2)  # Thick horizontal line at y=0


# Add divider between ImageNet and other datasets
if args.scenario == "class_instance":
    ax.axvline(x=0.75, color='gray', linestyle='--')

fig.set_facecolor('white')
ax.set_facecolor('white')
fig.patch.set_edgecolor('white')
plt.tight_layout()
plt.savefig("shape_bias.png", dpi=600, facecolor="white")






