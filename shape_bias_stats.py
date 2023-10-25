import sys
import argparse
import pandas as pd
import scipy.stats as stats
import copy

def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="toybox", help="Dataset, e.g. core50, toybox, ilab2mlight, imagenet, cifar100")
    parser.add_argument('--include_style_transfer', default=False, action='store_true', help='Include style-transferred ImageNet perturbation')
    parser.add_argument('--pretraining', default=False, action='store_true', help="Whether or not we are testing the results of pretraining")
    parser.add_argument('--n_runs', type=int, default=5, help="Number of runs")
    parser.add_argument('--topk', type=int, default=None, help="e.g. 1 for top1 acc, 5 for top5 acc. If left to 'None', look for files without topk suffix")

    parser.add_argument('--ablations', nargs='+', help="List of ablations, separated by spaces. e.g. '--ablations stock_pretrain_image_replay other_ablation'", required=True)

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])


if args.pretraining:
    scenarios = ("iid",)
else:
    if args.dataset == "cifar100" or args.dataset == "imagenet":
        scenarios = ("class_iid",)
    else:
        scenarios = ("class_instance", "class_iid")

# Note: "unablated" must always be first
ablations = ("unablated",) + tuple(args.ablations)

perturbations = [
    "spatial_perturb_acc",
    "feature_perturb_acc",
]
if args.dataset == "imagenet" and args.include_style_transfer:
    perturbations.append("style_perturb_acc")

all_dfs = {scenario: {} for scenario in scenarios}

print("Paired-sample t-test results (all paired with unablated/no perturbation): ")


for scenario in scenarios:

    for ablation in ablations:
        if "MobileNet" in ablation:
            model = "MobileNet"
        else:
            model = "SqueezeNet"

        if args.pretraining:
            offline_suffix = "_offline"
        else:
            offline_suffix = ""

        if args.pretraining:
            pt_infix = "_pretrain"
        else:
            pt_infix = ""

        if args.topk is None:
            topk_suffix = ""
        else:
            topk_suffix = "_top" + str(args.topk)

        run_dfs = []
        for run in range(args.n_runs):
            try:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + args.dataset + pt_infix + "_" + ablation + "/" + scenario + "/Crumb_" + model
                    + offline_suffix + "/runs-" + str(run) + "/shape_bias_accs_run" + str(run) + topk_suffix + ".csv"))
            except FileNotFoundError:
                run_dfs.append(pd.read_csv(
                    "./ablation_study/" + args.dataset + pt_infix + "_" + ablation + "/" + scenario + "/AugMem_" + model
                    + offline_suffix + "/runs-" + str(run) + "/shape_bias_accs_run" + str(run) + topk_suffix +  ".csv"))

        all_dfs[scenario][ablation] = pd.concat(run_dfs)

        if ablation == "unablated":
            unablated_run_dfs = copy.deepcopy(run_dfs)
        else:  # if ablation != "unablated"
            print('===========================')
            print(ablation + ", " + scenario + ":")
            print("Mean final accuracy for no perturbation:" + str(all_dfs[scenario][ablation]["unablated_acc"].mean()))
            for perturbation in perturbations:
                print("------------------")
                print("Mean final " + perturbation + ": " + str(all_dfs[scenario][ablation][perturbation].mean()))
                rel_drop_unablated = (all_dfs[scenario]["unablated"]["unablated_acc"] - all_dfs[scenario]["unablated"][perturbation]) / all_dfs[scenario]["unablated"]["unablated_acc"]
                rel_drop_thisablation = (all_dfs[scenario][ablation]["unablated_acc"] - all_dfs[scenario][ablation][perturbation]) / all_dfs[scenario][ablation]["unablated_acc"]
                print("Relative drop in accuracy for unablated, " + perturbation + ": " + str(rel_drop_unablated.mean()))
                print("Relative drop in accuracy for " + ablation + ", " + perturbation + ": " + str(rel_drop_thisablation.mean()))
                print("CRUMB advantage: " + str(round(rel_drop_thisablation.mean() - rel_drop_unablated.mean(), 5)))
                print("CRUMB advantage SEM: ")

                rel_drops_unablated = [(dfs["unablated_acc"] - dfs[perturbation]) / dfs["unablated_acc"] for dfs in unablated_run_dfs]
                rel_drops_thisablation = [(dfs["unablated_acc"] - dfs[perturbation]) / dfs["unablated_acc"] for dfs in run_dfs]

                mean_diffs = [rel_drops_thisablation[i].mean() - rel_drops_unablated[i].mean() for i in range(args.n_runs)]
                # print(mean_diffs)
                print(stats.tstd(mean_diffs))

                # print("paired-samples t-test:")
                # print(stats.ttest_rel(rel_drop_unablated, rel_drop_thisablation))
                print("Wilcoxon signed-rank test:")
                result = stats.wilcoxon(rel_drop_unablated, rel_drop_thisablation)
                print("p =", result.pvalue)
                print("corrected p < 0.05?", result.pvalue < 0.05 / 6)






