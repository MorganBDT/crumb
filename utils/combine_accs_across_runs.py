# For results folders where runs were done separately - .csv result files are in subfolders with names like "runs-2".

import os
import sys
import argparse
import pandas as pd


def get_args(argv):
    p = argparse.ArgumentParser()

    p.add_argument('-d', '--directory', type=str, help="The results folder within which to combine runs. Formatted like 'core50_outputs/class_iid/Crumb_SqueezeNet'")
    p.add_argument('--separate_runs', nargs='+', default=None, help='Specify which runs to combine. Use like --separate_runs 0 1 2 3 4. Make sure the run numbers are sorted in ascending order')

    # return parsed arguments
    args = p.parse_args(argv)
    return args


args = get_args(sys.argv[1:])

csv_names = [
    "top1_test_1st_mem_all_runs.csv",
    "top1_test_all_mem_all_runs.csv",
    "top1_test_current_task_mem_all_runs.csv",
    "top1_test_1st_direct_all_runs.csv",
    "top1_test_all_direct_all_runs.csv",
    "top1_test_current_task_direct_all_runs.csv",
    "top5_test_1st_mem_all_runs.csv",
    "top5_test_all_mem_all_runs.csv",
    "top5_test_current_task_mem_all_runs.csv",
    "top5_test_1st_direct_all_runs.csv",
    "top5_test_all_direct_all_runs.csv",
    "top5_test_current_task_direct_all_runs.csv",
]

dirs = [os.path.join(args.directory, "runs-" + str(r)) for r in args.separate_runs]

print(dirs)

for csv in csv_names:
    if os.path.isfile(os.path.join(dirs[0], csv)):
        dfs = [pd.read_csv(os.path.join(d, csv), header=None) for d in dirs]
        combined_df = pd.concat(dfs)
        combined_df.to_csv(os.path.join(args.directory, csv), header=False, index=False)


