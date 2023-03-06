import os
import sys
import argparse
import json
import csv


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, default=None, help="Path of input json")
    parser.add_argument("--run", type=int, default=None, help="Run number (different runs have different data orderings)")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

with open(args.input, 'r') as f:
    full_json = json.load(f)

fmaps = [
    ["base_classes_top1", "test_task1_top1"],
    ["base_classes_top5", "test_task1_top5"],
    ["seen_classes_top1", "test_all_top1"],
    ["seen_classes_top5", "test_all_top5"],
]

if args.run is not None:  # append run number to file name
    for fmap in fmaps:
        fmap[1] = fmap[1] + "_run" + str(args.run)

out_path = args.input[0:-len(args.input.split("/")[-1])]
print("Output directory: " + out_path)

for fmap in fmaps:
    with open(os.path.join(out_path, fmap[1] + ".csv"), 'w') as f:
        write = csv.writer(f)
        write.writerows([full_json[fmap[0]]])
