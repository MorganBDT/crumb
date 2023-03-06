import os
import sys
import pandas as pd

# This script uses the task_filelists generated for CRUMB experiments to create "imagenet_class_order.txt" files for
# REMIND experiments that use different data orderings.

filelist_location = sys.argv[1]

for run in range(0, 10):

    filelists_files = [
        os.path.join(
            filelist_location,
            "imagenet_task_filelists/class_iid/run"+str(run)+"/stream/train_task_0"+str(t)+"_filelist.txt"
        ) for t in range(0, 10)
    ]

    class_order_dfs = []
    for filelist in filelists_files:

        # Read task_filelist file into dataframe
        df = pd.read_csv(filelist, sep=" ", header=None)
        df.columns = ["im_path", "class_num"]

        # Get alphanumeric identifier for imagenet class, e.g. n03450230, from path of the form:
        # train/n03450230/n03450230_29892.JPEG
        df["class_str"] = df.apply(lambda row: row["im_path"].split("/")[1], axis=1)

        classes_df = df.groupby("class_str", as_index=False).first()

        classes_df = classes_df.sort_values(["class_num"], ascending=True)

        print(filelist + ": " + str(len(classes_df)) + " classes")

        class_order_dfs.append(classes_df)

    class_df = pd.concat(class_order_dfs)

    assert(len(class_df) == 1000)
    assert(len(class_df["class_str"].unique()) == 1000)

    class_df["class_str"].to_csv("imagenet_class_order_run" + str(run) + ".txt", index=False, header=False)
