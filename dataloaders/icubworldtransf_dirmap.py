import os
import sys
import argparse
import pandas as pd


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', type=str, default="/media/KLAB37/datasets/icubworldtransf")
    parser.add_argument('--n_instances', type=int, default=10)
    parser.add_argument('--every_k_frames', type=int, default=10,
                        help="Sample every k frames. Footage is at 10 fps, so a value of 10 means 1 fps.")
    parser.add_argument('--cam', type=str, default="left", help="Use the 'left' or 'right' cam images")

    return parser.parse_args(argv)


args = get_args(sys.argv[1:])

# These four directories should be inside the dataset_root folder (called something like icubworldtransf)
part_dirs = ["part1", "part2", "part3", "part4"]

sessions = ['MIX', 'ROT2D', 'ROT3D', 'SCALE', 'TRANSL']

class_dict = {}
class_ind = 0
for part_dir in part_dirs:
    assert part_dir in os.listdir(args.dataset_root)
    for class_name in os.listdir(os.path.join(args.dataset_root, part_dir)):
        class_path = os.path.join(part_dir, class_name)
        if os.path.isdir(os.path.join(args.dataset_root, class_path)):
            class_dict[class_name] = class_path
            class_ind += 1

dfs = []
for class_ind, class_name in enumerate(list(class_dict.keys())):
    for instance_ind in range(args.n_instances):
        for session_ind, session in enumerate(sessions):
            session_path = os.path.join(class_dict[class_name], class_name + str(instance_ind+1), session)
            day_dir = sorted([d for d in os.listdir(os.path.join(args.dataset_root, session_path))
                              if os.path.isdir(os.path.join(args.dataset_root, session_path, d))])[0]
            session_path = os.path.join(session_path, day_dir, args.cam)
            ims = sorted([im for im in os.listdir(os.path.join(args.dataset_root, session_path)) if ".jpg" in im])
            df_list = []
            for im_ind, im in enumerate(ims[::args.every_k_frames]):
                row = {
                    "class": class_ind,
                    "object": instance_ind,
                    "session": session_ind,
                    "im_num": im_ind,
                    "im_path": os.path.join(session_path, im),
                }
                df_list.append(row)
            dfs.append(pd.DataFrame(df_list))

img_df = pd.concat(dfs, ignore_index=True)

img_df = img_df.sort_values(by=["class", "object", "session", "im_num"], ignore_index=True)

img_df.to_csv("icubworldtransf_dirmap.csv", index=False)
