import os
import sys
import pandas as pd


# USAGE: python imagenet_dirmap.py <path to imagenet dataset directory>

# PLEASE NOTE: the images in the ImageNet2012 folder labelled "val" are used as the test set for training purposes.

if len(sys.argv) > 1:
    DATA_DIR = sys.argv[1]
else:
    DATA_DIR = "./../data/imagenet"

# Get class names
class_names = [
    file for file in os.listdir(os.path.join(DATA_DIR, "train"))
    if os.path.isdir(os.path.join(DATA_DIR, "train", file))
]
class_names.sort()
class_dicts_df = pd.DataFrame([{"class": class_names[i], "label": i} for i in range(len(class_names))])
class_dicts_df.to_csv("imagenet_classes.csv", index=False)
imagenet_class_inds = pd.read_csv('imagenet_classes.csv', index_col=0, usecols=["class", "label"], squeeze=True).to_dict()

image_list = []
for train_test_idx, train_test in enumerate(["train", "val"]):
    for img_class in class_names:
        img_files = [f for f in os.listdir(os.path.join(DATA_DIR, train_test, img_class)) if f.endswith(".JPEG")]
        for fname in img_files:
            image_list.append({
                "class": imagenet_class_inds[img_class],
                "object": 0,
                "session": train_test_idx,
                "im_path": os.path.join(train_test, img_class, fname),
            })

img_df = pd.DataFrame(image_list)
img_df = img_df.sort_values(by=["class", "object", "session", "im_path"], ignore_index=True)
img_df["im_num"] = img_df.groupby(["class", "object", "session"]).cumcount() + 1

img_df.to_csv("imagenet_dirmap.csv")
print(img_df.head())
