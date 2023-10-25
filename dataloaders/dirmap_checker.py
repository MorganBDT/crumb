# Can use this utility script to check if classes are balanced (number of images per class).
import pandas as pd

dataset = "ilab2mlight"

img_df = pd.read_csv(dataset + "_dirmap.csv")

print(img_df["session"].nunique())

# Keep training images only
if dataset == "core50":
    img_df = img_df[~img_df["session"].isin([3, 7, 10])]
elif dataset == "icubworldtransf":
    img_df = img_df[~img_df["session"].isin([0])]
if dataset == "toybox":
    img_df = img_df[~img_df["session"].isin([3, 6, 9])]
if dataset == "ilab2mlight":
    img_df = img_df[~img_df["session"].isin([4, 8])]

print(img_df["class"].value_counts())

