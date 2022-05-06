#!python3

import os
import glob
import shutil
import numpy as np
import pandas as pd


def copy_augmented_images(uuids, dir_path, new_dir):
    # copies all images of the given uuids from the sub_dirs of dir_path
    # to the sub_dirs of the new_dir
    sub_dirs = ["Cband", "Sband", "Xband", "masks"]
    for sub_dir in sub_dirs:
        for uuid in uuids:
            file_path = os.path.join(dir_path, sub_dir, f"{uuid}.png")
            new_path = os.path.join(new_dir, sub_dir, f"{uuid}.png")
            shutil.copyfile(file_path, new_path)


def copy_augmentation_fold_images(fold_dicts, df_results, new_dir):
    # copies augmentated images based on the folds assigned to them in fold_dicts
    # and using the uuid and kfolds in df_reults
    for fold_dict in fold_dicts:
        dir_path = fold_dict["path"]
        folds = fold_dict["folds"]
        for fold in folds:
            uuids = df_results.query(f"kfold == {fold}")["uuid"].values
            uuids = [f"0000{x}"[-4:] for x in uuids]
            copy_augmented_images(uuids, dir_path, new_dir)


def combine_sets(fold_dicts, new_dir, fold_files=None):
    for sub_dir in ["", "Cband", "Sband", "Xband"]:
        results1 = os.path.join(new_dir, sub_dir, "results.csv")
        df_results1 = pd.read_csv(results1)
        df_results1["kfold"] = -1
        for fold, fold_file in enumerate(fold_files):
            with open(fold_file) as fin:
                uuids = [int(x.strip('\n')) for x in fin.readlines()[1:]]

            idx = df_results1.query(f"uuid in {uuids}").index
            df_results1.loc[idx, "kfold"] = fold

        df_results1.to_csv(results1, index=False)
    
    df_results = pd.read_csv(os.path.join(new_dir, "results.csv"))
    copy_augmentation_fold_images(fold_dicts, df_results, new_dir)


if __name__ == "__main__":
    dir1 = "/home/sdan8/sar-data-fusion-UNet/augmented_dataset/rotation_0"
    dir2 = "/home/sdan8/sar-data-fusion-UNet/augmented_dataset/rotation_45"
    dir3 = "/home/sdan8/sar-data-fusion-UNet/augmented_dataset/rotation_90"
    new_dir = "/home/sdan8/sar-data-fusion-UNet/augmented_dataset/rotation_C_norm_1"
    fold_files = [f"/home/sdan8/sar-data-fusion-UNet/test_set_fold{i}.txt" for i in range(4)]
    fold_dicts = [{"path": dir1, "folds": [0, 1]},
                  {"path": dir2, "folds": [2]},
                  {"path": dir3, "folds": [3]}]
    combine_sets(fold_dicts, new_dir, fold_files)

