#!python3

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd


def copy_augmented_images(sub_dirs, uuids, dir_path, new_dir):
    # copies all images of the given uuids from the sub_dirs of dir_path
    # to the sub_dirs of the new_dir
    for sub_dir in sub_dirs:
        print(f"    Copying images from {os.path.join(dir_path, sub_dir)}")
        for uuid in uuids:
            file_path = os.path.join(dir_path, sub_dir, f"{uuid}.png")
            new_path = os.path.join(new_dir, sub_dir, f"{uuid}.png")
            shutil.copyfile(file_path, new_path)


def copy_augmentation_fold_images(fold_dicts, df_results, new_dir, sub_dirs):
    # copies augmentated images based on the folds assigned to them in fold_dicts
    # and using the uuid and kfolds in df_reults
    for fold_dict in fold_dicts:
        dir_path = fold_dict["path"]
        folds = fold_dict["folds"]
        print(f"  Copying data for folds: {folds} from {dir_path}")
        for fold in folds:
            uuids = df_results.query(f"aug_fold == {fold}")["uuid"].values
            uuids = [f"0000{x}"[-4:] for x in uuids]
            copy_augmented_images(sub_dirs, uuids, dir_path, new_dir)


def combine_sets(fold_dicts, new_dir, results_file, sub_dirs=["Cband", "Sband", "Xband", "masks"]):
    print(f"Creating {new_dir}")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    # shutil.copytree(dir1, new_dir)
    os.mkdir(new_dir)
    for sub_dir in sub_dirs:
        os.mkdir(os.path.join(new_dir, sub_dir))
    
    band_dirs = [x for x in sub_dirs if "band" in x] + [""]
    for sub_dir in band_dirs:
        new_results_file = os.path.join(new_dir, sub_dir, "results.csv")
        shutil.copyfile(results_file, new_results_file)

    df_results = pd.read_csv(os.path.join(new_dir, "results.csv"))
    copy_augmentation_fold_images(fold_dicts, df_results, new_dir, sub_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", type=str)
    parser.add_argument("--fold1", type=str)
    parser.add_argument("--new_dir", type=str)
    parser.add_argument("--results_csv", type=str,
                        default="data/results_with_aug_folds.csv")
    parser.add_argument("--bands", type=str, default="CSX")
    
    args = parser.parse_args()
    dir1 = args.fold0
    dir2 = args.fold1
    new_dir = args.new_dir
    results_file = args.results_csv
    sub_dirs = [f"{x}band" for x in args.bands] + ["masks"]
    fold_dicts = [{"path": dir1, "folds": [0]},
                  {"path": dir2, "folds": [1]}]
    combine_sets(fold_dicts, new_dir, results_file, sub_dirs=sub_dirs)

