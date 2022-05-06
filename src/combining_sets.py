#!python3

import os
import glob
import shutil
import numpy as np
import pandas as pd


def combine_sets(dir1, dir2, new_dir, fold_files=None):
    print(f"Creating new combined set at {new_dir}")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    shutil.copytree(dir1, new_dir)
    
    for sub_dir in ["Cband", "Sband", "Xband", "masks"]:
        print(f"Copying images from {os.path.join(dir2, sub_dir)}")
        file_list = glob.glob(os.path.join(dir2, sub_dir, "*.png"))
        for file_path in file_list:
            new_name = f"{1}{os.path.basename(file_path)[1:]}"
            new_path = os.path.join(new_dir, sub_dir, new_name)
            shutil.copyfile(file_path, new_path)
    
    for sub_dir in ["", "Cband", "Sband", "Xband"]:
        print(f"Copying results.csv from {os.path.join(dir2, sub_dir)}")
        results1 = os.path.join(new_dir, sub_dir, "results.csv")
        results2 = os.path.join(dir2, sub_dir, "results.csv")
        df_results1 = pd.read_csv(results1)
        df_results2 = pd.read_csv(results2)
        df_results2["uuid"] += 1000
        df_new_results = pd.concat([df_results1, df_results2])
        df_new_results.reset_index(drop=True, inplace=True)
        
        if fold_files is not None:
            df_new_results["kfold"] = -1
            for fold, fold_file in enumerate(fold_files):
                with open(fold_file) as fin:
                    uuids = [int(x.strip('\n')) for x in fin.readlines()[1:]]
                    new_uuids = uuids + [1000+x for x in uuids]

                idx = df_new_results.query(f"uuid in {new_uuids}").index
                df_new_results.loc[idx, "kfold"] = fold

        df_new_results.to_csv(results1, index=False)


if __name__ == "__main__":
    base_dir = "/home/sdan8/sar-data-fusion-UNet/augmented_dataset/circular_crop/"
    dir1 = os.path.join(base_dir, "rotation_0")
    dir2 = os.path.join(base_dir, "rotation")
    new_dir = os.path.join(base_dir, "rotation_2X")
    fold_files = [f"/home/sdan8/sar-data-fusion-UNet/test_set_fold{i}.txt" for i in range(4)]
    combine_sets(dir1, dir2, new_dir, fold_files)

