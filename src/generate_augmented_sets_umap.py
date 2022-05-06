import os
import glob
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def load_augmented_data(augmented_dir, augmented_sets, bands):
    # process for converting image to tensor
    process = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize(512),
                torchvision.transforms.ToTensor()
            ])

    image_array = []
    main_df = None

    for augmented_set in augmented_sets:
        for band in bands:
            image_dir = f"{augmented_dir}{augmented_set}/{band}/"

            # load results table, add augmentation label, and merge with main
            results_df = pd.read_csv(f"{image_dir}results.csv")
            results_df.insert(len(results_df.columns), "augmnt", augmented_set)
            if main_df is None:
                main_df = results_df
            else:
                main_df = pd.concat([main_df, results_df])

            # load all pngs from the directory
            image_list = glob.glob(f"{image_dir}*.png")
            image_file_names = [os.path.basename(x) for x in image_list]
            # sort images based on the file name to match uuid order in table
            image_file_names = sorted(image_file_names)

            for image_file in image_file_names:
                image_path = f"{image_dir}{image_file}"
                # load file and process
                with Image.open(image_path) as img:
                    tensor = process(img)
                    # flatten the image and add to array
                    image_array.append(tensor.flatten().numpy())

    # convert the entire image array to numpy array
    image_array = np.array(image_array)
    
    return image_array, main_df


def umap_loop(image_array, main_df, n=2):
    neighbors_list = [5, 30, 100, 200]
    for neighbors in neighbors_list:
        print(f"  neighbors = {neighbors}")
        if n > 2:
            min_dist = 0.0
        else:
            min_dist = 0.1
        umap = UMAP(random_state=42, n_components=n, n_neighbors=neighbors,
                    min_dist=0.0)
        umap = umap.fit_transform(image_array)
        for i in range(n):
            main_df[f"umap{i+1}_{neighbors}"] = umap[:, i]
        # main_df[f"umap2_{neighbors}"] = umap[:, 1]
    return main_df


def main_loop(augmented_dir, augmented_sets, bands, output_file, n=2):
    print("loading datasets")
    image_array, main_df = load_augmented_data(augmented_dir, augmented_sets, bands)
    print("completed!")
    print("running UMAP iterations... ")
    umap_loop(image_array, main_df, n=n)
    print("completed!")
    print(f"saving results to {output_file}")
    main_df.to_csv(output_file)


if __name__ == "__main__":
    bands = ["Cband", "Sband", "Xband"]
    augmented_dir = "data/augmented_dataset/"

    augmented_sets = ["rotation_0", "rotation_15", "rotation_30", "rotation_45",
                      "rotation_60", "rotation_75", "rotation_90", "rotation_105",
                      "rotation_120", "rotation_135", "rotation_150",
                      "rotation_165", "rotation_180"]
    for n in [2, 5, 10]:
        main_loop(augmented_dir, augmented_sets, bands,
                  f"umap_rotation_{n}D.csv", n)
