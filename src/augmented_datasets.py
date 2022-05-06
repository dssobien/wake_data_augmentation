#!python3

import os
import glob
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import data_augmentation
import tqdm
import shutil


TRANSFORM_DICTIONARY = {
    "rotation": [data_augmentation.RandomRotation((0, 359), 1, probability=1.0)],
    "perspective_25": [data_augmentation.RandomPerspective(0.25, probability=1.0)],
    "perspective_50": [data_augmentation.RandomPerspective(0.50, probability=1.0)],
    "perspective_75": [data_augmentation.RandomPerspective(0.75, probability=1.0)],
    "perspective_100": [data_augmentation.RandomPerspective(1.00, probability=1.0)],
    "random_crop_768": [data_augmentation.RandomCrop(768, probability=1.0),
                        data_augmentation.Rescale(1024)],
    "random_crop_896": [data_augmentation.RandomCrop(896, probability=1.0),
                        data_augmentation.Rescale(1024)],
    "center_crop_768": [data_augmentation.RandomCrop(768, probability=0.0),
                        data_augmentation.Rescale(1024)],
    "center_crop_896": [data_augmentation.RandomCrop(896, probability=0.0),
                        data_augmentation.Rescale(1024)],
    "random_noise_001": [data_augmentation.RandomNoise(0.001, probability=1.0)],
    "random_noise_003": [data_augmentation.RandomNoise(0.003, probability=1.0)],
    "random_noise_03": [data_augmentation.RandomNoise(0.03, probability=1.0)],
    "pad_and_crop": [data_augmentation.RandomPad(450),
                     data_augmentation.RandomCrop(1024, probability=1.0)]
                       }


def save_wake_and_masks(dataset, image_out_dir, mask_out_dir):
    # iterates through the dataset and saves the transformed images and masks
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    for idx in tqdm.tqdm(range(len(dataset)), desc="images", position=2, leave=False):
        ds = dataset[idx]
        image = torchvision.transforms.ToPILImage()(ds["image"])
        mask = torchvision.transforms.ToPILImage()(ds["mask"])
        file_name = (f"0000{idx}")[-4:] + ".png"
        image.save(os.path.join(image_out_dir, file_name))
        mask.save(os.path.join(mask_out_dir, file_name))


def create_augmented_dataset(data_dir, mask_dir, transform_output_dir, transform):
    # given a transformation, iterates through each band of the wake imagery by
    # creating the band dataset and passing to the function to save the images
    check_output_dir(transform_output_dir)
    source = os.path.join(data_dir, "results.csv")
    destination = os.path.join(transform_output_dir, "results.csv")
    shutil.copy(source, destination)
    bands = ["C", "S", "X"]
    for band in tqdm.tqdm(bands, desc="bands", position=1, leave=False):
        # print(f"  starting band {band}...", end=" ")
        # create the image and mask dataset for the given transform and current band
        ds = data_augmentation.SARDataset(data_dir, [band], mask_dir, transform)
        # ensure the band and masks output directories exist
        output_image_dir = os.path.join(transform_output_dir, f"{band}band")
        check_output_dir(output_image_dir)
        output_masks_dir = os.path.join(transform_output_dir, "masks")
        check_output_dir(output_masks_dir)
        # save the images and masks
        save_wake_and_masks(ds, output_image_dir, output_masks_dir)
        # print("completed")

        # copy results.csv
        source = os.path.join(data_dir, f"{band}band", "results.csv")
        destination = os.path.join(transform_output_dir, f"{band}band", "results.csv")
        shutil.copy(source, destination)

    
def looping_augmented_dataset(data_dir, mask_dir, output_dir, transforms_dict):
    # data_dir and mask_dir are locations of the current images to be augmented
    # output_dir is the root directory for the augmented data to go
    # each transform will have its own sub-directory, then each band and the
    # masks will have their own sub-sub-directory
    check_output_dir(output_dir)
    for name, transform_list in tqdm.tqdm(transforms_dict.items(), desc="transforms", position=0):
        # print(f"starting augmentation {name}")
        # create the pytorch transform process from the list of transforms
        transform = torchvision.transforms.Compose(transform_list)
        # ensure the transform output directory exist
        transform_output_dir = os.path.join(output_dir, name)
        check_output_dir(transform_output_dir)
        # iterate through all bands and images for this transform
        create_augmented_dataset(data_dir, mask_dir, transform_output_dir, transform)
        # print("  completed this augmentation")
    # print("Done!")


def check_output_dir(output_dir):
    # ensure the output directory exists before trying to write to it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


# def image_compare(transform_output_dir, n=5):
#     # given the transform_output_dir, open and plot sample images across each band
#     # for a chose image ID, all band and mask images are plotted in a row to ensure
#     # the tranformation was applied consistantly
#     all_files = glob.glob(f"{transform_output_dir}/masks/*.png")
#     all_files = [os.path.basename(x) for x in all_files]
#     random.seed(42)
#     files = random.sample(all_files, n)
#     print(files)
    
#     image_dirs = ["C", "S", "X", "masks"]
#     fig, axs = plt.subplots(nrows=n, ncols=4, figsize=(12, 3*n))
#     for i, file_name in enumerate(files):
#         for j, img_dir in enumerate(image_dirs):
#             if i == 0:
#                 axs[i, j].set_title(img_dir)
#             if img_dir == "masks":
#                 directory = img_dir
#             else:
#                 directory = f"{img_dir}band"
#             img = Image.open(os.path.join(transform_output_dir, directory, file_name))
#             axs[i, j].imshow(img, cmap="gray")
#             axs[i, j].axis("off")
#     # plt.subplots_adjust(wspace=0.1, hspace=-0.8)
#     plt.tight_layout()


def image_compare(transform_output_dir, n=5, save_file=None):
    # given the transform_output_dir, open and plot sample images across each band
    # for a chose image ID, all band and mask images are plotted in a row to ensure
    # the tranformation was applied consistantly
    all_files = glob.glob(f"{transform_output_dir}/masks/*.png")
    if len(all_files) == 0:
        # no appropriate file in this dir
        print(f"no relevant images in {transform_output_dir}")
        return
    elif len(all_files) < n:
        # sample all files in the directory if it contains less than desired
        n = len(all_files)

    all_files = [os.path.basename(x) for x in all_files]
    random.seed(42)
    files = random.sample(all_files, n)
    
    image_dirs = ["C", "S", "X", "masks"]
    fig, axs = plt.subplots(nrows=n, ncols=5, figsize=(15, 3*n))
    for i, file_name in enumerate(files):
        for j, img_dir in enumerate(image_dirs):
            if i == 0:
                axs[i, j].set_title(img_dir)
            if img_dir == "masks":
                directory = img_dir
            else:
                directory = f"{img_dir}band"
            img = Image.open(os.path.join(transform_output_dir, directory, file_name))
            axs[i, j].imshow(img, cmap="gray")
            axs[i, -1].imshow(img, alpha=0.25, cmap="gray")
            axs[i, j].axis("off")
        axs[i, -1].axis("off")
    axs[0, -1].set_title("Combined")
    # plt.subplots_adjust(wspace=0.1, hspace=-0.8)
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

        
def check_output_dir(output_dir):
    # ensure the output directory exists before trying to write to it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def all_augmentations_image_compare(augmentations_dir, save_image_dir):
    check_output_dir(save_image_dir)
    dir_path, transform_dirs, _ = next(os.walk(augmentations_dir))
    for transform_dir in tqdm.tqdm(transform_dirs, desc="transforms", position=0):
        transform_output_dir = os.path.join(dir_path, transform_dir)
        save_file = os.path.join(save_image_dir, f"{transform_dir}.png")
        # print(f"comparing results for {transform_output_dir}...", end=' ')
        image_compare(transform_output_dir, n=24, save_file=save_file)
        # print("completed!")
        # print(f"saving image to {save_file}")


if __name__ == "__main__":
    TRANSFORM_DICTIONARY = {
        "rotation": [data_augmentation.RandomRotation((0, 359), 1, probability=1.0)],
        "rotation_0": [data_augmentation.RandomRotation((0, 0), 1, probability=1.0)],
        "rotation_15": [data_augmentation.RandomRotation((15, 15), 1, probability=1.0)],
        "rotation_30": [data_augmentation.RandomRotation((30, 30), 1, probability=1.0)],
        "rotation_45": [data_augmentation.RandomRotation((45, 45), 1, probability=1.0)],
        "rotation_60": [data_augmentation.RandomRotation((60, 60), 1, probability=1.0)],
        "rotation_75": [data_augmentation.RandomRotation((75, 75), 1, probability=1.0)],
        "rotation_90": [data_augmentation.RandomRotation((90, 90), 1, probability=1.0)],
        "rotation_105": [data_augmentation.RandomRotation((105, 105), 1, probability=1.0)],
        "rotation_120": [data_augmentation.RandomRotation((120, 120), 1, probability=1.0)],
        "rotation_135": [data_augmentation.RandomRotation((135, 135), 1, probability=1.0)],
        "rotation_150": [data_augmentation.RandomRotation((150, 150), 1, probability=1.0)],
        "rotation_165": [data_augmentation.RandomRotation((165, 165), 1, probability=1.0)],
        "rotation_180": [data_augmentation.RandomRotation((180, 180), 1, probability=1.0)],
                       }

    data_dir = "/home/sdan8/wake_data_augmentation/data/circular_crop_dataset"
    mask_dir = "/home/sdan8/wake_data_augmentation/data/circular_crop_dataset/masks"
    augmentations_dir = "/home/sdan8/wake_data_augmentation/data/augmented_dataset"
    looping_augmented_dataset(data_dir, mask_dir, augmentations_dir, TRANSFORM_DICTIONARY)

    save_image_dir = "/home/sdan8/wake_data_augmentation/data/augmented_dataset/image_compare"
    all_augmentations_image_compare(augmentations_dir, save_image_dir)

