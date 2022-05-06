#!/bin/sh

mkdir data/combined_dataset

# C-Band
python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_75\
 --new_dir data/combined_dataset/rotation_C_2D

python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_15\
 --new_dir data/combined_dataset/rotation_C_5D

python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_15\
 --new_dir data/combined_dataset/rotation_C_10D

python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_15\
 --new_dir data/combined_dataset/rotation_C_all


# S-Band
python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_75\
 --new_dir data/combined_dataset/rotation_S_2D_1

python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_180\
 --new_dir data/combined_dataset/rotation_S_2D_2


# X-Band
python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_75\
 --new_dir data/combined_dataset/rotation_X_2D_1

python src/combining_sets_on_augmentation.py\
 --fold0 data/augmented_dataset/rotation_0\
 --fold1 data/augmented_dataset/rotation_180\
 --new_dir data/combined_dataset/rotation_X_2D_2
