#!/bin/bash

# baseline latent space study
python src/sweepUNetClass.py --train_dir data/augmented_dataset\
 --train_data_dirs rotation_0 --test_dir data/augmented_dataset\
 --test_data_dirs rotation_0 --train_bands C\
 --test_bands C S X\
 --n_epochs 60 --unet_epochs 60 --n_iter 5

mv outputs_rotation_0 outputs_baseline_latent_rotation_0

python src/sweepUNetClass.py --train_dir data/augmented_dataset\
 --train_data_dirs rotation --test_dir data/augmented_dataset\
 --test_data_dirs rotation --train_bands C\
 --test_bands C S X\
 --n_epochs 60 --unet_epochs 60 --n_iter 5

mv outputs_rotation outputs_baseline_latent_rotation

# baseline performance study
python src/sweepUNetClass.py --train_dir data/augmented_dataset\
 --train_data_dirs rotation_0 --test_dir data/augmented_dataset\
 --test_data_dirs rotation_0 rotation_90 --train_bands C S X\
 --n_epochs 60 --unet_epochs 60 --n_iter 5

# C-band performance study
python src/sweepUNetClass.py --train_dir data/combined_dataset\
 --train_data_dirs rotation_C_2D rotation_C_5D rotation_C_10D rotation_C_all\
 --test_dir data/augmented_dataset\
 --test_data_dirs rotation_0 rotation_90 --train_bands C\
 --n_epochs 60 --unet_epochs 60 --n_iter 5

# S-band performance study
python src/sweepUNetClass.py --train_dir data/combined_dataset\
 --train_data_dirs rotation_S_2D_1 rotation_S_2D_2\
 --test_dir data/augmented_dataset\
 --test_data_dirs rotation_0 rotation_90 --train_bands S\
 --n_epochs 60 --unet_epochs 60 --n_iter 5

# X-band performance study
python src/sweepUNetClass.py --train_dir data/combined_dataset\
 --train_data_dirs rotation_X_2D_1 rotation_X_2D_2\
 --test_dir data/augmented_dataset\
 --test_data_dirs rotation_0 rotation_90 --train_bands X\
 --n_epochs 60 --unet_epochs 60 --n_iter 5
