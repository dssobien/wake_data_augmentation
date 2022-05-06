#!/bin/bash

# create the circular cropped images used in this study
python src/circular_crop_dataset.py

# create the augmented datasets
python src/augmented_datasets.py

# create the umap latent space representation
python src/generate_augmented_sets_umap.py

# run script to combine augmentation sets
bash run_combine_on_augs.sh

# make directory for the trained models
mkdir trainedModels

# run all the models
bash run_models.sh
