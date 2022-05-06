# Improving Deep Learning Through Data Augmentation and Latent Space

## Requirements
See the [environment.yml](environment.yml) file for the full list of required packages. To recreate this environment using conda, run the command `conda env create -f environment.yml` in the cloned repository.

## Network Architecture

The U-net is based on the original architecture by [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) and implemented by [Naoto Usuyama](https://github.com/usuyama/pytorch-unet).

The classifier consists of a 2D convolution, 2D batch norm, ReLU layer, followed by a fully-connected layer which outputs a single value. The classifier takes the U-net wake mask output as an input in addition to any SAR bands.

## Running the Experiments
The `run_all.sh` script goes through all the steps to recreate the circular crop data set and subsequent augmented datasets, UMAP latent space representations, the combined augmentation training sets using the `run_combine_on_augs.sh` script, and finally runs all the experiments training and testing the deep learning models using the `run_models.sh` script. The entire process will take a long time to run, even on a GPU, so we recommend running individual commands from the `run_all.sh` script. All the training and testing of models is controlled by the `run_models.sh` script and the interface it calls through `src/sweepUNetClass.py`. For more information on running different experiments call `python src/sweepUNetClass.py --help` for more information.

## Viewing the Results
Jupyter notebooks are set up in the `notebooks/` directory to read the results of the experiments and recreate the analysis in the paper. You must run the models first, either from the `run_all.sh` or `run_models.sh` script. Each experiment in the `run_models.sh` script will output a different results directory, and they must be run before the notebooks can be used.
