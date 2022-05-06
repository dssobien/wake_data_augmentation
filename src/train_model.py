#!/usr/bin/env python3

import torch
import baseline_models as models
import pytorch_unet
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import scipy
import matplotlib.pyplot as plt
import dataloader
import data_augmentation
import torch.nn as nn
import os
import copy
import torchvision
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from PIL import Image
import diceloss as DL
from tqdm import tqdm
from optparse import OptionParser
from find_gpu import find_gpu
import glob


def train(model, n_epochs=4, p_train=8.0/9.0, p_val=1.0/9.0, C=False, S=False, X=False, unet_path=None, classifier_path=None, progress=False,
		n_batch=1, prog_bar=False, glorot=False, n_iter=None, 
		transforms=None):
	"""
	Main training loop.
	n_epochs = number of training epochs
	"""
	os.environ['CUDA_VISIBLE_DEVICES']=str(find_gpu())
	use_cuda = torch.cuda.is_available()

	# loss arrays
	loss_array = []
	class_loss_array = []
	val_loss_array = []
	class_val_loss_array = []

	# data: count the number of bands needed
	bands = []
	if C:
		bands.append("C")
	if S:
		bands.append("S")
	if X:
		bands.append("X")
	n_channels = len(bands)

	mask_dir = "./current_data/masks"
	dataset = data_augmentation.SARDataset("./current_data/", bands, mask_dir, transforms)

	# prepare datasets
	n_train = int(len(dataset) * p_train)
	n_val = int(len(dataset) * p_val)
	n_train += int(len(dataset) - n_train - n_val)
	print(model.name)
	print(f"Epochs = {n_epochs}")
	print(f"Batch size = {n_batch}")
	print("n_train:",n_train,"n_val:",n_val)
	train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
	trainingdata = torch.utils.data.DataLoader(train, batch_size=n_batch, shuffle=True, num_workers=4)
	validationdata = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)

	name = model.name
	
	name += f"_{n_batch}b"
	name += f"_{n_epochs}e"
	if n_iter is not None:
		name += f"_{n_iter}i"
	print(name)

	best_loss = 1e10

	# extra: progress bar to display during training
	if prog_bar is True:
		pbar = tqdm(total=n_epochs*n_train/n_batch + n_epochs*n_val)
	for e in range(n_epochs):
		if prog_bar is True:
			pbar.set_description("Epoch %s" % str(e + 1))
		# epoch loop
		train_loss, val_loss = model.train_one_epoch(trainingdata, validationdata)
		loss_array.append(train_loss)
		val_loss_array.append(val_loss)
		
		print(f"Epoch {e+1} loss: {val_loss_array[-1]}")

		if val_loss_array[-1] < best_loss:
			print('saving best model')
			best_loss = val_loss_array[-1]
			best_model_wts = copy.deepcopy(model.state_dict())		

		# adjust learn rate based on validation loss

		if progress and ((e+1) % 5 == 0):
			figure = plt.figure()
			valmask = torch.sigmoid(valmask)

			ax = figure.add_subplot(1, 3, 1)
			plt.imshow(maskToImage(val_test, use_cuda), cmap='gray')
			ax.set_title('Input Image')
			plt.axis("off")

			ax = figure.add_subplot(1, 3, 2)
			plt.imshow(maskToImage(masktrue, use_cuda), cmap='gray')
			ax.set_title('Ground Truth Mask')
			plt.axis("off")

			ax = figure.add_subplot(1, 3, 3)
			plt.imshow(maskToImage(valmask, use_cuda), cmap='gray')
			ax.set_title('U-Net Output Mask')
			plt.axis("off")
			plt.clim([0, 1])
			#plt.savefig("progress_e" + str(e+1) + ".png")
			try:
				plt.savefig(f"output/progress_{name}/epoch_{e+1}.png")
			except:
				os.mkdir(f"output/progress_{name}")
				plt.savefig(f"output/progress_{name}/epoch_{e+1}.png")
			plt.close()
		

	if prog_bar is True:
		pbar.close()

	# save networks
	torch.save(best_model_wts, unet_path)

	# plot training and validation loss
	figure = plt.figure()
	xpoints = np.linspace(0,n_epochs,n_epochs)
	plt.plot(xpoints, loss_array, label="Training Loss")
	plt.plot(xpoints, val_loss_array, label="Validation Loss")
	plt.legend()
	plt.savefig(f"output/loss_{name}.png")
	plt.close()
	

def maskToImage(mask, use_cuda):
	"""
	Taking a wake mask in the form of a PyTorch tensor, return a numpy array
	"""
	if use_cuda:
		return mask.cpu().detach().numpy()[0,0,:,:]
	return mask.detach().numpy()[0,0,:,:]


