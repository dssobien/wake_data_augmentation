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
import data_augmentation
import csv

def train(n_epochs=4, p_train=8.0/9.0, p_val=1.0/9.0, C=False, S=False, X=False, unet_path=None, classifier_path=None, progress=False,
		n_batch=1, prog_bar=False, model='other', glorot=False, learning_rate=1e-3, n_iter=None, bce_weight=None,
		weight_decay=None, transforms=None):
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

	#if bands == ['S'] or bands == ['X']:
	#	learning_rate = 1e-4

	mask_dir = "./current_data/masks"
	dataset = data_augmentation.SARDataset("./current_data/", bands, mask_dir, transforms)

	# prepare datasets
	n_train = int(len(dataset) * p_train)
	n_val = int(len(dataset) * p_val)
	n_train += int(len(dataset) - n_train - n_val)
	print(model)
	print(f"Epochs = {n_epochs}")
	print(f"Batch size = {n_batch}")
	print("n_train:",n_train,"n_val:",n_val)
	train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
	trainingdata = torch.utils.data.DataLoader(train, batch_size=n_batch, shuffle=True, num_workers=4)
	validationdata = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)
	
	# initial NN, loss function, and optimizer
	if model == 'original':
		Unet = models.Unet(band_list = bands, in_channels=n_channels, out_channels=1)
	elif model == 'other':
		Unet = pytorch_unet.UNet(band_list = bands, in_channels=n_channels, out_channels=1)

	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
			nn.init.zeros_(m.bias)
	if glorot is True:
		Unet.apply(weight_init)

	#if use_cuda and torch.cuda.device_count() > 1:
	#	Unet = nn.DataParallel(Unet)
	#print("Unet params:",get_trainable_params(Unet))
	if use_cuda:
		Unet.cuda()
		print('using GPU')
	else:
		print('using CPU')
	
	# loss function, gradient descent, and learn rate scheduler
	if bce_weight is None:
		loss = DL.DiceLoss(use_cuda=use_cuda, bce_weight=0.0, smooth=1.0)
	else:
		loss = DL.DiceLoss(use_cuda=use_cuda, bce_weight=bce_weight, smooth=1.0)
	pos_weight = torch.Tensor([1.0/2.0])
	if use_cuda:
		pos_weight = pos_weight.cuda()
	if learning_rate is None:
		learning_rate = 1e-3
	# learner = torch.optim.Adam(Unet.parameters(), lr=1e-3, weight_decay=0.1)
	if weight_decay is None:
		learner = torch.optim.Adam(Unet.parameters(), lr=learning_rate, weight_decay=0.1)
	else:
		learner = torch.optim.Adam(Unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

	name = f"unet_{''.join(bands)}"
	if glorot is True:
		name += f"_glorot"
	name += f"_{n_epochs}e_{n_batch}b_lr{learning_rate}"
	if n_iter is not None:
		name += f"_{n_iter}i"
	if bce_weight is not None:
		name += f"_{bce_weight}bce"
	if weight_decay is not None:
		name += f"_{weight_decay}wd"
	print(name)
	if unet_path is None:
		unet_path = f"./trainedModels/Unet_{''.join(bands)}.pth"

	best_loss = 1e10

	# extra: progress bar to display during training
	if prog_bar is True:
		pbar = tqdm(total=n_epochs*n_train/n_batch + n_epochs*n_val)
	for e in range(n_epochs):
		if prog_bar is True:
			pbar.set_description("Epoch %s" % str(e + 1))
		loss_array.append(0)
		# training loop
		for i, data in enumerate(trainingdata):
			# prepare network for training
			#Unet.zero_grad()
			learner.zero_grad()
			Unet.train()

			# load current training image and randomly rotate it
			testimage = data["image"]
			testimage = Variable(testimage)

			if use_cuda:
				testimage = testimage.cuda()


			# generate wake mask 
			maskraw = Unet(testimage)

			# load ground truth data and rotate to the same angle as the training data
			masktrue = data["mask"]
			if use_cuda:
				masktrue = masktrue.cuda()

			# calculate loss of wake mask and backpropagate
			lossvalue = loss(maskraw, masktrue)
			loss_array[-1] += lossvalue.item()
			lossvalue.backward()
			learner.step()

			# update progress bar
			if prog_bar is True:
				pbar.update()
		
		loss_array[-1] /= (i + 1) # determine average epoch training loss


		val_loss_array.append(0)
		# validation data
		for j, valdata in enumerate(validationdata):
			# prepare network for evaluation
			Unet.eval()

			# load validation data and ground truth
			val_test = valdata["image"]
			masktrue = valdata["mask"]
			if use_cuda:
				val_test = val_test.cuda()
				masktrue = masktrue.cuda()

			# forward propagate validation data
			with torch.no_grad():
				valmask = Unet(val_test)

			# calculate loss
			valloss = loss(valmask, masktrue)
			val_loss_array[-1] += valloss.item()

			if prog_bar is True:
				pbar.update()
		val_loss_array[-1] /= (j + 1)
		print(f"Epoch {e+1} Unet loss: {val_loss_array[-1]}")

		if val_loss_array[-1] < best_loss:
			print('saving best model')
			best_loss = val_loss_array[-1]
			best_model_wts = copy.deepcopy(Unet.state_dict())		

		# adjust learn rate based on validation loss

		if progress and ((e+1) % 20 == 0):
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
	#torch.save(Unet, "./trainedModels/Unet.pth")
	#torch.save(Unet, unet_path)
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

def get_trainable_params(model):
	"""
	Return the number of trainable parameters in a network
	"""
	params = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in params])

def sampleTest(no_wake, wake, p_train_val, p_test):
	random.shuffle(no_wake)
	random.shuffle(wake)
	n_train_val = round((len(no_wake) + len(wake)) * p_train_val / 2.0)
	n_test = round((len(no_wake) + len(wake)) * p_test / 2.0)

	return no_wake[:n_test] + wake[:n_test], no_wake[n_test:n_train_val + n_test] + wake[n_test:n_train_val + n_test]

def prepareResultsCsv(train_val):
	newFile = "./current_data/Cband/results.csv"
	dataFile = "./current_data/results.csv"
	with open(newFile, "w") as newResults:
		fieldNames = ["uuid", "run_name", "band", "look_angle", "polarization", "inclination_angle", "contains_wake"]
		writer = csv.DictWriter(newResults, fieldnames=fieldNames)
		writer.writeheader()

		with open(dataFile, "r") as data:
			reader = csv.DictReader(data)
			for row in reader:
				if row["uuid"] in train_val:
					writer.writerow(row)	

if __name__ == "__main__":
	optParser = OptionParser("%prog [options]", version="20201110")
	optParser.add_option("--Cband", action="store_true", dest="use_C", help="train using C-band data", default=False)
	optParser.add_option("--Sband", action="store_true", dest="use_S", help="train using S-band data", default=False)
	optParser.add_option("--Xband", action="store_true", dest="use_X", help="train using X-band data", default=False)
	optParser.add_option("-u", "--unet", action="store", dest="unet_path", help="U-net model output file path", default=None)
	optParser.add_option("-c", "--classifier", action="store", dest="classifier_path", help="U-net model output file path", default=None)
	optParser.add_option("-n", "--n_epochs", action="store", type=int, dest="n_epochs", help="number of training epochs", default=int(2))
	optParser.add_option("-b", "--n_batch", action="store", type=int, dest="n_batch", help="training batch size", default=int(1))
	optParser.add_option("--progress", action="store_true", dest="progress", help="output training progress images", default=False)
	optParser.add_option("--prog_bar", action="store_true", dest="prog_bar", help="include progress bar while training", default=False)
	optParser.add_option("--model", action="store", dest="model", help="U-net model name", default="other")
	optParser.add_option("--glorot", action="store_true", dest="glorot", help="use Glorot normal initializer for convolutional layer weights", default=False)
	optParser.add_option("--learning_rate", action="store", type=float, dest="learning_rate", help="training batch size", default=0.001)
	optParser.add_option("--bce_weight", action="store", type=float, dest="bce_weight", help="ratio of BCE to Dice loss", default=0.0)
	options, args = optParser.parse_args()
	if not options.use_C and not options.use_S and not options.use_X:
		raise RuntimeError("At least one SAR band needs to be specified for training")
	# print("N_epochs",options.n_epochs)
	train(C=options.use_C, S=options.use_S, X=options.use_X, n_epochs=options.n_epochs, unet_path=options.unet_path, classifier_path=options.classifier_path, progress=options.progress,
		n_batch=options.n_batch, prog_bar=options.prog_bar, model=options.model, glorot=options.glorot, learning_rate=options.learning_rate, bce_weight=options.bce_weight)

