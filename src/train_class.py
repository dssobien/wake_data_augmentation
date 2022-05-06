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
from tqdm import tqdm
from optparse import OptionParser
from find_gpu import find_gpu
import glob
import data_augmentation

def train(n_epochs=4, p_train=8.0/9.0, p_val=1.0/9.0, C=False, S=False, X=False, unet_path=None, classifier_path=None, progress=False, use_large_classifier=True,
		n_batch=1, prog_bar=False, model='other', transforms=None,
		weight_decay=0.1, learning_rate=1e-3):
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
	while n_train % n_batch != 0:
		n_train -= 1
	n_val = int(len(dataset) - n_train)
	while n_val % n_batch != 0:
		n_val -= 1
	# n_val = int(len(dataset) * p_val)
	# n_train += int(len(dataset) - n_train - n_val)
	print(model)
	print(f"Epochs = {n_epochs}")
	print(f"Batch size = {n_batch}")
	print("n_train:",n_train,"n_val:",n_val)
	train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
	trainingdata = torch.utils.data.DataLoader(train, batch_size=n_batch, shuffle=True, num_workers=4)
	# validationdata = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)
	validationdata = torch.utils.data.DataLoader(val, batch_size=n_batch, shuffle=True, num_workers=4)

	if model == 'original':
		Unet = models.Unet(band_list = bands, in_channels=n_channels, out_channels=1)
	elif model == 'other':
		Unet = pytorch_unet.UNet(band_list = bands, in_channels=n_channels, out_channels=1)
	if unet_path is not None:
		print(f"Loading model: {unet_path}")
		Unet.load_state_dict(torch.load(unet_path))
		Unet.eval()
		if use_cuda:
			Unet.cuda()
	
	if classifier_path is None:
		classifier_path = f"./trainedModels/classifier_{''.join(bands)}.pth"
	
	# initial NN, loss function, and optimizer
	if use_large_classifier:
		classifier = models.ClassifierUnet(band_list=bands, in_channels=n_channels, out_channels=1, n_batch=n_batch)
	else:
		classifier = models.ClassifierSmall(band_list=bands, in_channels=n_channels, out_channels=1, n_batch=n_batch)
	print("Classifier params:",get_trainable_params(classifier))
	if use_cuda:
		classifier.cuda()
	
	# loss function, gradient descent, and learn rate scheduler
	pos_weight = torch.Tensor([1.0])
	if use_cuda:
		pos_weight = pos_weight.cuda()
	classifierLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	# classifierLearner = torch.optim.SGD(classifier.parameters(), lr=1e-3, weight_decay=0.1)
	classifierLearner = torch.optim.SGD(classifier.parameters(), lr=learning_rate,
						weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifierLearner, patience=1)
	
	if unet_path is not None:
		name = f"class_{''.join(bands)}_unet_{n_epochs}e_{n_batch}b"
	else:
		name = f"class_{''.join(bands)}_{n_epochs}e_{n_batch}b"
	name += f"_{learning_rate}lr"
	name += f"_{weight_decay}wd"
	print(name)

	best_loss = 1e10

	# extra: progress bar to display during training
	if prog_bar is True:
		pbar = tqdm(total=n_epochs*n_train/n_batch + n_epochs*n_val)
	# initialize value for best model weights
	best_model_wts = copy.deepcopy(classifier.state_dict())		
	for e in range(n_epochs):
		if prog_bar is True:
			pbar.set_description("Epoch %s" % str(e + 1))
		loss_array.append(0)
		if e > -1:
			class_loss_array.append(0)
		# training loop
		for i, data in enumerate(trainingdata):
			# prepare network for training
			Unet.zero_grad()
			classifierLearner.zero_grad()
			classifier.train()

			# load current training image and randomly rotate it
			testimage = data["image"]
			testimage = Variable(testimage)

			if use_cuda:
				testimage = testimage.cuda()

			classifierTruth = torch.Tensor([float(float(data["contains_wake"][0]) > 0.1)]) # single value no matter how many bands are trained

			if use_cuda:
				classifierTruth = classifierTruth.cuda()

			# train classifier
			if unet_path is not None:
				with torch.no_grad():
					inputMask = Unet(testimage)
				classOutput = classifier(testimage, inputMask)
			else:
				classOutput = classifier(testimage, None)
			classLoss = classifierLoss(classOutput, classifierTruth)
			class_loss_array[-1] += classLoss.item()
			classLoss.backward()
			classifierLearner.step()

			# update progress bar
			if prog_bar is True:
				pbar.update()
		
		class_loss_array[-1] /= (i + 1) # determine average epoch training loss


		class_val_loss_array.append(0)
		# validation data
		for j, valdata in enumerate(validationdata):
			# prepare network for evaluation
			classifier.eval()

			# load validation data and ground truth
			val_test = valdata["image"]
			val_class = torch.Tensor([float(float(valdata["contains_wake"][0]) > 0.1)])
			if use_cuda:
				val_test = val_test.cuda()
				val_class = val_class.cuda()

			# forward propagate validation data
			with torch.no_grad():
				if unet_path is not None:
					inputMask = Unet(val_test)
					valoutputclass = classifier(val_test, inputMask)
				else:
					valoutputclass = classifier(val_test, None)

			valclassloss = classifierLoss(valoutputclass, val_class)
			class_val_loss_array[-1] += valclassloss.item()

			if prog_bar is True:
				pbar.update()
		class_val_loss_array[-1] /= (j + 1)
		scheduler.step(class_val_loss_array[-1])
		print(f"Epoch {e+1} val loss: {class_val_loss_array[-1]}")
		
		if class_val_loss_array[-1] < best_loss:
			print('saving best model')
			best_loss = class_val_loss_array[-1]
			best_model_wts = copy.deepcopy(classifier.state_dict())		


		if progress and ((e+1) % 5 == 0):
			figure = plt.figure()

			ax = figure.add_subplot(1, 3, 1)
			plt.imshow(maskToImage(val_test, use_cuda), cmap='gray')
			ax.set_title('Input Image')
			plt.axis("off")

			ax = figure.add_subplot(1, 3, 2)
			plt.imshow(maskToImage(valdata["mask"], use_cuda), cmap='gray')
			ax.set_title('Ground Truth Mask')
			plt.axis("off")

			if unet_path is not None:
				ax = figure.add_subplot(1, 3, 3)
				plt.imshow(maskToImage(inputMask, use_cuda), cmap='gray')
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

	# save network
	torch.save(best_model_wts, classifier_path)

	figure = plt.figure()
	xpoints = np.linspace(0,n_epochs,n_epochs)
	plt.plot(xpoints, class_loss_array, label="Training Loss")
	plt.plot(xpoints, class_val_loss_array, label="Validation Loss")
	plt.legend()
	plt.savefig(f"output/class_loss_{name}.png")
	plt.close()

def maskToImage(mask, use_cuda):
	"""
	Taking a wake mask in the form of a PyTorch tensor, return a numpy array
	"""
	if use_cuda:
		return mask.cpu().detach().numpy()[0,0,:,:]
	return mask.detach().numpy()[0,0,:,:]

def unetMasks(image_tensor, unet):
	final_images = []
	for i in range(image_tensor.size(0)):
		image = image_tensor[i].unsqueeze(0)
		print(image.shape)
		#image = image.squeeze(0)
		mask = unet(image)
		#mask = mask.unsqueeze(0)
		final_images.append(mask)
	image_tensor_out = torch.cat(final_images, 0)
	return image_tensor_out

def get_trainable_params(model):
	"""
	Return the number of trainable parameters in a network
	"""
	params = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in params])

if __name__ == "__main__":
	optParser = OptionParser("%prog [options]", version="20201116")
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
	options, args = optParser.parse_args()
	if not options.use_C and not options.use_S and not options.use_X:
		raise RuntimeError("At least one SAR band needs to be specified for training")
	print("N_epochs",options.n_epochs)
	train(C=options.use_C, S=options.use_S, X=options.use_X, n_epochs=options.n_epochs, unet_path=options.unet_path, classifier_path=options.classifier_path, progress=options.progress,
	n_batch=options.n_batch, prog_bar=options.prog_bar, model=options.model)
