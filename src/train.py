#!/usr/bin/env python3

import torch
import baseline_models as models
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import scipy
import matplotlib.pyplot as plt
import dataloader
import torch.nn as nn
import glob
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from PIL import Image
import diceloss as DL
from tqdm import tqdm
from optparse import OptionParser

def train(n_epochs=4, p_train=8.0/9.0, p_val=1.0/9.0, C=False, S=False, X=False, unet_path=None, classifier_path=None, progress=False):
	"""
	Main training loop.
	n_epochs = number of training epochs
	"""
	use_cuda = torch.cuda.is_available()
	n_batch = 1

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
	
	masks = loadMasks(glob.glob("./current_data/masks/*.png"))
	dataset = dataloader.SARDataset("./current_data/", bands, masks)

	# prepare datasets
	n_train = int(len(dataset) * p_train)
	n_val = int(len(dataset) * p_val)
	n_train += int(len(dataset) - n_train - n_val)
	print("n_train:",n_train,"n_val:",n_val)
	train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
	trainingdata = torch.utils.data.DataLoader(train, batch_size=n_batch, shuffle=True, num_workers=4)
	validationdata = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)
	
	# initial NN, loss function, and optimizer
	Unet = models.Unet(band_list = bands, in_channels=n_channels, out_channels=1)
	#print(Unet)
	classifier = models.ClassifierSmall(band_list = bands, in_channels=n_channels, out_channels=1)
	#classifier = models.ClassifierLarge(band_list = bands, in_channels=n_channels, out_channels=1)
	#print(classifier)
	#if use_cuda and torch.cuda.device_count() > 1:
	#	Unet = nn.DataParallel(Unet)
	#	classifier = nn.DataParallel(classifier)
	#print("Unet params:",get_trainable_params(Unet))
	#print("Classifier params:",get_trainable_params(classifier))
	if use_cuda:
		Unet.cuda()
		classifier.cuda()
	
	# loss function, gradient descent, and learn rate scheduler
	loss = DL.DiceLoss(use_cuda=use_cuda, bce_weight=0.0, smooth=1.0)
	pos_weight = torch.Tensor([1.0/2.0])
	if use_cuda:
		pos_weight = pos_weight.cuda()
	classifierLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	learner = torch.optim.Adam(Unet.parameters(), lr=1e-3, weight_decay=0.1)
	classifierLearner = torch.optim.SGD(classifier.parameters(), lr=1e-2, weight_decay=0.1)
	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifierLearner, 'max', patience=2)

	# extra: progress bar to display during training
	pbar = tqdm(total=n_epochs*n_train + n_epochs*n_val)

	for e in range(n_epochs):
		pbar.set_description("Epoch %s" % str(e + 1))
		loss_array.append(0)
		if e > -1:
			class_loss_array.append(0)
		# training loop
		for i, data in enumerate(trainingdata):
			# prepare network for training
			#Unet.zero_grad()
			#classifier.zero_grad()
			learner.zero_grad()
			classifierLearner.zero_grad()
			Unet.train()
			classifier.train()

			# load current training image and randomly rotate it
			testimage = data["image"]
			testimage = Variable(testimage)

			if use_cuda:
				testimage = testimage.cuda()

			classifierTruth = torch.Tensor([float(float(data["contains_wake"][0]) > 0.1)]) # single value no matter how many bands are trained

			# generate wake mask 
			maskraw = Unet(testimage)

			# load ground truth data and rotate to the same angle as the training data
			masktrue = data["mask"]
			#figure = plt.figure()
			#plt.imshow(maskToImage(testimage, use_cuda), cmap='gray')
			#plt.axis("off")
			#plt.clim([0,1])
			#plt.savefig("testmask.png")
			#exit(0)
			if use_cuda:
				masktrue = masktrue.cuda()
				classifierTruth = classifierTruth.cuda()

			# calculate loss of wake mask and backpropagate
			lossvalue = loss(maskraw, masktrue)
			loss_array[-1] += lossvalue.item()
			lossvalue.backward()
			learner.step()

			# train classifier
			if e > -1:
				classOutput = classifier(testimage, torch.sigmoid(maskraw))
				classLoss = classifierLoss(classOutput, classifierTruth)
				class_loss_array[-1] += classLoss.item()
				classLoss.backward()
				classifierLearner.step()

			# update progress bar
			pbar.update()
		
		loss_array[-1] /= (i + 1) # determine average epoch training loss
		if e > -1:
			class_loss_array[-1] /= (i + 1) # determine average epoch training loss


		val_loss_array.append(0)
		if e > -1:
			class_val_loss_array.append(0)
		# validation data
		for j, valdata in enumerate(validationdata):
			# prepare network for evaluation
			Unet.eval()
			classifier.eval()

			# load validation data and ground truth
			val_test = valdata["image"]
			val_class = torch.Tensor([float(float(valdata["contains_wake"][0]) > 0.1)])
			masktrue = valdata["mask"]
			if use_cuda:
				val_test = val_test.cuda()
				val_class = val_class.cuda()
				masktrue = masktrue.cuda()

			# forward propagate validation data
			with torch.no_grad():
				valmask = Unet(val_test)
				valoutputclass = classifier(val_test, valmask)

			if e > -1:
				#valoutputclass = classifier(val_test, valmask)
				valclassloss = classifierLoss(valoutputclass, val_class)
				class_val_loss_array[-1] += valclassloss.item()

			# calculate loss
			valloss = loss(valmask, masktrue)
			val_loss_array[-1] += valloss.item()

			pbar.update()
		val_loss_array[-1] /= (j + 1)
		print(f"Epoch {e} Unet loss: {val_loss_array[-1]}")
		if e > -1:
			class_val_loss_array[-1] /= (j + 1)
		
		# adjust learn rate based on validation loss
		#scheduler.step(class_val_loss_array[-1])

		if progress:
			figure = plt.figure()
			valmask = torch.sigmoid(valmask)
			plt.imshow(maskToImage(valmask, use_cuda), cmap='gray')
			plt.axis("off")
			plt.clim([0, 1])
			plt.savefig("progress_e" + str(e+1) + ".png")
			plt.close()
		

	pbar.close()

	# view a validation data case
#	print("Wake judgement:",torch.sigmoid(valoutputclass).item())
#	valmask = valmask.cpu().detach()[0,:,:,:]
#	valmask = valmask.unsqueeze(0)	
#	mask = torch.sigmoid(valmask)
#
#	## plot
#	figure = plt.figure()
#	ax = plt.subplot(221)
#	plt.imshow(maskToImage(val_test, use_cuda), cmap='gray')
#	plt.axis("off")
#	plt.clim([0,1])
#	ax = plt.subplot(222)
#	plt.imshow(maskToImage(masktrue, use_cuda), cmap='gray')
#	plt.axis("off")
#	plt.clim([0,1])
#	ax = plt.subplot(223)
#	plt.imshow(maskToImage(mask, use_cuda), cmap='gray')
#	plt.axis("off")
#	plt.clim([0,1])
#	ax = plt.subplot(224)
#	plt.imshow(maskToImage(mask, use_cuda) > 0.60, cmap='gray')
#	plt.axis("off")
#	plt.clim([0,1])
#	plt.savefig("./test.png")

	# save networks
	#torch.save(Unet, "./trainedModels/Unet.pth")
	#torch.save(classifier, "./trainedModels/classifier.pth")
	#torch.save(Unet, unet_path)
	#torch.save(classifier, classifier_path)
	torch.save(Unet.state_dict(), unet_path)
	torch.save(classifier.state_dict(), classifier_path)

	# plot training and validation loss
	figure = plt.figure()
	xpoints = np.linspace(0,n_epochs,n_epochs)
	plt.plot(xpoints, loss_array, label="Training Loss")
	plt.plot(xpoints, val_loss_array, label="Validation Loss")
	plt.legend()
	plt.savefig("./loss.png")
	plt.close()
	
	figure = plt.figure()
	xpoints = np.linspace(0,n_epochs,n_epochs)
	plt.plot(xpoints, class_loss_array, label="Training Loss")
	plt.plot(xpoints, class_val_loss_array, label="Validation Loss")
	plt.legend()
	plt.savefig("./class_loss.png")
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

def loadMasks(paths):
	"""
	Given a list of mask paths, return a dictionary of mask images referenced by their name as determined by their filepath
	The mask name should match the run_name given in the results.csv file
	"""
	process = transforms.Compose([
		transforms.Grayscale(),
		transforms.ToTensor()
	])
	masks = {}
	for p in paths:	
		maskImage = Image.open(p)
		fileName = p.split("/")[-1] # if path p is "./current_data/masks/testMask.png", this line saves "testMask.png" as fileName
		maskName = (fileName.split(".png")[0])[:-4] # if fileName = "testMask.png", this line saves "test" as the maskName
		masks[maskName] = process(maskImage)
	return masks

if __name__ == "__main__":
	optParser = OptionParser("%prog [options]", version="20201110")
	optParser.add_option("--Cband", action="store_true", dest="use_C", help="train using C-band data", default=False)
	optParser.add_option("--Sband", action="store_true", dest="use_S", help="train using S-band data", default=False)
	optParser.add_option("--Xband", action="store_true", dest="use_X", help="train using X-band data", default=False)
	optParser.add_option("-u", "--unet", action="store", dest="unet_path", help="U-net model output file path", default="./trainedModels/Unet.pth")
	optParser.add_option("-c", "--classifier", action="store", dest="classifier_path", help="U-net model output file path", default="./trainedModels/classifier.pth")
	optParser.add_option("-n", "--n_epochs", action="store", type=int, dest="n_epochs", help="number of training epochs", default=int(2))
	optParser.add_option("--progress", action="store_true", dest="progress", help="output training progress images", default=False)
	options, args = optParser.parse_args()
	if not options.use_C and not options.use_S and not options.use_X:
		raise RuntimeError("At least one SAR band needs to be specified for training")
	print("N_epochs",options.n_epochs)
	train(C=options.use_C, S=options.use_S, X=options.use_X, n_epochs=options.n_epochs, unet_path=options.unet_path, classifier_path=options.classifier_path, progress=options.progress)
