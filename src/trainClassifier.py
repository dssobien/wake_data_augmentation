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
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from PIL import Image
from tqdm import tqdm
from optparse import OptionParser
import data_augmentation

def train(n_epochs=4, p_train=8.0/9.0, p_val=1.0/9.0, C=False, S=False, X=False, unet_path=None, classifier_path=None, progress=False, use_large_classifier=True, transforms=None):
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
	
	mask_dir = "./current_data/masks"
	dataset = data_augmentation.SARDataset("./current_data/", bands, mask_dir, transforms)

	# prepare datasets
	n_train = int(len(dataset) * p_train)
	n_val = int(len(dataset) * p_val)
	n_train += int(len(dataset) - n_train - n_val)
	print("n_train:",n_train,"n_val:",n_val)
	train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
	trainingdata = torch.utils.data.DataLoader(train, batch_size=n_batch, shuffle=True, num_workers=4)
	validationdata = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True, num_workers=4)
	
	# initial NN, loss function, and optimizer
	if use_large_classifier:
		classifier = models.ClassifierLarge(band_list = bands, in_channels=n_channels, out_channels=1)
	else:
		classifier = models.ClassifierSmall(band_list = bands, in_channels=n_channels, out_channels=1)
	print("Classifier params:",get_trainable_params(classifier))
	if use_cuda:
		classifier.cuda()
	
	# loss function, gradient descent, and learn rate scheduler
	pos_weight = torch.Tensor([1.0])
	if use_cuda:
		pos_weight = pos_weight.cuda()
	classifierLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	classifierLearner = torch.optim.SGD(classifier.parameters(), lr=1e-3, weight_decay=0.1)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifierLearner, patience=1)

	# extra: progress bar to display during training
	#pbar = tqdm(total=n_epochs*n_train + n_epochs*n_val)

	for e in range(n_epochs):
		#pbar.set_description("Epoch %s" % str(e + 1))
		loss_array.append(0)
		if e > -1:
			class_loss_array.append(0)
		# training loop
		for i, data in enumerate(trainingdata):
			# prepare network for training
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
			classOutput = classifier(testimage)
			classLoss = classifierLoss(classOutput, classifierTruth)
			class_loss_array[-1] += classLoss.item()
			classLoss.backward()
			classifierLearner.step()

			# update progress bar
			#pbar.update()
		
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
				valoutputclass = classifier(val_test)

			valclassloss = classifierLoss(valoutputclass, val_class)
			class_val_loss_array[-1] += valclassloss.item()

			#pbar.update()
		class_val_loss_array[-1] /= (j + 1)
		#scheduler.step(class_val_loss_array[-1])
		

	#pbar.close()

	# save network
	torch.save(classifier.state_dict(), classifier_path)

	figure = plt.figure()
	xpoints = np.linspace(0,n_epochs,n_epochs)
	plt.plot(xpoints, class_loss_array, label="Training Loss")
	plt.plot(xpoints, class_val_loss_array, label="Validation Loss")
	plt.legend()
	plt.savefig("./class_loss.png")
	plt.close()

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
	optParser.add_option("-u", "--unet", action="store", dest="unet_path", help="U-net model output file path", default="./trainedModels/Unet.pth")
	optParser.add_option("-c", "--classifier", action="store", dest="classifier_path", help="U-net model output file path", default="./trainedModels/classifier.pth")
	optParser.add_option("-n", "--n_epochs", action="store", type=int, dest="n_epochs", help="number of training epochs", default=int(2))
	optParser.add_option("--progress", action="store_true", dest="progress", help="output training progress images", default=False)
	options, args = optParser.parse_args()
	if not options.use_C and not options.use_S and not options.use_X:
		raise RuntimeError("At least one SAR band needs to be specified for training")
	print("N_epochs",options.n_epochs)
	train(C=options.use_C, S=options.use_S, X=options.use_X, n_epochs=options.n_epochs, unet_path=options.unet_path, classifier_path=options.classifier_path, progress=options.progress)
