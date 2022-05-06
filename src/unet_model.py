#!/usr/bin/env python3

import torch
import pytorch_unet
import torch.nn as nn
from torch.autograd import Variable
import diceloss as DL


class UnetModel:
	def __init__(self, bands, n_channels, unet_path=None, glorot=False,
			learning_rate=1e-3, bce_weight=None,
			weight_decay=None):

		use_cuda = torch.cuda.is_available()
		self.use_cuda = use_cuda

		# set up the model
		Unet = pytorch_unet.UNet(band_list=bands, in_channels=n_channels, out_channels=1)

		def weight_init(m):
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
				nn.init.zeros_(m.bias)
		if glorot is True:
			Unet.apply(weight_init)

		if use_cuda:
			Unet.cuda()
			print('Unet using GPU')
		else:
			print('Unet using CPU')
	
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
		if weight_decay is None:
			learner = torch.optim.Adam(Unet.parameters(), lr=learning_rate, weight_decay=0.1)
		else:
			learner = torch.optim.Adam(Unet.parameters(), lr=learning_rate, weight_decay=weight_decay)

		if glorot is True:
			name = f"Unet_{''.join(bands)}_glorot_lr{learning_rate}"
		else:
			name = f"Unet_{''.join(bands)}_lr{learning_rate}"
		if bce_weight is not None:
			name += f"_{bce_weight}bce"
		if weight_decay is not None:
			name += f"_{weight_decay}wd"
		if unet_path is None:
			unet_path = f"./trainedModels/Unet_{''.join(bands)}.pth"

		self.Unet = Unet
		self.name = name
		self.learner = learner
		self.loss = loss
		self.path = unet_path

	def state_dict(self):
		return self.Unet.state_dict()

	def save_model(self):
		torch.save(self.Unet.state_dict(), self.path)

	def load_model(self):
		self.Unet.load_state_dict(torch.load(self.path))

	def train_one_epoch(self, trainingdata, validationdata):
		loss_array = []
		val_loss_array = []
		# training loop
		loss_array.append(0)
		for i, data in enumerate(trainingdata):
			self.learner.zero_grad()
			self.Unet.train()

			# load current training image and randomly rotate it
			testimage = data["image"]
			testimage = Variable(testimage)

			if self.use_cuda:
				testimage = testimage.cuda()


			# generate wake mask 
			maskraw = self.Unet(testimage)

			# load ground truth data and rotate to the same angle as the training data
			masktrue = data["mask"]
			if self.use_cuda:
				masktrue = masktrue.cuda()

			# calculate loss of wake mask and backpropagate
			lossvalue = self.loss(maskraw, masktrue)
			loss_array[-1] += lossvalue.item()
			lossvalue.backward()
			self.learner.step()
		loss_array[-1] /= (i + 1) # determine average epoch training loss

		val_loss_array.append(0)
		# validation data
		for j, valdata in enumerate(validationdata):
			# prepare network for evaluation
			self.Unet.eval()

			# load validation data and ground truth
			val_test = valdata["image"]
			masktrue = valdata["mask"]
			if self.use_cuda:
				val_test = val_test.cuda()
				masktrue = masktrue.cuda()

			# forward propagate validation data
			with torch.no_grad():
				valmask = self.Unet(val_test)

			# calculate loss
			valloss = self.loss(valmask, masktrue)
			val_loss_array[-1] += valloss.item()

		val_loss_array[-1] /= (j + 1)

		return (loss_array[-1], val_loss_array[-1])


	def predict(self, img):
		self.Unet.eval()
		if self.use_cuda is True:
			self.Unet.cuda()
		mask = self.Unet(img)
		return mask

