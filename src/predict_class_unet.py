#!/usr/bin/env python

import os
import torch
from optparse import OptionParser
from PIL import Image
import baseline_models as models
import pytorch_unet
from torchvision import transforms
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

def predict(unet_path, classifier_path, output_path, compare, info, use_cuda,
		Cband, Sband, Xband, no_plot, classifierThreshold=0.9,
		use_large_classifier=True, process=None):
	nBands = 0
	name = ""
	for b in [Cband, Sband, Xband]:
		if b != None:
			nBands += 1
			name = os.path.basename(b)

	givenBands = []
	givenBands.append("C") if Cband != None else ""
	givenBands.append("S") if Sband != None else ""
	givenBands.append("X") if Xband != None else ""

	# load network and saved weights
	classifier = models.ClassifierUnet(band_list = givenBands, in_channels=len(givenBands), out_channels=1)
	classifier.load_state_dict(torch.load(classifier_path))
	classifier.eval()
	nChannels = classifier.in_channels


	Unet = pytorch_unet.UNet(band_list=givenBands, in_channels=nChannels, out_channels=1)
	if unet_path is not None:
		Unet.load_state_dict(torch.load(unet_path))
		Unet.eval()

	if info:
		#print("U-net uses data bands", str(unet.band_list))
		print("Classifier uses data bands", str(classifier.band_list))
		exit(0)

	if (nBands != nChannels) or (givenBands != classifier.band_list):
		raise RuntimeError("Network requires " + str(classifier.band_list)  + " inputs, " + str(givenBands) + " were given")
	if use_cuda:
		#unet.cuda()
		classifier.cuda()
		Unet.cuda()
	
	# load input
	# transforms.Normalize(mean=0.5, std=0.5)
	if process is None:
		print("No test process specified. Using default process.")
		process = transforms.Compose([
			transforms.Grayscale(),
			transforms.ToTensor(),
		])
	input = torch.Tensor(1, nBands, 1024, 1024).float()
	for i, b in enumerate(j for j in [Cband, Sband, Xband] if j != None):
		image = Image.open(b)
		input[0,i] = process(image) 
	
	if use_cuda:
		input = input.cuda()

	# process input through NN's
	with torch.no_grad():
		if unet_path is not None:
			inputMask = Unet(input)
			classifierOutput = torch.sigmoid(classifier(input, inputMask)).squeeze(0)
		else:
			classifierOutput = torch.sigmoid(classifier(input, None)).squeeze(0)

	# adjust output to useable form
	classifierOutput = classifierOutput.detach().cpu().numpy()

	'''
	print("Classifier output:",classifierOutput)
	if classifierOutput > classifierThreshold:
		print("Input contains wake")
	else:
		print("Input does not contain wake")
	'''

	# plot if desired
	if not no_plot:
		fig = plt.figure(dpi=277.3)
		if compare:
			ax = plt.subplot(121)
			# input_image = transforms.ToPILImage()(input)
			# plt.imshow(input_image, cmap="gray")
			# plt.imshow(input.cpu().detach().squeeze(0).numpy(), cmap="gray")
			plt.imshow(input.cpu().detach().numpy()[0,0,:,:], cmap="gray")
			plt.title("SAR input")
			plt.axis("off")
		plt.clim([0,1])
		if compare and unet_path is not None:
			ax = plt.subplot(122)
			# mask_image = transforms.ToPILImage()(inputMask)
			# plt.imshow(mask_image, cmap="gray")
			# plt.imshow(inputMask.cpu().detach().squeeze(0).numpy(), cmap="gray")
			plt.imshow(inputMask.cpu().detach().numpy()[0,0,:,:], cmap="gray")
			plt.title("Wake mask")
		plt.axis("off")
		# if compare:
		# 	plt.savefig(output_path, bbox_inches="tight")
		# else:
		# 	plt.savefig(output_path, bbox_inches="tight", pad_inches = 0)
		try:
			plt.savefig(f"{output_path}{name}", bbox_inches="tight", pad_inches=0)
		except:
			if not os.path.exists(output_path):
				os.mkdir(output_path)
			plt.savefig(f"{output_path}{name}", bbox_inches="tight", pad_inches=0)
		plt.close()
	return classifierOutput #classifierOutput > classifierThreshold
	

if __name__ == "__main__":
	optParser = OptionParser("%prog [options]", version="20201110")
	optParser.add_option("-u","--unet", action="store", dest="unet_path", help="path to trained PyTorch U-net model", default="./trainedModels/Unet.pth")
	optParser.add_option("-c","--classifier", action="store", dest="classifier_path", help="path to trained PyTorch classifier model", default="./trainedModels/classifier.pth")
	optParser.add_option("-i","--info", action="store_true", dest="info", help="print information about argument neural networks then quit", default=False)
	optParser.add_option("-o","--output", action="store", dest="output_path", help="path to output directory", default="./output.png")
	optParser.add_option("--Cband", action="store", dest="Cband", help="file path to C-band SAR image")
	optParser.add_option("--Sband", action="store", dest="Sband", help="file path to S-band SAR image")
	optParser.add_option("--Xband", action="store", dest="Xband", help="file path to X-band SAR image")
	optParser.add_option("--compare", action="store_true", dest="compare", help="plot input image on the same plot", default=False)
	optParser.add_option("--no-plot", action="store_true", dest="no_plot", help="", default=False)
	options, args = optParser.parse_args()
	predict(
		unet_path = options.unet_path,
		classifier_path = options.classifier_path,
		output_path = options.output_path,
		compare = options.compare,
		info = options.info,
		Cband = options.Cband,
		Sband = options.Sband,
		Xband = options.Xband,
		use_cuda = torch.cuda.is_available(),
		no_plot = options.no_plot
		)
