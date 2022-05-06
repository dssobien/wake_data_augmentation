from torchvision import datasets, transforms
import scipy
import torch.utils.data
import os, csv
import torchvision
import torchvision.transforms
import torchvision.transforms.functional as TF
from PIL import Image

class SARDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, band_list, mask_dict):
		"""
		data_dir: path to the Cband/Sband/Xband folders
		band_list: a list of bands which will be used e.g. ["C", "X"] or ["S"]
		mask_dict: dictionary of the mask files
		"""
		self.samples = []
		self.data_dir = data_dir
		self.band_list = band_list
		self.mask_dict = mask_dict
		
		with open(data_dir + band_list[0] + "band/results.csv", "r") as csvfile:
			reader = csv.DictReader(csvfile)

			for row in reader:
				self.samples.append(row)
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		currSample = self.samples[idx]
		imageName = "0000" + currSample["uuid"] + ".png"
		imageName = imageName[-8:]

		process = torchvision.transforms.Compose([
			torchvision.transforms.Grayscale(),
			torchvision.transforms.ToTensor()
		])

		image = None 
		for i, b in enumerate(self.band_list):
			newImage = Image.open(self.data_dir + b + "band/" + str(imageName))
			newImage = process(newImage)
			if image == None:
				image = torch.Tensor(len(self.band_list), newImage.shape[1], newImage.shape[2])
			image[i] = newImage
		currSample["image"] = image
		currSample["mask"] = self.mask_dict[currSample["run_name"]]
		return currSample
