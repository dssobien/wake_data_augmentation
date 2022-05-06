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
import csv
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.transforms.functional as TF
import random
from PIL import Image
import diceloss as DL
from tqdm import tqdm
from optparse import OptionParser
from find_gpu import find_gpu


def maskToImage(mask, use_cuda):
	"""
	Taking a wake mask in the form of a PyTorch tensor, return a numpy array
	"""
	if use_cuda:
		return mask.cpu().detach().numpy()[0,0,:,:]
	return mask.detach().numpy()[0,0,:,:]


def rotateImage(image_tensor, angle=None):
	"""
	Rotate an image. If angle argument is specified, rotate the image to the specified angle.
	If angle argument is not specified, rotate the image in a random 90 degree increment.
	Returns the angle in addition to the rotated tensor
	"""
	if angle == None:
		angle = random.randint(0,3) * 90
	final_images = []
	for i in range(image_tensor.size(0)):
		image = image_tensor[i].unsqueeze(0)
		image = image.squeeze(0)
		image = TF.to_pil_image(image)
		image = TF.rotate(image, angle)
		image = TF.to_tensor(image)
		image = image.unsqueeze(0)
		final_images.append(image)
	image_tensor_out = torch.cat(final_images, 0)
	return image_tensor_out, angle


def get_trainable_params(model):
	"""
	Return the number of trainable parameters in a network
	"""
	params = filter(lambda p: p.requires_grad, model.parameters())
	return sum([np.prod(p.size()) for p in params])


def calc_f1_score(master_results, band, model):
	band_dict = master_results[band][model]
	precision = float(band_dict["tp"]) / (float(band_dict["tp"] + band_dict["fp"]) + 1e-10)
	recall = float(band_dict["tp"]) / (float(band_dict["tp"] + band_dict["fn"]) + 1e-10)
	F1 = 2.0 * (precision * recall) / (precision + recall + 1e-10)
	return {"iteration": str(-1), "band": band, "model": model, 
		"tp": str(band_dict["tp"]), "fp": str(band_dict["fp"]),
		"tn": str(band_dict["tn"]), "fn": str(band_dict["fn"]), 
		"conf": [str(precision), str(recall), str(F1)]}


def update_band_counts(results_dict, master_results):
	for r_dict in results_dict:
		band = r_dict["band"]
		model = r_dict["model"]
		master_results[band][model]["tp"] += r_dict["tp"]
		master_results[band][model]["fp"] += r_dict["fp"]
		master_results[band][model]["tn"] += r_dict["tn"]
		master_results[band][model]["fn"] += r_dict["fn"]


def assess_results(trueresult, testresult, results_dict):
	roundedtestresult = testresult > classifierThreshold
	if trueresult == True and roundedtestresult == trueresult:
		# true positive
		results_dict["tp"] += 1
	elif trueresult == True and roundedtestresult != trueresult:
		# false negative
		results_dict["fn"] += 1
	elif trueresult == False and roundedtestresult == trueresult:
		# true negative
		results_dict["tn"] += 1
	else:
		# false positive
		results_dict["fp"] += 1
	results_dict["conf"].append(str(testresult))
	return results_dict


def sampleTest(no_wake, wake, p_train_val, p_test):
	random.shuffle(no_wake)
	random.shuffle(wake)
	n_train_val = round((len(no_wake) + len(wake)) * p_train_val / 2.0)
	n_test = round((len(no_wake) + len(wake)) * p_test / 2.0)

	return no_wake[:n_test] + wake[:n_test], no_wake[n_test:n_train_val + n_test] + wake[n_test:n_train_val + n_test]


def sampleTestFromFile(no_wake, wake, p_train_val, p_test, test_file_name):
	random.shuffle(no_wake)
	random.shuffle(wake)
	n_train_val = round((len(no_wake) + len(wake)) * p_train_val / 2.0)
	n_test = round((len(no_wake) + len(wake)) * p_test / 2.0)
	# read in the ids for test
	print(f"using {test_file_name} for test cases")
	with open(test_file_name, 'r') as fin:
		lines = fin.readlines()
	test_ids = [l.strip('\n') for l in lines[1:]]

	i_max = len(wake)
	i = 0
	train_ids = [x for x in no_wake if x not in test_ids]
	while (len(train_ids) < 120) and (i < i_max):
		if wake[i] not in test_ids:
			train_ids.append(wake[i])
		i += 1

	random.shuffle(test_ids)
	random.shuffle(train_ids)
	return test_ids, train_ids


def testFoldFromFile(no_wake, wake, file_name):
	# read in the ids for test fold
	print(f"using {file_name} for test cases")
	with open(file_name, 'r') as fin:
		lines = fin.readlines()
	test_ids = [l.strip('\n') for l in lines[1:]]
	
	# get training images from no wake and wake index lists
	train_no_wake_ids = [x for x in no_wake if x not in test_ids]
	train_wake_ids = [x for x in wake if x not in test_ids]
	# combine the training lists and shuffle
	train_ids = train_no_wake_ids + train_wake_ids
	random.shuffle(train_ids)

	return test_ids, train_ids


def prepareResultsCsv(train_val):
	newFile = "./current_data/Cband/results.csv"
	dataFile = "./current_data/results.csv"
	with open(newFile, "w") as newResults:
		fieldNames = ["uuid", "run_name", "band", "look_angle", "polarization", "inclination_angle", "contains_wake", "kfold", "aug_fold"]
		writer = csv.DictWriter(newResults, fieldnames=fieldNames)
		writer.writeheader()

		with open(dataFile, "r") as data:
			reader = csv.DictReader(data)
			for row in reader:
				uuid = ("0000" + row["uuid"])[-4:]
				if uuid in train_val:
					writer.writerow(row)	


def read_hyperparameters_file(file_name):
	with open(file_name, 'r') as fin:
		lines = fin.readlines()
	lines = [x.strip('\n').split(',') for x in lines]
	keys = lines[0]
	values = [np.array(x, dtype='float') for x in lines[1:]]
	values = np.array(values, dtype='float')
	hp_dict = {}

	for i, key in enumerate(keys):
		hp_dict[key] = values[:,i]

	return hp_dict


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
		maskName = (fileName.split(".")[0])[:-4] # if fileName = "testMask.png", this line saves "test" as the maskName
		masks[maskName] = process(maskImage)
	return masks


def make_string_list(array, prefix='0000', length=4):
    new_list = [f"{prefix}{x}" for x in array]
    return [x[-length:] for x in new_list]


def get_training_img_indexes(results_df, fold=None):
    if fold is not None and "kfold" in results_df.columns:
        results_df = results_df.query(f"kfold != {fold}")
    no_wake = results_df.query("contains_wake == 0")["uuid"].values
    wake = results_df.query("contains_wake == 1")["uuid"].values
    return make_string_list(no_wake), make_string_list(wake)

