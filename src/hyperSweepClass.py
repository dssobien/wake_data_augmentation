#!/usr/bin/env python3

import trainClassifier, predictClassifier, os, shutil, random, csv, os, torch
import run_sweep
from find_gpu import find_gpu
from helperFunctions import calc_f1_score, update_band_counts, assess_results
from helperFunctions import sampleTest, sampleTestFromFile, prepareResultsCsv 
from helperFunctions import read_hyperparameters_file 
import torchvision
import data_augmentation

def main(data_wake, data_nowake, p_train_val=0.9, p_test=0.1,
		n_epochs=30, n_iter=10, classifierThreshold=0.6, 
		bands=None, test_file_name=None,
		hp_file_name='hyperparameters.txt',
		transforms=None):

	os.environ['CUDA_VISIBLE_DEVICES']=str(find_gpu())

	if os.path.exists("./data"):
		shutil.rmtree("./data")
	shutil.copytree("./data_fusion_expanded_runs_dataset", "./data")

	file_name = "hyperparameter_class_sweep"
	file_name += f"_{n_epochs}e_{n_iter}i"

	statisticsFilePath = "./" + file_name + "_statistics.csv"
	statisticsFile = open(statisticsFilePath, "a")
		
	statsfields = ["iteration", "hyperparam_set","band","model","tp","fp","tn","fn","conf"]
	statswriter = csv.DictWriter(statisticsFile, fieldnames=statsfields)
	statswriter.writeheader()

	if bands is None:
		bands = ['C', 'S', 'X', 'CSX']

	print("Beginning All Band Sweep")
	master_results = {}
	for band in bands:
		master_results.setdefault(band, 
				{"unet": {"tp": 0, "fp": 0, "tn":0, "fn":0},
				"clss": {"tp": 0, "fp": 0, "tn":0, "fn":0},
				"base": {"tp": 0, "fp": 0, "tn":0, "fn":0}})

	hp_dict = read_hyperparameters_file(hp_file_name)
	hyperparameters = list(hp_dict.keys())
	hp_len = len(hp_dict[hyperparameters[0]])

	for k in range(hp_len):
		print(f"Hyperparameter Set {k}")
		print("-"*20)
		class_learning_rate = hp_dict['learning_rate'][k] 
		class_weight_decay = hp_dict['weight_decay'][k] 
		sweep_train_test_models(data_wake, data_nowake, p_train_val, p_test,
					n_epochs, n_iter, classifierThreshold,
					class_weight_decay, bands, 
					test_file_name, class_learning_rate,
					master_results, statswriter,
					hyperparam_set=k, transforms=transforms)
	statisticsFile.close()


def sweep_train_test_models(data_wake, data_nowake, p_train_val, p_test,
				n_epochs, n_iter, classifierThreshold,
				class_weight_decay, bands, 
				test_file_name, class_learning_rate,
				master_results, statswriter,
				hyperparam_set, transforms):
	bce_weight = 0.7
	unet_learning_rate = 1e-4
	unet_weight_decay = 1e-4
	for i in range(n_iter):
		print(f"Iteration {i}")
		print("-"*20)
		# sample reduced training, validation, and testing data
		if test_file_name is None:
			testvalues, trainvalues = sampleTest(data_nowake, data_wake, p_train_val, p_test)
		else:
			testvalues, trainvalues = sampleTestFromFile(data_nowake, data_wake, p_train_val,
									p_test, test_file_name)
		for v in testvalues:
			if v in trainvalues:
				raise ValueError(f"uuid: {v} in testing and training sets")
		datacombined = data_nowake + data_wake

		# register sampled data points
		prepareResultsCsv(trainvalues)
		shutil.copyfile("./current_data/Cband/results.csv", "./current_data/Sband/results.csv")
		shutil.copyfile("./current_data/Cband/results.csv", "./current_data/Xband/results.csv")

		for band in bands:
			C = False
			S = False
			X = False
			if band == 'C':
				C = True
			elif band == 'S':
				S = True
			elif band == 'X':
				X = True
			elif band == 'CSX':
				C = True
				S = True
				X = True
				
			# train
			print(f"Training {band} Band")
			run_sweep.train(n_epochs=n_epochs, p_train=8.0/9.0,
					p_val=1.0/9.0, C=C, S=S, X=X,
					unet_path=f"./trainedModels/Unet_{band}.pth",
					classifier_path=f"./trainedModels/classifier_{band}",
					n_iter=i, bce_weight=bce_weight,
					unet_learning_rate=unet_learning_rate,
					unet_weight_decay=unet_weight_decay,
					class_learning_rate=class_learning_rate,
					class_weight_decay=class_weight_decay,
					transforms=transforms)

			# test
			print(f"Testing {band} Band")
			testresult = run_sweep.predict(
					unet_path = f"./trainedModels/Unet_{band}.pth",
					classifier_path = f"./trainedModels/classifier_{band}",
					output_path = f"{band}_.png",
					compare = False, info = False,
					use_cuda = torch.cuda.is_available(),
					C=C, S=S, X=X, no_plot=True,
					classifierThreshold = classifierThreshold,
					testvalues=testvalues)

			for result in testresult:
				result['iteration'] = str(i)
				result['hyperparam_set'] = str(hyperparam_set)
				statswriter.writerow(result)
			update_band_counts(testresult, master_results)


		ground_truth = {"iteration": str(i), "band": "Truth", "model": "Truth",
				"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": [],
				"hyperparam_set": str(hyperparam_set)}
		for t in testvalues:
			trueresult = int(t) > 31 # hackish way to get the result
			ground_truth["conf"].append(str(trueresult))
		statswriter.writerow(ground_truth)

		# TODO update from here to end of function for decision level fusion
		os.remove("./current_data/Cband/results.csv")
		os.remove("./current_data/Sband/results.csv")
		os.remove("./current_data/Xband/results.csv")
		# end ith iteration


if __name__ == "__main__":
	data_nowake = ["000" + str(t) for t in range(10)] + ["00" + str(t) for t in range(11,32)]
	data_wake = ["00" + str(t) for t in range(32, 100)] + ["0" + str(t) for t in range(100, 224)]

	test_file_name = "test_set_v0.txt"
	# hp_file_name = "hyperparameters.txt"
	hp_file_name = "hyperparameters_2.txt"

	p_train_val = 0.9
	p_test = 0.1
	
	n_epochs = 60
	# n_iter = 10
	# testing values
	# n_epochs = 5
	n_iter = 1

	classifierThreshold = 0.825

	bands = ['C']  # , 'S']

	transforms = torchvision.transforms.Compose([
			data_augmentation.RandomPerspective(0.25),
			data_augmentation.RandomRotation((0, 359), 10),
			data_augmentation.RandomCrop(768),
			data_augmentation.RandomNoise(0.003),
			data_augmentation.Rescale(1024)
			])

	main(data_wake, data_nowake, p_train_val=p_train_val,
		p_test=p_test, n_epochs=n_epochs, n_iter=n_iter,
		classifierThreshold=classifierThreshold,
		bands=bands, test_file_name=test_file_name,
		hp_file_name=hp_file_name,
		transforms=transforms)

