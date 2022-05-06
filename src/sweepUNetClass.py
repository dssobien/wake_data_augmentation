#!/usr/bin/env python3

import pandas as pd
import trainClassifier, predictClassifier, os, shutil, random, csv, os, torch
import run_sweep
from find_gpu import find_gpu
from helperFunctions import calc_f1_score, update_band_counts, assess_results
from helperFunctions import sampleTest, sampleTestFromFile, prepareResultsCsv 
from helperFunctions import testFoldFromFile, get_training_img_indexes 
import data_augmentation
import torchvision
import argparse

def main(data_wake, data_nowake, p_train_val=0.9, p_test=0.1,
		n_epochs=30, n_iter=10, classifierThreshold=0.6, 
		bands=None, test_file_name=None, bce_weight=None,
		unet_learning_rate=None, unet_weight_decay=None,
		class_learning_rate=None, class_weight_decay=None,
		transforms=None, test_process=None,
		data_dir="./data_fusion_expanded_runs_dataset",
		testing_dirs=None, unet_epochs=30, test_bands=None,
		fold=None):

	os.environ['CUDA_VISIBLE_DEVICES']=str(find_gpu())

	if os.path.exists("./current_data"):
		shutil.rmtree("./current_data")
	shutil.copytree(data_dir, "./current_data")

	# if no specific testing directory given, use the data_dir
	# this will pull testing images from the same set as training
	# images, but will not use the exact same images
	if testing_dirs is None:
		testing_dirs = [data_dir]

	file_name = "unet_classifier_sweep"
	file_name += f"_{n_epochs}ec_{unet_epochs}eu_{n_iter}i"
	if fold is not None:
		file_name += f"_fold{fold}"

	statisticsFilePath = "./" + file_name + "_statistics.csv"
	statisticsFile = open(statisticsFilePath, "a")
		
	statsfields = ["iteration","band","model","test_dir","test_band","tp","fp","tn","fn","conf"]
	statswriter = csv.DictWriter(statisticsFile, fieldnames=statsfields)
	statswriter.writeheader()

	if bands is None:
		bands = ['C', 'S', 'X', 'CSX']

	print("Beginning All Band Sweep")
	overall_results = {}
	for band in bands:
		overall_results.setdefault(band, 
				{"unet": {"tp": 0, "fp": 0, "tn":0, "fn":0},
				"clss": {"tp": 0, "fp": 0, "tn":0, "fn":0},
				"base": {"tp": 0, "fp": 0, "tn":0, "fn":0}})

	sweep_train_test_models(data_wake, data_nowake, p_train_val, p_test,
				n_epochs, unet_epochs, n_iter, classifierThreshold,
				bce_weight, unet_weight_decay, bands, 
				test_file_name, unet_learning_rate,
				overall_results, statswriter,
				transforms, test_process, testing_dirs,
				class_learning_rate, class_weight_decay,
				test_bands, fold)

	for band in bands:
		for model in ["unet", "clss", "base"]:
			statswriter.writerow(calc_f1_score(overall_results, band, model))

	statisticsFile.close()


def sweep_train_test_models(data_wake, data_nowake, p_train_val, p_test,
				n_epochs, unet_epochs, n_iter, classifierThreshold,
				bce_weight, unet_weight_decay, bands, 
				test_file_name, unet_learning_rate,
				overall_results, statswriter,
				transforms, test_process, testing_dirs,
				class_learning_rate, class_weight_decay,
				test_bands=None, fold=None):
	for i in range(n_iter):
		print(f"Iteration {i}")
		print("-"*20)
		# sample reduced training, validation, and testing data
		if test_file_name is None:
			testvalues, trainvalues = sampleTest(data_nowake, data_wake, p_train_val, p_test)
		elif fold is not None:
			testvalues, trainvalues = testFoldFromFile(data_nowake, data_wake, test_file_name)
		else:
			testvalues, trainvalues = sampleTestFromFile(data_nowake, data_wake, p_train_val,
									p_test, test_file_name)
		for v in testvalues:
			if v in trainvalues:
				raise ValueError(f"uuid: {v} in testing and training sets")
		datacombined = data_nowake + data_wake
		# write the training image uuids to check them later
		with open(f"output/fold{fold}_train.txt", "w") as fout:
			fout.write("\n".join(str(item) for item in trainvalues))

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
			run_sweep.train(n_epochs=n_epochs, unet_epochs=unet_epochs,
					p_train=8.0/9.0,
					p_val=1.0/9.0, C=C, S=S, X=X,
					unet_path=f"./trainedModels/Unet_{band}.pth",
					classifier_path=f"./trainedModels/classifier_{band}",
					n_iter=i, bce_weight=bce_weight,
					unet_learning_rate=unet_learning_rate,
					unet_weight_decay=unet_weight_decay,
					transforms=transforms,
					class_learning_rate=class_learning_rate,
					class_weight_decay=class_weight_decay,
					)
		
			if test_bands is None:
				test_band_sweep = [band]
			else:
				test_band_sweep = test_bands
			for test_band in test_band_sweep:
				C_test = False
				S_test = False
				X_test= False
				if test_band == 'C':
					C_test = True
				elif test_band == 'S':
					S_test = True
				elif test_band == 'X':
					X_test = True
				elif test_band == 'CSX':
					C_test = True
					S_test = True
					X_test = True

				# test
				print(f"Testing {band} Band on {test_band} Band Data")
				run_sweep.predict(overall_results, statswriter,
						unet_path=f"./trainedModels/Unet_{band}.pth",
						classifier_path=f"./trainedModels/classifier_{band}",
						output_path=f"output/test_{band}_{i}i_test{test_band}/",
						compare=True, info=False,
						use_cuda=torch.cuda.is_available(),
						C=C_test, S=S_test, X=X_test, no_plot=True,
						classifierThreshold=classifierThreshold,
						testvalues=testvalues,
						test_process=test_process,
						testing_dirs=testing_dirs,
						iteration=i, train_bands=band)
			print(f"{band} Band testing complete for iteration {i}")

		ground_truth = {"iteration": str(i), "band": "Truth", "model": "Truth",
				"test_dir": "n/a", "test_band": "n/a",
				"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		for t in testvalues:
			trueresult = int(t) > 31 # hackish way to get the result
			ground_truth["conf"].append(str(trueresult))
		statswriter.writerow(ground_truth)
		print(f"all testing complete for iteration {i}")

		# TODO update from here to end of function for decision level fusion
		os.remove("./current_data/Cband/results.csv")
		os.remove("./current_data/Sband/results.csv")
		os.remove("./current_data/Xband/results.csv")
		# end ith iteration


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run a training and testing "
													"sweep for all three models.")
	test_names_default = ["test_set_fold0.txt", "test_set_fold1.txt",
							"test_set_fold2.txt", "test_set_fold3.txt"]
	test_names_default = [os.path.join("data", x) for x in test_names_default]
	parser.add_argument("--test_file_names", type=str, nargs='*',
						default=test_names_default,
						help="list file names that define the test splits")
	parser.add_argument("--train_dir", type=str, nargs=1,
						help="directory that contains all the training data")
	parser.add_argument("--train_data_dirs", type=str, nargs="*",
						help="list of subdirectories that contain specific "
								"sets of training data")
	parser.add_argument("--test_dir", type=str, nargs=1,
						help="directory that contains all the testing data")
	parser.add_argument("--test_data_dirs", type=str, nargs="*",
						help="list of subdirectories that contain specific "
								"sets of testing data")
	
	parser.add_argument("--train_bands", type=str, nargs="+",
						help="list of bands to use for training")
	parser.add_argument("--test_bands", type=str, nargs="*",
						help="bands for testing on each model. If you want to "
							"test a model only on the band it was trained on "
							"DO NOT use this argument.",
						default=None)
	parser.add_argument("--n_epochs", type=int, default=60)
	parser.add_argument("--unet_epochs", type=int, default=60)
	parser.add_argument("--n_iter", type=int, default=5)

	args = parser.parse_args()
	test_file_names = args.test_file_names
	base_dir = args.train_dir[0]
	data_dirs = args.train_data_dirs
	augmented_base_dir = args.test_dir[0]
	augmented_dirs = args.test_data_dirs
	bands = args.train_bands
	test_bands = args.test_bands
	n_epochs = args.n_epochs
	unet_epochs = args.unet_epochs
	n_iter = args.n_iter

	# base_dir = "./combined_dataset"
	# # base_dir = "./augmented_dataset"
	# # data_dirs = ["rotation_S_2D_1", "rotation_S_2D_2"]
	# data_dirs = ["rotation_X_2D_1", "rotation_X_2D_2"]
	# # data_dirs = ["rotation_0"]

	# # augmented_base_dir = "./augmented_dataset"
	# augmented_base_dir = "./augmented_dataset/circular_crop"
	# augmented_dirs = ["rotation_0", "rotation_90"]
	# # augmented_dirs = ["rotation_0", "rotation_15", "rotation_30", "rotation_45",
	# # 			"rotation_60", "rotation_75", "rotation_90", "rotation_105",
	# # 			"rotation_120", "rotation_135", "rotation_150", "rotation_165",
	# # 			"rotation_180", "rotation"]

	# # bands = ['C', 'S', 'X'] 
	# bands = ['C']
	# test_bands = ['C', 'S', 'X']
	# test_bands = None  # if None, tests only on the band(s) used for training

	# n_epochs = 60
	# unet_epochs = 60
	# n_iter = 5

	data_dirs = [os.path.join(base_dir, x) for x in data_dirs]
	augmented_dirs = [os.path.join(augmented_base_dir, x) for x in augmented_dirs]

	p_train_val = 0.9
	p_test = 0.1

	classifierThreshold = 0.6
	bce_weight = 0.7
	unet_learning_rate = 1e-4
	unet_weight_decay = 1e-4
	class_learning_rate = 1e-4
	class_weight_decay = 1e-4

	# transforms = torchvision.transforms.Compose([
		# data_augmentation.RandomPerspective(0.25),
		# data_augmentation.RandomRotation((0, 359), probability=0.5),
		# data_augmentation.RandomCrop(768),
		# data_augmentation.RandomNoise(0.003),
		# data_augmentation.Rescale(1024)
	# ])
	transforms = None

	# test set image processing and transforms/augmentations
	test_process = torchvision.transforms.Compose([
		torchvision.transforms.Grayscale(),
		torchvision.transforms.ToTensor(),
	])

	for data_dir in data_dirs:
		print(f"starting {data_dir}")
		results_df = pd.read_csv(os.path.join(data_dir, "results.csv"))
		output_dir = f"outputs_{os.path.basename(data_dir)}"
		os.mkdir(output_dir)
		for fold, test_file_name in enumerate(test_file_names):
			if not os.path.exists("output"):
				os.mkdir("output")
			data_nowake, data_wake = get_training_img_indexes(results_df, fold=fold)

			main(data_wake, data_nowake, p_train_val=p_train_val,
				p_test=p_test, n_epochs=n_epochs, n_iter=n_iter,
				classifierThreshold=classifierThreshold,
				bands=bands, test_file_name=test_file_name,
				bce_weight=bce_weight, unet_learning_rate=unet_learning_rate,
				unet_weight_decay=unet_weight_decay,
				transforms=transforms, test_process=test_process,
				class_weight_decay=class_weight_decay,
				class_learning_rate=class_learning_rate,
				data_dir=data_dir, testing_dirs=augmented_dirs,
				unet_epochs=unet_epochs, test_bands=test_bands,
				fold=fold)

			output_file = f"unet_classifier_sweep_{n_epochs}ec_{unet_epochs}eu_{n_iter}i_fold{fold}_statistics.csv"
			# new_dir = f"outputs_{os.path.basename(data_dir)}_fold{fold}"
			new_dir = os.path.join(output_dir, f"fold{fold}")
			if os.path.exists(new_dir):
				shutil.rmtree(new_dir)
			new_file = os.path.join(new_dir, f"{os.path.basename(data_dir)}_{output_file}")

			shutil.copytree("output", new_dir)
			shutil.copyfile(output_file, new_file)

			if os.path.exists(output_file):
				os.remove(output_file)
			if os.path.exists("output"):
				shutil.rmtree("output")

