#!/usr/bin/env python3

import trainClassifier, predictClassifier, os, shutil, random, csv, os, torch
from find_gpu import main as find_gpu

def main(data_wake, data_nowake, bandCombos, p_train_val=0.9, p_test=0.1,
		n_epochs=15, n_iter=100, use_large_classifier=True,
		classifierThreshold=0.6, use_reduced_data=False):

	os.environ['CUDA_VISIBLE_DEVICES']=str(find_gpu())

	if use_reduced_data:
		p_train_val /= 3.0
		p_test /= 3.0

	if os.path.exists("./data"):
		shutil.rmtree("./data")
	shutil.copytree("./data_real", "./data")

	file_name = "decisionFusion"
	if use_large_classifier is True:
		file_name += "_largeClass"
	else:
		file_name += "_smallClass"
	if use_reduced_data is False:
		file_name += "_largeData"
	else:
		file_name += "_smallData"
	file_name += f"_{n_epochs}e_{n_iter}i"

	statisticsFilePath = "./" + file_name + "_statistics.csv"
	statisticsFile = open(statisticsFilePath, "a")
		
	statsfields = ["iteration","band","tp","fp","tn","fn","conf"]
	statswriter = csv.DictWriter(statisticsFile, fieldnames=statsfields)
	statswriter.writeheader()

	print("Beginning All Band Sweep")
	c_band = {"tp": 0, "fp": 0, "tn":0, "fn":0, "band": "C"}
	s_band = {"tp": 0, "fp": 0, "tn":0, "fn":0, "band": "S"}
	x_band = {"tp": 0, "fp": 0, "tn":0, "fn":0, "band": "X"}
	csx_band = {"tp": 0, "fp": 0, "tn":0, "fn":0, "band": "CSX"}
	for i in range(n_iter):
		print(f"Iteration {i}")
		print("-"*20)
		# sample reduced training, validation, and testing data
		testvalues, trainvalues = sampleTest(data_nowake, data_wake, p_train_val, p_test)
		datacombined = data_nowake + data_wake

		# register sampled data points
		prepareResultsCsv(trainvalues)
		shutil.copyfile("./current_data/Cband/results.csv", "./current_data/Sband/results.csv")
		shutil.copyfile("./current_data/Cband/results.csv", "./current_data/Xband/results.csv")

		# train
		print("Training C Band")
		trainClassifier.train(n_epochs=n_epochs, p_train=8.0/9.0,
					p_val=1.0/9.0, C=True, S=False, X=False,
					unet_path="./trainedModels/Unet_C.pth",
					classifier_path="./trainedModels/classifier_C.pth",
					use_large_classifier=use_large_classifier)
		print("Training S Band")
		trainClassifier.train(n_epochs=n_epochs, p_train=8.0/9.0,
					p_val=1.0/9.0, C=False, S=True, X=False,
					unet_path="./trainedModels/Unet_S.pth",
					classifier_path="./trainedModels/classifier_S.pth",
					use_large_classifier=use_large_classifier)
		print("Training X Band")
		trainClassifier.train(n_epochs=n_epochs, p_train=8.0/9.0,
					p_val=1.0/9.0, C=False, S=False, X=True,
					unet_path="./trainedModels/Unet_X.pth",
					classifier_path="./trainedModels/classifier_X.pth",
					use_large_classifier=use_large_classifier)
		print("Training CSX Band")
		trainClassifier.train(n_epochs=n_epochs, p_train=8.0/9.0,
					p_val=1.0/9.0, C=True, S=True, X=True,
					unet_path="./trainedModels/Unet_CSX.pth",
					classifier_path="./trainedModels/classifier_CSX.pth",
					use_large_classifier=use_large_classifier)


		# test
		results_C = {"iteration": str(i), "band": "C", "tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		results_S = {"iteration": str(i), "band": "S", "tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		results_X = {"iteration": str(i), "band": "X", "tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		results_CSX = {"iteration": str(i), "band": "CSX", "tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		ground_truth = {"iteration": str(i), "band": "Truth", "tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		for t in testvalues:
			print("Testing C Band")
			testresult_C = predictClassifier.predict(
					#unet_path = "./trainedModels/Unet.pth",
					classifier_path = "./trainedModels/classifier_C.pth",
					output_path = "C_" + t + ".png",
					compare = False, info = False,
					use_cuda = torch.cuda.is_available(),
					Cband = "./current_data/Cband/" + t + ".png",
					Sband=None, Xband=None,	no_plot = True,
					classifierThreshold = classifierThreshold,
					use_large_classifier = use_large_classifier)
			print("Testing S Band")
			testresult_S = predictClassifier.predict(
					#unet_path = "./trainedModels/Unet.pth",
					classifier_path = "./trainedModels/classifier_S.pth",
					output_path = "S_" + t + ".png",
					compare = False, info = False,
					use_cuda = torch.cuda.is_available(),
					Sband = "./current_data/Sband/" + t + ".png",
					Cband=None, Xband=None,	no_plot = True,
					classifierThreshold = classifierThreshold,
					use_large_classifier = use_large_classifier)
			print("Testing X Band")
			testresult_X = predictClassifier.predict(
					#unet_path = "./trainedModels/Unet.pth",
					classifier_path = "./trainedModels/classifier_X.pth",
					output_path = "X_" + t + ".png",
					compare = False, info = False,
					use_cuda = torch.cuda.is_available(),
					Xband = "./current_data/Xband/" + t + ".png",
					Sband=None, Cband=None,	no_plot = True,
					classifierThreshold = classifierThreshold,
					use_large_classifier = use_large_classifier)
			print("Testing CSX Band")
			testresult_CSX = predictClassifier.predict(
					#unet_path = "./trainedModels/Unet.pth",
					classifier_path = "./trainedModels/classifier_CSX.pth",
					output_path = "CSX_" + t + ".png",
					compare = False, info = False,
					use_cuda = torch.cuda.is_available(),
					Cband = "./current_data/Cband/" + t + ".png",
					Sband = "./current_data/Sband/" + t + ".png",
					Xband = "./current_data/Xband/" + t + ".png",
					no_plot = True,
					classifierThreshold = classifierThreshold,
					use_large_classifier = use_large_classifier)

			trueresult = int(t) > 31 # hackish way to get the result since all uuids from 0000 to 0031 are noSeas => no wake
			ground_truth["conf"].append(str(trueresult))

			results_C = assess_results(trueresult, testresult_C, results_C)
			results_S = assess_results(trueresult, testresult_S, results_S)
			results_X = assess_results(trueresult, testresult_X, results_X)
			results_CSX = assess_results(trueresult, testresult_CSX, results_CSX)
			# end tth test

		statswriter.writerow(results_C)
		statswriter.writerow(results_S)
		statswriter.writerow(results_X)
		statswriter.writerow(results_CSX)
		statswriter.writerow(ground_truth)

		c_band = update_band_counts(results_C, c_band)
		s_band = update_band_counts(results_S, s_band)
		x_band = update_band_counts(results_X, x_band)
		csx_band = update_band_counts(results_CSX, csx_band)

		# TODO update from here to end of function for decision level fusion
		os.remove("./current_data/Cband/results.csv")
		os.remove("./current_data/Sband/results.csv")
		os.remove("./current_data/Xband/results.csv")
		# end ith iteration

	c_final = calc_f1_score(c_band)
	s_final = calc_f1_score(s_band)
	x_final = calc_f1_score(x_band)
	csx_final = calc_f1_score(csx_band)

	#statswriter.writerow({"iteration": str(-1), "tp": str(bandtp), "fp": str(bandfp), "tn": str(bandtn), "fn": str(bandfn), "conf": [str(precision), str(recall), str(F1)]})
	statswriter.writerow(c_final)
	statswriter.writerow(s_final)
	statswriter.writerow(x_final)
	statswriter.writerow(csx_final)
	statisticsFile.close()
	# end bth band


def calc_f1_score(band_dict):
	precision = float(band_dict["tp"]) / (float(band_dict["tp"] + band_dict["fp"]) + 1e-10)
	recall = float(band_dict["tp"]) / (float(band_dict["tp"] + band_dict["fn"]) + 1e-10)
	F1 = 2.0 * (precision * recall) / (precision + recall + 1e-10)
	return {"iteration": str(-1), "band": band_dict["band"], 
		"tp": str(band_dict["tp"]), "fp": str(band_dict["fp"]),
		"tn": str(band_dict["tn"]), "fn": str(band_dict["fn"]), 
		"conf": [str(precision), str(recall), str(F1)]}


def update_band_counts(results_dict, band_dict):
	band_dict["tp"] += results_dict["tp"]
	band_dict["fp"] += results_dict["fp"]
	band_dict["tn"] += results_dict["tn"]
	band_dict["fn"] += results_dict["fn"]
	return band_dict


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
	data_nowake = ["000" + str(t) for t in range(10)] + ["00" + str(t) for t in range(11,32)]
	data_wake = ["00" + str(t) for t in range(32, 95)]

	bandCombos = ["C", "S", "X", "CSX"]

	use_reduced_data = False

	p_train_val = 0.9
	p_test = 0.1
	
	# n_epochs = 15
	n_epochs = 30
	n_iter = 100

	# testing values
	#n_epochs = 3
	#n_iter = 3

	use_large_classifier = True 
	classifierThreshold = 0.6

	main(data_wake, data_nowake, bandCombos, p_train_val=p_train_val,
		p_test=p_test, n_epochs=n_epochs, n_iter=n_iter,
		use_large_classifier=use_large_classifier,
		classifierThreshold=classifierThreshold)

