#!python3

import os
import shutil
import train_unet
import train_class
import trainClassifier
import predictClassifier
import predict_class_unet
from helperFunctions import update_band_counts
import train_model
from unet_model import UnetModel

def train(n_epochs, unet_epochs, p_train, p_val, C, S, X, unet_path, classifier_path, n_iter, bce_weight=None,
		unet_weight_decay=None, unet_learning_rate=None, transforms=None,
		class_weight_decay=None, class_learning_rate=None, progress=False):
	class_path = classifier_path

	'''
	# data: count the number of bands needed
	bands = []
	if C:
		bands.append("C")
	if S:
		bands.append("S")
	if X:
		bands.append("X")
	n_channels = len(bands)
	# train the unet model
	unet_model = UnetModel(bands, n_channels,  unet_path, bce_weight=bce_weight,
				weight_decay=unet_weight_decay,
				learning_rate=unet_learning_rate)
	train_model.train(unet_model, n_epochs, p_train, p_val, C, S, X, unet_path,
				class_path, n_batch=4, progress=progress, n_iter=n_iter, 
				transforms=transforms)
	'''
	try:
		# train the unet model
		train_unet.train(unet_epochs, p_train, p_val, C, S, X, unet_path, class_path,
					n_batch=4, progress=True, n_iter=n_iter, bce_weight=bce_weight,
					weight_decay=unet_weight_decay, learning_rate=unet_learning_rate,
					transforms=transforms)
	except Exception as error:
		print("Error encountered training U-Net model")
		print(error)

	try:
		# train classifier using the unet model
		train_class.train(n_epochs, p_train, p_val, C, S, X, unet_path,
					class_path+'_unet.pth',	n_batch=1,
					progress=progress,
					weight_decay=class_weight_decay,
					learning_rate=class_learning_rate,
					transforms=transforms)
	except Exception as error:
		print("Error encountered training unet Classifier model")
		print(error)

	try:
		# train the classifier without unet model
		train_class.train(n_epochs, p_train, p_val, C, S, X, None,
					class_path+'.pth',
					n_batch=1,
					progress=progress,
					weight_decay=class_weight_decay,
					learning_rate=class_learning_rate,
					transforms=transforms)
	except Exception as error:
		print("Error encountered training clss Classifier model")
		print(error)

	try:
		# train the baseline classifier
		trainClassifier.train(n_epochs, p_train, p_val, C, S, X, None,
					class_path+'_baseline.pth',
					progress=progress,
					transforms=transforms)
	except Exception as error:
		print("Error encountered training base Classifier model")
		print(error)


def predict(overall_results, statswriter, unet_path, classifier_path,
		output_path, compare, info, use_cuda, C, S, X, no_plot,
		classifierThreshold, testvalues, test_process, testing_dirs,
		iteration, train_bands):
	class_path = classifier_path
	bands = []
	if C:
		bands.append("C")
	if S:
		bands.append("S")
	if X:
		bands.append("X")
	bands = ''.join(bands)

	if type(classifierThreshold) is dict:
		threshold = classifierThreshold.get(bands, 0.6)
		if bands not in classifierThreshold.keys():
			print("using default threshold: 0.6")
	elif type(classifierThreshold) is float:
		threshold = classifierThreshold
	else:
		threshold = 0.6
		print("using default threshold: 0.6")

	i = str(iteration)
	if not no_plot:
		if os.path.exists(output_path):
			shutil.rmtree(output_path)
		os.mkdir(output_path)

	for test_dir in testing_dirs:
		print(f"testing with {test_dir}")
		if os.path.exists("./data_test"):
			shutil.rmtree("./data_test")
		shutil.copytree(test_dir, "./data_test")
		results_unet = {"iteration": i, "band": train_bands, "model": "unet",
			"test_dir": os.path.basename(test_dir),
			"test_band": bands,
			"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		results_clss = {"iteration": i, "band": train_bands, "model": "clss",
			"test_dir": os.path.basename(test_dir),
			"test_band": bands,
			"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
		results_base = {"iteration": i, "band": train_bands, "model": "base",
			"test_dir": os.path.basename(test_dir),
			"test_band": bands,
			"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}

		plot_output_path = f"{output_path}{os.path.basename(test_dir)}/"
		if not no_plot:
			if os.path.exists(plot_output_path):
				shutil.rmtree(plot_output_path)
			os.mkdir(plot_output_path)

		# run predictions for these testvalues in this directory
		predict_on_dir(testvalues, results_unet, results_clss, results_base,
				unet_path, class_path, plot_output_path, compare, info,
				use_cuda, C, S, X, no_plot, threshold,
				test_process)

		testresults = [results_unet, results_clss, results_base]
		for result in testresults:
			statswriter.writerow(result)
		update_band_counts(testresults, overall_results)

	'''
	ground_truth = {"iteration": i, "band": "Truth", "model": "Truth",
		"test_dir": "n/a",
		"tp": 0, "fp": 0, "tn": 0, "fn": 0, "conf": []}
	for t in testvalues:
		trueresult = int(t) > 31 # hackish way to get the result
		ground_truth["conf"].append(str(trueresult))
	statswriter.writerow(ground_truth)
	'''


def predict_on_dir(testvalues, results_unet, results_clss, results_base,
			unet_path, class_path, output_path, compare, info,
			use_cuda, C, S, X, no_plot, threshold,
			test_process):
	# iterate through test images for this directory and save results to the
	# dictionaries
	for t in testvalues:
		Cband = "./data_test/Cband/" + t + ".png" if C else None
		Sband = "./data_test/Sband/" + t + ".png" if S else None
		Xband = "./data_test/Xband/" + t + ".png" if X else None

		# predict classifier using the unet model
		testresult_unet = predict_class_unet.predict(
					unet_path, class_path+'_unet.pth', output_path,
					compare, info, use_cuda, Cband, Sband,
					Xband, no_plot, threshold,
					process=test_process)

		# train the classifier without unet model
		testresult_clss = predict_class_unet.predict(
					None, class_path+'.pth', output_path,
					compare, info, use_cuda, Cband, Sband,
					Xband, no_plot, threshold,
					process=test_process)

		# train the baseline classifier
		testresult_base = predictClassifier.predict(
					class_path+'_baseline.pth', output_path,
					compare, info, use_cuda, Cband, Sband,
					Xband, no_plot, threshold,
					process=test_process)

		trueresult = int(t) > 31 # hackish way to get the 

		results_unet = assess_results(trueresult, testresult_unet, results_unet, threshold)
		results_clss = assess_results(trueresult, testresult_clss, results_clss, threshold)
		results_base = assess_results(trueresult, testresult_base, results_base, threshold)


def assess_results(trueresult, testresult, results_dict, threshold):
	roundedtestresult = testresult > threshold
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

