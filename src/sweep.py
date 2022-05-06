#!/usr/bin/env python3

import trainClassifier, predictClassifier, os, shutil, random, csv, os, torch
import random, time
from find_gpu import main as find_gpu

def main(data_wake, data_nowake, bandCombos, p_train_val=0.9, p_test=0.1,
		n_epochs=15, n_iter=100, use_large_classifier=False,
		classifierThreshold=0.6, use_reduced_data=False):

	os.environ['CUDA_VISIBLE_DEVICES']=str(find_gpu())

	if use_reduced_data:
		p_train_val /= 3.0
		p_test /= 3.0

	if os.path.exists("./data"):
		shutil.rmtree("./data")
	shutil.copytree("./data_real", "./data")

	tmp_num = f"{round(time.time())}{random.randint(10000, 99999)}"
	unet_path=f"./trainedModels/Unet_{tmp_num}.pth"
	classifier_path=f"./trainedModels/classifier_{tmp_num}.pth"

	for b in bandCombos:
		if "C" in b:
			C = True
		else:
			C = False
		if "S" in b:
			S = True
		else:
			S = False
		if "X" in b:
			X = True
		else:
			X = False

		file_name = b
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
		
		statsfields = ["iteration","tp","fp","tn","fn","conf"]
		statswriter = csv.DictWriter(statisticsFile, fieldnames=statsfields)
		statswriter.writeheader()

		print("Beginning band(s)",b)
		bandtp = 0
		bandtn = 0
		bandfp = 0
		bandfn = 0
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
			trainClassifier.train(
						n_epochs=n_epochs,
						p_train=8.0/9.0,
						p_val=1.0/9.0,
						C=C,
						S=S,
						X=X,
						unet_path=unet_path,
						classifier_path=classifier_path,
						use_large_classifier=use_large_classifier
					      )


			# test
			tp = 0 # true positives
			tn = 0 # true negatives
			fp = 0 # false positives
			fn = 0 # false negatives
			conf = [] # confidence
			for t in testvalues:
				testresult = predictClassifier.predict(
						unet_path=unet_path,
						classifier_path = classifier_path,
						output_path = b + "_" + t + ".png",
						compare = False,
						info = False,
						use_cuda = torch.cuda.is_available(),
						Cband = "./current_data/Cband/" + t + ".png" if C else None,
						Sband = "./current_data/Sband/" + t + ".png" if S else None,
						Xband = "./current_data/Xband/" + t + ".png" if X else None,
						no_plot = True,
						classifierThreshold = classifierThreshold,
						use_large_classifier = use_large_classifier
						)

				trueresult = int(t) > 31 # hackish way to get the result since all uuids from 0000 to 0031 are noSeas => no wake

				roundedtestresult = testresult > classifierThreshold
				if trueresult == True and roundedtestresult == trueresult:
					# true positive
					tp += 1
				elif trueresult == True and roundedtestresult != trueresult:
					# false negative
					fn += 1
				elif trueresult == False and roundedtestresult == trueresult:
					# true negative
					tn += 1
				else:
					# false positive
					fp += 1
				conf.append(str(testresult))
				# end tth test

			statswriter.writerow({"iteration": str(i), "tp": str(tp), "fp": str(fp), "tn": str(tn), "fn": str(fn), "conf": conf})

			bandtp += tp
			bandtn += tn
			bandfp += fp
			bandfn += fn

			# delete
			os.remove("./current_data/Cband/results.csv")
			os.remove("./current_data/Sband/results.csv")
			os.remove("./current_data/Xband/results.csv")
			# end ith iteration
		precision = float(bandtp) / (float(bandtp + bandfp) + 1e-10)
		recall = float(bandtp) / (float(bandtp + bandfn) + 1e-10)
		F1 = 2.0 * (precision * recall) / (precision + recall + 1e-10)
		statswriter.writerow({"iteration": str(-1), "tp": str(bandtp), "fp": str(bandfp), "tn": str(bandtn), "fn": str(bandfn), "conf": [str(precision), str(recall), str(F1)]})
		statisticsFile.close()
		# end bth band


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

	#bandCombos = ["C", "S", "X", "CSX"]
	#bandCombos = ["CSX"]
	bandCombos = ["C"]

	use_reduced_data = False

	p_train_val = 0.9
	p_test = 0.1
	
	#n_epochs = 15
	#n_iter = 100
	# testing values
	n_epochs = 15
	n_iter = 3

	use_large_classifier = True 
	#use_large_classifier = False
	classifierThreshold = 0.6

	main(data_wake, data_nowake, bandCombos, p_train_val=p_train_val,
		p_test=p_test, n_epochs=n_epochs, n_iter=n_iter,
		use_large_classifier=use_large_classifier,
		classifierThreshold=classifierThreshold)

