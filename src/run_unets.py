#!python3

import train_unet
import train_class
import trainClassifier
import predictClassifier
import predict_class_unet


n_epochs = 30
p_train = 0.8
p_val = 0.2
#bce_weights = [0.3, 0.7, 0.9]
bce_weights = [0.7]
weight_decays = [0.01, 1e-5, 1e-8]
learning_rate = 0.0001

for bce_weight in bce_weights:
	for weight_decay in weight_decays:
		train_unet.train(n_epochs, p_train, p_val, C=False, S=True, X=False, n_batch=4, progress=True,
				bce_weight=bce_weight, weight_decay=weight_decay, learning_rate=learning_rate)

		train_unet.train(n_epochs, p_train, p_val, C=False, S=False, X=True, n_batch=4, progress=True,
				bce_weight=bce_weight, weight_decay=weight_decay, learning_rate=learning_rate)


