import pandas as pd
import numpy as np
import os
from scipy import stats
# standardize the data
from sklearn.preprocessing import StandardScaler
# neural nets
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# https://keras.io/optimizers/
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import r2_score
# for binary classification
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# to write to csv
import csv
from get_binary_mood import get_binary_mood


class GlobalModel:
	def __init__(self, parameter_dict, train_data, test_data=None):
		# convert ordinal mood to binary mood
		self.train_data = get_binary_mood(train_data)
		self.test_data = get_binary_mood(test_data)

		# X is everything except mood variable
		self.X_train = self.train_data.drop('mood', axis=1)
		# isolate mood labels
		self.Y_train = np.ravel(self.train_data.mood)
		# test data and test labels
		self.X_test = self.test_data.drop('mood', axis=1)
		self.Y_test = np.ravel(self.test_data.mood)

		# Scale the training and test sets
		scaler = StandardScaler().fit(self.X_train)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)

	def train(self, n_epochs, batch_size, hidden_units_args, input_dim, activation, verbose):
		# linear stack of layers
		model = Sequential()
		model.add(Dense(hidden_units=hidden_units_args[0], input_dim=input_dim, activation=activation))
		model.add(Dense(hidden_units=hidden_units_args[1], activation=activation))
		model.add(Dense(hidden_units=hidden_units_args[2], activation=activation))
		
		# Add an output layer 
		# sigmoid activation function so that your output is a probability
		# This means that this will result in a score between 0 and 1, 
		# indicating how likely the sample is to have the target “1”, or how likely the wine is to be red.
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss=loss,
				  optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
				  metrics=['accuracy'])
		model.fit(self.X_train, self.Y_train,epochs=n_epochs, batch_size=batch_size, verbose=verbose)


	def test(self):
			test_covariates = self.X_test
			test_labels = self.Y_test
			Y_pred = model.predict(self.X_test).ravel()
			# in order to calculate AUC, we need both classes 1 and 0 represented in the data
			if 1 in Y_test and 0 in Y_test:
				# false and true positive rates and thresholds
				fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
				auc_value = auc(fpr, tpr)
				print('AUC: ' + str(auc))
			else:
				fpr, tpr, thresholds, auc_value = ["","","",""] #initialize since we're printing to the csv

			# since the class labels are binary, choose a probability cutoff to make the predictions binary
			Y_pred = Y_pred > 0.50
			# evaluate performance
			score = model.evaluate(X_test, Y_test,verbose=1)
			precision = precision_score(Y_test, Y_pred)
			recall = recall_score(Y_test, Y_pred)
			# F1 score - weighted average of precision and recall
			f1 = f1_score(Y_test,Y_pred)
			# Cohen's kappa - classification accuracy normalized by the imbalance of the classes in the data
			cohen = cohen_kappa_score(Y_test, Y_pred)


	def write_metrics_to_csv(self, output_path, fpr=None, tpr=None, auc_value=None, score=None, precision=None, recall=None, f1=None, cohen=None):
		csv_columns = ['Number of Test Obs', 'FPR', 'TPR', 'AUC', 'Best Learning Rate By Validation Loss', 'Score','Precision', 'Recall', 'F1', 'Cohen']
		global_binary_metrics = {"Number of Test Obs": self.Y_test.shape[0], "FPR": fpr, "TPR": tpr, "AUC": auc_value, 'Score': score, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Cohen': cohen}
		with open(output_path, 'a') as csvfile:
						writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
						writer.writerow(global_binary_metrics)
