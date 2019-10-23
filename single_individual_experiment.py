'''
Runs the following files:
init.json (take in parameters and then run:)
split_train_test_individual.py
IndividualModel.py (calls get_binary_mood.py)

Maintenance:
10/08/2019 Created

To-do: in split_train_test, need to add a function for split_train_valid_test
Need to check if we have enough test / valid data before splitting.
'''


# take in cv

# later add graphics from analyzing results

import json
import sys
import os
import csv

# import user-defined functions
from split_train_test_individual import split_train_test_individual
from get_binary_mood import get_binary_mood
from IndividualModel import IndividualModel

import datetime
start = datetime.datetime.now()

with open('init.json') as file:
		parameter_dict = json.load(file)

counter = 0

for file in os.listdir(parameter_dict['input_directory']):
	filename = os.fsdecode(file)
	user = filename[5:11]
	if filename.lower().endswith(".csv"):
		full_filepath = os.path.join(parameter_dict['input_directory'], filename)
		train_covariates, train_labels, test_covariates, test_labels = split_train_test_individual(
			file = full_filepath, 
			cv = parameter_dict['cv'])
		# go to next file if we have no test data
		if train_covariates is not None:
			# we use the same methods as global model
			individual_model = IndividualModel(parameter_config = parameter_dict)
			individual_model.train(X_dict = train_covariates, Y_dict = train_labels)
			predictions = individual_model.predict(X_dict = test_covariates)
			print(predictions)
			metrics = individual_model.evaluate(test_covariates = test_covariates, 
				test_labels = test_labels, predictions = predictions, plotAUC = True)
			print(metrics)
			metrics['user'] = user

			# when running this for the first time, create new file and write header to csv
			if counter == 0:
				# write metrics to csv
				csv_columns = list(metrics.keys())
				with open(parameter_dict['output_path'], 'w') as csvfile:
				    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
				    writer.writeheader()
				    writer.writerow(metrics)
				counter += 1

			# append rest of individuals
			if counter > 0:
				# write metrics to csv
				csv_columns = list(metrics.keys())
				with open(parameter_dict['output_path'], 'a') as csvfile:
				    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
				    writer.writerow(metrics)
			
	else:
		# if not a .csv file
		continue


print('results written to :' + parameter_dict['output_path'])

finish = datetime.datetime.now() - start
print('Time to finish: ' + str(finish.total_seconds()))
