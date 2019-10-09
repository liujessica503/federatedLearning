'''Runs the following files:
init.json (take in parameters and then run:)
split_train_test.py
individual_model.py (not yet created -- calls get_binary_mood.py)

Maintenance:
10/08/2019 Created

To-do: in split_train_test, need to add a function for split_train_valid_test
Need to check if we have enough test / valid data before splitting.
'''


# take in cv

# later add graphics from analyzing results

import json
import os

# import user-defined functions
from split_train_test_individual import split_train_test_individual
from get_binary_mood import get_binary_mood
from GlobalModel import GlobalModel

import datetime
start = datetime.datetime.now()

with open('init.json') as file:
		parameter_dict = json.load(file)

for file in os.listdir(parameter_dict['input_directory']):
	filename = os.fsdecode(file)
	if filename.lower().endswith(".csv"):
		full_filepath = os.path.join(parameter_dict['input_directory'], filename)
		# the below function returns none if we have no test data
		train_covariates, train_labels, test_covariates, test_labels = split_train_test_individual(
			file = full_filepath, 
			cv = parameter_dict['cv'])
		if train_covariates is not None:
			# we use the same methods as global model
			individual_model = GlobalModel(parameter_config = parameter_dict)
			individual_model.train(X_dict = train_covariates, Y_dict = train_labels)
			predictions = individual_model.predict(X_dict = test_covariates)
			
	else:
		# if not a .csv file
		continue

finish = datetime.datetime.now() - start
print('Time to finish: ' + str(finish.total_seconds()))
