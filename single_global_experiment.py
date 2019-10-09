'''Runs the following files:
init.json (take in parameters and then run:)
split_train_test.py
globalmodel.py (calls get_binary_mood.py)

Maintenance:
10/01/2019 Created

To-do: in split_train_test, need to add a function for split_train_valid_test
Need to check if we have enough test / valid data before splitting.
'''


# take in cv

# later add graphics from analyzing results

import json

# import user-defined functions
from split_train_test_global import split_train_test_global
from get_binary_mood import get_binary_mood
from GlobalModel import GlobalModel

import datetime
start = datetime.datetime.now()

with open('init.json') as file:
		parameter_dict = json.load(file)

train_covariates, train_labels, test_covariates, test_labels = split_train_test_global(
	directory = parameter_dict['input_directory'], 
	cv = parameter_dict['cv'])

global_model = GlobalModel(parameter_config = parameter_dict)
global_model.train(X_dict = train_covariates, Y_dict = train_labels)
predictions = global_model.predict(X_dict = test_covariates)

finish = datetime.datetime.now() - start
print(finish.total_seconds())
