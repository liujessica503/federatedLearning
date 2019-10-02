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
from split_train_test import split_train_test
from get_binary_mood import get_binary_mood
from GlobalModel import GlobalModel

with open('init.json') as file:
		parameter_dict = json.load(file)

train_data, test_data = split_train_test(directory = parameter_dict['Data']['input_directory'], cv = parameter_dict['Data']['cv'])


global_model = GlobalModel(parameter_dict = parameter_dict, train_data = train_data, test_data = test_data)
global_model.train(n_epochs = parameter_dict['ModelFitParams']['n_epochs'], batch_size = parameter_dict['ModelFitParams']['batch_size'], verbose = parameter_dict['ModelFitParams']['verbose'], hidden_units_args = parameter_dict['NeuralNetStructure']['hidden_units'], input_dim = parameter_dict['NeuralNetStructure']['input_dim'], activation = parameter_dict['NeuralNetStructure']['activation'])
global_model.test(test_covariates = self.X_test, test_labels = self.Y_test)
