'''Runs the following files:
init.json (take in parameters and then run:)
split_train_test.py (calls get_binary_mood.py)
globalmodel.py (calls plot_auc.py)

Maintenance:
10/01/2019 Created

To-do: in split_train_test, need to add a function for split_train_valid_test
Need to check if we have enough test / valid data before splitting.
'''


# take in cv

# later add graphics from analyzing results

import json
import sys

# import user-defined functions
from split_train_test_global import split_train_test_global
from GlobalModel import GlobalModel
from UserDayData import UserDayData

import datetime
start = datetime.datetime.now()

with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

train_covariates, train_labels, train_user_day_pairs, test_covariates, test_labels, test_user_day_pairs = split_train_test_global(
    directory = parameter_dict['input_directory'], 
    cv = parameter_dict['cv'])

train_data = UserDayData(train_covariates, train_labels, train_user_day_pairs)
test_data = UserDayData(test_covariates, test_labels, test_user_day_pairs)

global_model = GlobalModel(parameter_config = parameter_dict)
global_model.train(train_data)
predictions = global_model.predict(test_data)
global_model.evaluate(test_data, predictions = predictions, outputFile = parameter_dict['output_path'], plotAUC = True)
finish = datetime.datetime.now() - start
print(finish.total_seconds())
