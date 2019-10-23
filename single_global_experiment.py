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
import csv

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
metrics = global_model.evaluate(test_data, predictions = predictions, plotAUC = True)
print(metrics)

# write metrics to csv
csv_columns = list(metrics.keys())
with open(parameter_dict['output_path'], 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerow(metrics)

print('results written to :' + parameter_dict['output_path'])
finish = datetime.datetime.now() - start
print('Time to finish: ' + str(finish.total_seconds()))
