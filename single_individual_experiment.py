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

import csv

# import user-defined functions
from split_train_test_global import split_train_test_global

from IndividualModel import IndividualModel

from UserDayData import UserDayData

import datetime
start = datetime.datetime.now()

with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

counter = 0

(
    train_covariates,
    train_labels,
    train_user_day_pairs,
    test_covariates,
    test_labels,
    test_user_day_pairs
) = split_train_test_global(
    directory=parameter_dict['input_directory'],
    cv=parameter_dict['cv'],
)

train_data = UserDayData(
    X=train_covariates, y=train_labels, user_day_pairs=train_user_day_pairs
)
test_data = UserDayData(
    X=test_covariates, y=test_labels, user_day_pairs=test_user_day_pairs
)
individual_model = IndividualModel(parameter_config=parameter_dict)
# the train method iterates over each individual and
# trains a model for each individual
individual_model.train(train_data)
# individual_model.model_dict to see trained models
predictions = individual_model.predict(test_data)
metrics_dict = individual_model.individual_evaluate(test_data, plotAUC=False)
import pdb; pdb.set_trace()

# write metrics to csv

# for the csv columns: get the names of our metrics
# by getting the values of the first dictionary entry
first_metrics_dict_entry = next(iter(metrics_dict.items()))
# since we have a dictionary within each value, now we want the keys
csv_columns = list(first_metrics_dict_entry[1].keys())

with open(parameter_dict['output_path'], 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for user in metrics_dict:
        user_metrics = metrics_dict[user]
        writer.writerow(user_metrics)


print('results written to: ' + parameter_dict['output_path'])

finish = datetime.datetime.now() - start
print('Time to finish: ' + str(finish.total_seconds()))
