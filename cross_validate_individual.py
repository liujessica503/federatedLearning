'''

This script does regular 3-fold cross-validation within the first 71*(cv) days
of each individual's data, where cv is set by init.json.
We will use the subsequent 71 days as the test set.

Outputs an unordered dictionary where the key is the individual user 
and the value is a list of length k of loss on k validation sets

Runs the following files:
init.json (take in parameters and then run:)
split_train_test_individual.py
IndividualModel.py (calls get_binary_mood.py)

# commented out code to preserve ordering while spliting our data into chunks and evaluating on each chunk

Maintenance:
10/22/19 Created

'''

import json
import sys
import pandas as pd
import numpy as np
import glob
import os
# import user-defined functions
from IndividualModel import IndividualModel
from split_train_test_individual import split_train_test_individual


k = 3
# split our 142 days of training data into k partitions
num_val_samples = (71*2) // k

# dictionary, key is user and value is val_loss_list
user_loss = {}

# load user-inputted parameters
with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

epochs = parameter_dict['epochs']

# code derived from Chollet 'Deep Learning with Python'

for file in os.listdir(parameter_dict['input_directory']):
    filename = os.fsdecode(file)
    user = filename[5:11]
    if filename.lower().endswith(".csv"):
        full_filepath = os.path.join(parameter_dict['input_directory'], filename)


        # k-fold cross-validation on an individual's data

        # list of length k of loss on validation set for this individual
        val_loss_list = []
        for i in range(k):
            print('processing fold #', i)

            # get all of our training data that we will then partition into k-folds
            train_covariates, train_labels, test_covariates, test_labels = split_train_test_individual(
                    file = full_filepath, 
                    cv = parameter_dict['cv'])
            # go to next file if we have no test data
            if train_covariates is not None:
                val_covariates = train_covariates[i * num_val_samples: (i + 1) * num_val_samples]
                val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
                partial_train_covariates = np.concatenate( [train_covariates[:i * num_val_samples],
                    train_covariates[(i + 1) * num_val_samples:]], axis=0)
                partial_train_labels = np.concatenate( [train_labels[:i * num_val_samples],
                    train_labels[(i + 1) * num_val_samples:]], axis=0)
                # build the keras model (already compiled)
                # model = build_model()
                # compile model
                model = IndividualModel(parameter_config = parameter_dict)
                modelFit = model.train(partial_train_covariates, partial_train_labels, validation_data = (val_covariates, val_labels))
                # get the loss for the last epoch
                val_loss = modelFit.history['val_loss'][epochs-1]
                val_loss_list.append(val_loss)
        # add user and list of validation losses to dictionary
        user_loss[user] = val_loss_list



'''
# train on the first 365/5 days, test on the next 365/5 days
# train on the first 365*2/5 days. test on the next 365/5 days
# ... train on the first 365*4/5 days, test on the next (last) 365/5 days

k = 3
val_loss_list = []
# split our 142 days of training data into k partitions
num_val_samples = (71*2) // k

# load user-inputted parameters
with open('init.json') as file:
        parameter_dict = json.load(file)

epochs = parameter_dict['epochs']

# code derived from Chollet 'Deep Learning with Python'
for i in range(k):
    print('processing fold #', i)
    # restrict to k-fold cv on 0:71*(2/3), 0:71*(4/3), 0:71*(6/3) (only do cv on the first 142 days of data)
    train_covariates, train_labels, test_covariates, test_labels = split_train_test_individual(
        file = full_filepath, trainRowStart = 0, trainRowEnd = i * num_val_samples, 
        testRowStart = i * num_val_samples, testRowEnd = (i + 1) * num_val_samples)
    # this below line may be unnecessary if we run this script within single_individual_experiment.py 
    # because the latter already checks if train_covariates is not None
    if train_covariates is not None:
        # we use the same methods as global model
        model = IndividualModel(parameter_config = parameter_dict)
        model.train(partial_train_covariates, partial_train_labels, validation_data = (val_covariates, val_labels))
        val_loss = history.history['val_loss'][epochs-1]
        val_loss_list.append(val_loss)
            
    else:
        continue
'''