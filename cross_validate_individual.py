'''

This script does regular 3-fold cross-validation within the first 71*(cv) days
of each individual's data, where cv is set by init.json.
We will use the subsequent 71 days as the test set.

Outputs an unordered dictionary where the key is the individual user
and the value is a list of length k of loss on k validation sets

# commented out code to preserve ordering while spliting our data
# into chunks and evaluating on each chunk

Maintenance:
10/22/19 Created

'''

# cut the row indices into partial_train/ validate
# user_day_data_fold = UserDayData(user_day_data.X.iloc[the_rows_we_want], )

import json
import sys
import numpy as np
import csv
import pickle
# standardize the data
from sklearn.preprocessing import StandardScaler
# import user-defined functions
from IndividualModel import IndividualModel
from split_train_test_global import split_train_test_global
from UserDayData import UserDayData
import keras
from keras import optimizers
# to clear model after use
from keras import backend as K

k = 3
# split our 142 days of training data into k partitions
num_val_samples = (71 * 2) // k

# dictionary of dictionaries. Key is user and value is user_loss_by_lr
user_loss = {}


# load user-inputted parameters
with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

epochs = parameter_dict['epochs']
batch_size = parameter_dict['batch_size']
verbose = parameter_dict['verbose']
# write cv results as we process each one
output_path = parameter_dict['output_path']
# output python dictionary so that we can read it in easily
cv_dict_file = parameter_dict['cv_dict_file']

(
    train_covariates,
    train_labels,
    train_user_day_pairs,
    test_covariates,
    test_labels,
    test_user_day_pairs
) = split_train_test_global(
    directory=parameter_dict['input_directory'],
    cv=parameter_dict['cv']
)

train_data = UserDayData(train_covariates, train_labels, train_user_day_pairs)
test_data = UserDayData(test_covariates, test_labels, test_user_day_pairs)

# this is also used in the IndividualModel train method
user_list = np.unique([x[0] for x in train_data.user_day_pairs])

for user in user_list:

    # dictionary, key is learning rate and value is val_loss_list
    user_loss_by_lr = {}

    for curr_lr in np.arange(0.15, 0.25, 0.01):
        # instantiate model with parameters from json file
        # AND custom learn rate
        model = IndividualModel(
            parameter_config=parameter_dict,
            parameter_overwrite={"lr": curr_lr}
        )
        # similar to IndividualModel train method
        # user_model = copy.deepcopy(model.template_model)
        # deep copy-ing model doesn't work in flux
        # so we clone, load weights, and compile instead
        user_model = keras.models.clone_model(model.template_model)
        user_model.set_weights(model.template_model.get_weights())
        user_model.compile(loss=parameter_dict['loss'],
                optimizer = optimizers.Adam(
                    lr=curr_lr, 
                    beta_1=0.9, 
                    beta_2=0.999,
                    decay=0.0, 
                    amsgrad=False
                ),
                metrics=['accuracy'],
            )   
        #user_model.set_weights(model.get_weights())
        # k-fold cross-validation on an individual's data

        # list of length k of loss on validation set for this individual
        val_loss_list = []
        for i in range(k):
            print('processing fold #', i, ' for user ', user)

            # get the indices of our training data
            # and subset it into indices for validation and training
            train_indices = train_data._get_rows_for_users([user])
            val_indices = train_indices[i * num_val_samples:(i + 1) * num_val_samples]
            partial_train_indices = train_indices[:i * num_val_samples]+ train_indices[(i + 1)*num_val_samples:]

            # extract data corresponding to our indices
            val_data = UserDayData(train_data.X.iloc[val_indices], train_data.y[val_indices], [train_data.user_day_pairs[i] for idx in val_indices])
            partial_train_data = UserDayData(train_data.X.iloc[partial_train_indices], train_data.y[partial_train_indices], [train_data.user_day_pairs[i] for idx in partial_train_indices])


            user_scaler = StandardScaler().fit(partial_train_data.X)
            partial_train_data.X = user_scaler.transform(partial_train_data.X)
            modelFit = user_model.fit(partial_train_data.X, partial_train_data.y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data = (val_data.X, val_data.y))
            # get the loss for the last epoch
            val_loss = modelFit.history['val_loss'][epochs-1]
            val_loss_list.append(val_loss)

        # each time we finish processing k-folds, write the validation results
        with open(output_path, 'a') as write_to_file:
            file_writer = csv.writer(write_to_file, delimiter=',')
            file_writer.writerow([user, curr_lr, val_loss_list])

        # add user and list of validation losses to dictionary
        user_loss_by_lr[curr_lr] = val_loss_list

        # delete model
        K.clear_session()

    user_loss[user] = user_loss_by_lr

    # save the user_loss dictionary
    with open(cv_dict_file, 'wb') as dict_file:
        pickle.dump(user_loss, dict_file)

'''
# code derived from Chollet 'Deep Learning with Python'
'''
