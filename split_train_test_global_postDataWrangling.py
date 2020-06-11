'''split data into training and test data based on which cross-validation block
we're using
Maintenance:
10/01/19 Created
'''

import pandas as pd
import numpy as np
import glob
import os
import re
# just doing 0-1 classification on WESAD data
# import user-defined function
# from get_mood_class import get_mood_class


def split_train_test_global(
    directory, cv=0, prediction_classes=0, loss_type='classification'
):
    train_data_dict = {}
    train_pairs_dict = {}

    test_data_dict = {}
    test_pairs_dict = {}

    # get all csv's in the input directory
    files_in_folder = glob.glob(os.path.join(directory, "*.csv"))

    for f in files_in_folder:
        # get subject ID from files named like /path/to/S10_feats_4.csv, where 10 is the subject ID
        userID = re.match(".*\/(S)([0-9]+)", f).group(2)

        data = pd.read_csv(f)

        # according to labels in data_wrangling.py, around line 255
        baseline_data = data.loc[data['1'] == 1]
        stress_data = data.loc[data['2'] == 1]
        amusement_data = data.loc[data['0'] == 1]

        # first 2/3 is training data, rest is test data
        baseline_data_train = baseline_data.iloc[0: baseline_data.shape[0] * 2 //3, :]
        stress_data_train = stress_data.iloc[0: stress_data.shape[0] * 2 //3, :]
        amusement_data_train = amusement_data.iloc[0: amusement_data.shape[0] * 2 //3, :]

        baseline_data_test = baseline_data.iloc[baseline_data.shape[0] * 2 //3:, :]
        stress_data_test = stress_data.iloc[stress_data.shape[0] * 2 //3:, :]
        amusement_data_test = amusement_data.iloc[amusement_data.shape[0] * 2 //3:, :]

        train_data = pd.concat([baseline_data_train, stress_data_train, amusement_data_train])
        train_data['label'] = 0
        train_data.loc[train_data['2'] == 1, 'label'] = 1
        test_data = pd.concat([baseline_data_test, stress_data_test, amusement_data_test])
        test_data['label'] = 0
        test_data.loc[test_data['2'] == 1, 'label'] = 1

        train_data = train_data.drop(['Unnamed: 0','0','1','2'], axis = 1)
        test_data = test_data.drop(['Unnamed: 0','0','1','2'], axis = 1)

        train_data_dict[f] = train_data
        train_days = train_data_dict[f].index
        train_pairs_dict[f] = [(int(userID), int(x)) for x in train_days]

        test_data_dict[f] = test_data
        test_days = test_data_dict[f].index
        test_pairs_dict[f] = [(int(userID), int(x)) for x in test_days]

        '''
        train_data_dict[f] = pd.read_csv(f).iloc[0:71 * cv, :]
        train_days = pd.read_csv(f).iloc[0:71 * cv, 0]
        train_pairs_dict[f] = [(int(userID), int(x[:4])) for x in train_days]

        test_data_dict[f] = pd.read_csv(f).iloc[71 * cv:, :]
        test_days = pd.read_csv(f).iloc[71 * cv:, 0]
        test_pairs_dict[f] = [(int(userID), int(x[:4])) for x in test_days]
        '''

    train_data = pd.concat(train_data_dict[f] for f in files_in_folder)
    train_pairs = [train_pairs_dict[f] for f in files_in_folder]
    train_user_day_pairs = [
        item for sublist in train_pairs for item in sublist
    ]

    test_data = pd.concat(test_data_dict[f] for f in files_in_folder)
    test_pairs = [test_pairs_dict[f] for f in files_in_folder]
    test_user_day_pairs = [item for sublist in test_pairs for item in sublist]

    #train_data = get_mood_class(train_data, prediction_classes, loss_type)
    #test_data = get_mood_class(test_data, prediction_classes, loss_type)

    train_covariates = train_data.drop('label', axis=1)
    train_labels = np.ravel(train_data.label)
    test_covariates = test_data.drop('label', axis=1)
    test_labels = np.ravel(test_data.label)

    return (
        train_covariates,
        train_labels,
        train_user_day_pairs,
        test_covariates,
        test_labels,
        test_user_day_pairs,
    )
