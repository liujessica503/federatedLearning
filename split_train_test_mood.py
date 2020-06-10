'''
use only mood from t-1 to t-7 as covariates 
to predict mood at time t
'''

import pandas as pd
import numpy as np
import glob
import os
# import user-defined function
from get_mood_class import get_mood_class


def split_train_test_mood(
    directory, cv, prediction_classes, loss_type
):
    train_data_dict = {}
    train_pairs_dict = {}

    test_data_dict = {}
    test_pairs_dict = {}

    # get all csv's in the input directory
    files_in_folder = glob.glob(os.path.join(directory, "*.csv"))

    for f in files_in_folder:
        userID = f[-15:-9]


        data = pd.read_csv(f)
        # get_mood_class.py will drop index column
        data = data[['Index', 'mood','mood.1','mood.2', 'mood.3', 'mood.4', 'mood.5', 'mood.6', 'mood.7']]

        train_data_dict[f] = data.iloc[0:71 * cv, :]
        train_days = data.iloc[0:71 * cv, 0]
        train_pairs_dict[f] = [(int(userID), int(x[:4])) for x in train_days]

        test_data_dict[f] = data.iloc[71 * cv:, :]
        test_days = data.iloc[71 * cv:, 0]
        test_pairs_dict[f] = [(int(userID), int(x[:4])) for x in test_days]

    train_data = pd.concat(train_data_dict[f] for f in files_in_folder)
    train_pairs = [train_pairs_dict[f] for f in files_in_folder]
    train_user_day_pairs = [
        item for sublist in train_pairs for item in sublist
    ]

    test_data = pd.concat(test_data_dict[f] for f in files_in_folder)
    test_pairs = [test_pairs_dict[f] for f in files_in_folder]
    test_user_day_pairs = [item for sublist in test_pairs for item in sublist]

    train_data = get_mood_class(train_data, prediction_classes, loss_type)
    test_data = get_mood_class(test_data, prediction_classes, loss_type)

    train_covariates = train_data.drop('mood', axis=1)
    train_labels = np.ravel(train_data.mood)
    test_covariates = test_data.drop('mood', axis=1)
    test_labels = np.ravel(test_data.mood)

    return (
        train_covariates,
        train_labels,
        train_user_day_pairs,
        test_covariates,
        test_labels,
        test_user_day_pairs,
    )
