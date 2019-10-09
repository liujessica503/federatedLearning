'''split data into training and test data based on which cross-validation block we're using
Maintenance:
10/01/19 Created
'''

import pandas as pd
import numpy as np
import glob
import os
from collections import OrderedDict
# import user-defined function
from get_binary_mood import get_binary_mood

def split_train_test_global(directory, cv):
    train_data_dict = {}
    train_pairs_dict = {}

    test_data_dict = {}
    test_pairs_dict = {}

    # get all csv's in the input directory
    files_in_folder = glob.glob(os.path.join(directory, "*.csv"))

    for f in files_in_folder:
        userID = f[-15:-9]

        train_data_dict[f] = pd.read_csv(f).iloc[0:71*cv,:]
        train_days = pd.read_csv(f).iloc[0:71*cv,0]
        train_pairs_dict[f] = [(int(userID), int(x[:4])) for x in train_days]

        test_data_dict[f] = pd.read_csv(f).iloc[71*cv:,:]
        test_days = pd.read_csv(f).iloc[71*cv:,0]
        test_pairs_dict[f] = [(int(userID), int(x[:4])) for x in test_days]

    train_data = pd.concat(train_data_dict[f] for f in files_in_folder)
    train_pairs = [train_pairs_dict[f] for f in files_in_folder]
    train_user_day_pairs = [item for sublist in train_pairs for item in sublist]

    test_data  = pd.concat(test_data_dict[f] for f in files_in_folder)
    test_pairs = [test_pairs_dict[f] for f in files_in_folder]
    test_user_day_pairs = [item for sublist in test_pairs for item in sublist]

    # convert mood variable and mood lags from ordinal measure to binary measure
    train_data = get_binary_mood(train_data)
    test_data = get_binary_mood(test_data)

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
        test_pairs
    )
