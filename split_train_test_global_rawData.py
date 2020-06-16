'''split data into training and test data based on which cross-validation block
we're using
Maintenance:
10/01/19 Created
'''

import pandas as pd
import numpy as np
import glob
import pickle
import os
import re
from collections import OrderedDict
import gzip
# just doing 0-1 classification on WESAD data
# import user-defined function
# from get_mood_class import get_mood_class


def write_chest_measurements(data):
    """
    take the raw data dictionary and make a new dictionary, where
    key is the index of the i-th observation
    value is a list of measurements, including the label corresponding to baseline, stress, amusement
    """
    chest_measurements = OrderedDict()
    for idx in np.arange(0,len(data['signal']['chest']['ACC'])):
        #for key in data['signal']['chest'].keys():
        values = []
        for key in ['label','ECG', 'EMG', 'EDA', 'Temp', 'Resp','ACC']:
            if key == 'label':
                values.append(data[key][idx])
            elif key == 'ACC':
                # ACC values are a list of length 3 (axes x,y,z)
                values.append(data['signal']['chest'][key][idx][0])
                values.append(data['signal']['chest'][key][idx][1])
                values.append(data['signal']['chest'][key][idx][2])
            else:
                values.append(data['signal']['chest'][key][idx][0])

        chest_measurements[idx] = values
    return chest_measurements

def preprocess_measurements(directory):
    '''
    save data files in our desired format
    '''
    # raw data files
    list_of_files = ['S2/S2.pkl', 'S3/S3.pkl', 'S4/S4.pkl', 
    'S5/S5.pkl', 'S6/S6.pkl', 'S7/S7.pkl', 'S8/S8.pkl', 'S9/S9.pkl', 
    'S10/S10.pkl', 'S11/S11.pkl', 'S13/S13.pkl', 'S14/S14.pkl',
    'S15/S15.pkl','S16/S16.pkl','S17/S17.pkl']

    list_of_files = [str(directory + x) for x in list_of_files]

    for f in list_of_files:
        # get subject ID from files named like /path/S10/S10.pkl, where 10 is the subject ID
        userID = re.match(".*\/(S)([0-9]+)", f).group(2)

        with open(f, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')
        chest_measurements = write_chest_measurements(raw_data)

        data = pd.DataFrame.from_dict(chest_measurements, orient='index', columns=['Label','ECG', 'EMG', 'EDA', 'Temp', 'Resp', 'ACC_x', 'ACC_y', 'ACC_z'])

        data_array = data.to_numpy()

        f = gzip.GzipFile(directory + "/S" + str(userID) + "/" + "S" + str(userID) + "_chest_rawData.npy.gz", "w")
        np.save(file=f, arr=data_array)
        f.close()
        print("file for user " + userID + " written")

    return

def split_train_test_global(
    directory, cv=0, prediction_classes=0, loss_type='classification'
):
    train_data_dict = {}
    train_pairs_dict = {}

    test_data_dict = {}
    test_pairs_dict = {}

    list_of_files = ['S2/', 'S3/', 'S4/', 
    'S5/', 'S6/', 'S7/', 'S8/', 'S9/', 
    'S10/', 'S11/', 'S13/', 'S14/',
    'S15/','S16/','S17/']

    list_of_files = [str(directory + x) for x in list_of_files]

    for f in list_of_files:
        # get subject ID from files named like /path/S10/S10.pkl, where 10 is the subject ID
        userID = re.match(".*\/(S)([0-9]+)", f).group(2)

        dataFile = gzip.GzipFile(f + "S" + str(userID) + "_chest_rawData.npy.gz", "r")
        dataNp = np.load(dataFile)
        data = pd.DataFrame(dataNp, columns = ['Label','ECG', 'EMG', 'EDA', 'Temp', 'Resp', 'ACC_x', 'ACC_y', 'ACC_z'])

        # according to labels in wesad_readme.pdf, 1 = baseline, 2 = stress, 3 = amusement
        baseline_data = data.loc[data['Label'] == 1]
        stress_data = data.loc[data['Label'] == 2]
        amusement_data = data.loc[data['Label'] == 3]

        # first 2/3 is training data, rest is test data
        baseline_data_train = baseline_data.iloc[0: baseline_data.shape[0] * 2 //3, :]
        stress_data_train = stress_data.iloc[0: stress_data.shape[0] * 2 //3, :]
        amusement_data_train = amusement_data.iloc[0: amusement_data.shape[0] * 2 //3, :]

        baseline_data_test = baseline_data.iloc[baseline_data.shape[0] * 2 //3:, :]
        stress_data_test = stress_data.iloc[stress_data.shape[0] * 2 //3:, :]
        amusement_data_test = amusement_data.iloc[amusement_data.shape[0] * 2 //3:, :]

        # recode Label 0 = non-stress, 1 = stress
        train_data = pd.concat([baseline_data_train, stress_data_train, amusement_data_train])
        train_data.loc[train_data['Label'] == 1, 'Label'] = 0
        train_data.loc[train_data['Label'] == 2, 'Label'] = 1
        train_data.loc[train_data['Label'] == 3, 'Label'] = 1
        test_data = pd.concat([baseline_data_test, stress_data_test, amusement_data_test])
        test_data.loc[test_data['Label'] == 1, 'Label'] = 0
        test_data.loc[test_data['Label'] == 2, 'Label'] = 1
        test_data.loc[test_data['Label'] == 3, 'Label'] = 1

        train_data_dict[f] = train_data
        train_days = train_data_dict[f].index
        train_pairs_dict[f] = [(int(userID), int(x)) for x in train_days]

        test_data_dict[f] = test_data
        test_days = test_data_dict[f].index
        test_pairs_dict[f] = [(int(userID), int(x)) for x in test_days]

    train_data = pd.concat(train_data_dict[f] for f in list_of_files)
    train_pairs = [train_pairs_dict[f] for f in list_of_files]
    train_user_day_pairs = [
        item for sublist in train_pairs for item in sublist
    ]

    test_data = pd.concat(test_data_dict[f] for f in list_of_files)
    test_pairs = [test_pairs_dict[f] for f in list_of_files]
    test_user_day_pairs = [item for sublist in test_pairs for item in sublist]

    #train_data = get_mood_class(train_data, prediction_classes, loss_type)
    #test_data = get_mood_class(test_data, prediction_classes, loss_type)

    train_covariates = train_data.drop('Label', axis=1)
    train_labels = np.ravel(train_data.Label)
    test_covariates = test_data.drop('Label', axis=1)
    test_labels = np.ravel(test_data.Label)

    return (
        train_covariates,
        train_labels,
        train_user_day_pairs,
        test_covariates,
        test_labels,
        test_user_day_pairs,
    )

