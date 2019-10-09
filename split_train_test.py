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

def split_train_test(directory, cv):
	train_data_dict = {}
	test_data_dict = {}

	# get all csv's in the input directory
	files_in_folder = glob.glob(os.path.join(directory, "*.csv"))

	train_data_dict = dict()
	test_data_dict = dict()
	# keep track of the order in which users' data is added, 
	# and how many rows of test data each user adds
	user_test_dict = OrderedDict()

	for f in files_in_folder:
		userID = f[-15:-9]
		train_data_dict[f] = pd.read_csv(f).iloc[0:71*cv,:]
		test_data_dict[f] = pd.read_csv(f).iloc[71*cv:,:]
		user_test_dict[userID] = test_data_dict[f].shape[0]

	train_data = pd.concat(train_data_dict[f] for f in files_in_folder)
	test_data  = pd.concat(test_data_dict[f] for f in files_in_folder)

	# convert mood variable and mood lags from ordinal measure to binary measure
	train_data = get_binary_mood(train_data)
	test_data = get_binary_mood(test_data)

	train_covariates = train_data.drop('mood', axis=1)
	train_labels = np.ravel(train_data.mood)
	test_covariates = test_data.drop('mood', axis=1)
	test_labels = np.ravel(test_data.mood)


	return train_covariates, train_labels, test_covariates, test_labels
