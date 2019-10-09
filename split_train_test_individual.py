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

def split_train_test_individual(file, cv):
	userID = file[-15:-9]	
	train_data = pd.read_csv(file).iloc[0:71*cv,:]
	test_data  = pd.read_csv(file).iloc[71*cv:,:]
	if test_data.shape[0] == 0:
		print('User ' + userID + ' has no test data. Skipping this user')
		return None, None, None, None
	else:
		# run neural net
		print('running neural net on ' + 'User ' + userID)

		# convert mood variable and mood lags from ordinal measure to binary measure
		train_data = get_binary_mood(train_data)
		test_data = get_binary_mood(test_data)

		train_covariates = train_data.drop('mood', axis=1)
		train_labels = np.ravel(train_data.mood)
		test_covariates = test_data.drop('mood', axis=1)
		test_labels = np.ravel(test_data.mood)
		return train_covariates, train_labels, test_covariates, test_labels
