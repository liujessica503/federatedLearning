'''split data into training and test data based on which cross-validation block we're using
Maintenance:
10/01/19 Created
'''

import pandas as pd
import numpy as np
import glob
import os
from collections import OrderedDict

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

	return train_data, test_data
