'''split data into training and test data based on which cross-validation block we're using
Maintenance:
10/01/19 Created
'''

import pandas as pd
import numpy as np
import glob
import os

def split_train_test(directory, cv):
	train_data_dict = {}
	test_data_dict = {}

	# get all csv's in the input directory
	files_in_folder = glob.glob(os.path.join(directory, "*.csv"))
	train_data = pd.concat((pd.read_csv(f).iloc[0:71*cv,:] for f in files_in_folder))
	test_data = pd.concat((pd.read_csv(f).iloc[71*cv:,:] for f in files_in_folder))

	# this is inefficient because we are reading test data above and below
	# I know list comprehension above is faster and preferable, I just can't think of a way to make the below happen at the same time as the above
	user_test_dict = {}
	for f in files_in_folder:
		userID = f[-15:-9]
		read_test_data = pd.read_csv(f).iloc[71*cv:,:]
		user_test_dict[userID] = read_test_data.shape[0]

	return train_data, test_data