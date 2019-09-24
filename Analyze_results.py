#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:23:53 2019

@author: jessica
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy import stats
# to write to csv
import csv

os.getcwd()
os.chdir('Downloads/IHS/4_11_19AdditionalCVBlocks/individual_users_neuralNet/cv2/cv2_ADAM_and_AUC_Results/')
# missingness in training blocks (missingness in day 1 to day 71*cv)
missing_mood = pd.read_csv('../../../../mood_na_cv3.csv')

# set which cross-validation block we're looking at 
cv = 2
'''
Read in results generated from 4_11_19AdditionalCVBlocks/individual_users_neuralNet/binary_classification_scripts/TensorFlow_Binary_split1.py
and split2, etc to split5 for given CV split.
'''

def rename_columns(filename):
    data = pd.read_csv(filename, sep=',')
    # column names are from TensorFlow_Binary_split1.py
    data.columns = ['User', 'Number of Test Obs', 'FPR', 'TPR', 'AUC', 'Score','Precision', 'Recall', 'F1', 'Cohen']
    return data

split1 = rename_columns('results_binary_split1.csv')
split2 = rename_columns('results_binary_split2.csv')
split3 = rename_columns('results_binary_split3.csv')
split4 = rename_columns('results_binary_split4.csv')
split5 = rename_columns('results_binary_split5.csv')

frames_to_combine = [split1, split2, split3, split4, split5]

results = pd.concat(frames_to_combine, ignore_index = True)

# extract user ID from user column
results['UserID'] = results['User'].str[5:11]

results.shape
plt.scatter(results['Number of Test Obs'], results['AUC'], alpha = 0.4)
MoreThan10TestObs = results[results['Number of Test Obs'] >= 10]
MoreThan10TestObs['AUC'].describe()
plt.hist(MoreThan10TestObs['AUC'], density = True, bins = 10)


'''
CV2: we have 389 users, which is less than 510, because if we had no test data for the 3rd block (used imputed data on first two blocks to train) 
we skipped this user.

CV3: we have 356 users, for the same reason.

'''

missing_mood.columns
# rename columns to more descriptive names
missing_mood.columns = ['Index','UserID','NumberOfDaysMissingMoodInCv']
# in order to merge two data frames, need to make the columns the same data type
missing_mood['UserID'] = missing_mood['UserID'].astype(str)

results_with_missing = pd.merge(results, missing_mood, how='left', on='UserID')
results_with_missing = results_with_missing.sort_values(by=['NumberOfDaysMissingMoodInCv'], ascending = True)





high_missing = results_with_missing[results_with_missing['NumberOfDaysMissingMoodInCv'] >= 71*cv*.4]
low_missing = results_with_missing[results_with_missing['NumberOfDaysMissingMoodInCv'] < 71*cv*.4]
high_missing.shape # 152 for >= 71*2*.5 for CV2. 91 for >= 71*2*.4 for CV3
low_missing.shape # 237 for < 71*2*.5 for CV2. 
high_missing.Precision.describe()
low_missing.Precision.describe()
plt.hist(low_missing.Precision, density = True, alpha = 0.5, color = 'red', bins = 20)
plt.hist(high_missing.Precision, density = True, alpha = 0.5, color = 'green', bins = 20)
plt.show()
high_missing.Recall.describe()
low_missing.Recall.describe()
plt.hist(high_missing.Recall, density = True, alpha = 0.5, color = 'green', bins = 20)
plt.hist(low_missing.Recall, density = True, alpha = 0.5, color = 'red', bins = 20)
plt.show()
# F1 score - weighted average of precision and recall
high_missing.F1.describe()
low_missing.F1.describe()
plt.hist(high_missing.F1, density = True, alpha = 0.5, color = 'green', bins = 20)
plt.hist(low_missing.F1, density = True, alpha = 0.5, color = 'red', bins = 20)
plt.show()