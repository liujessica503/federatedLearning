#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:53:42 2019

@author: jessica

We combine all users into one global model and compare the neural network classification performance to that of training individual models.
Global model is defined as: get all of user 1's training data in the form of (days x predictors), i.e. (71*2 x 126) for user 1 for CV2. 
Then append user 's (71*2 x 126) training data. Our test set is: combine user 1's available data out of the next 71 days, user 2's available data out of the next 71 days, etc.

"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy import stats
# standardize the data
from sklearn.preprocessing import StandardScaler
# neural nets
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# https://keras.io/optimizers/
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import r2_score
# for binary classification
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# to write to csv
import csv
# for grabbing all of .csv extension
import glob

############################ GLOBAL MODEL ##############

directory = 'Downloads/IHS/4_11_19AdditionalCVBlocks/individual_users_neuralNet/cv2/'
subdirectory = ['cv2_split1','cv2_split2','cv2_split3','cv2_split4','cv2_split5']

# we want to get combine all individuals' training data into a single data frame
# that is, each individuals' first 71*cv days of training data
# about 15 seconds
global_train_dictionary = {}
for i in subdirectory:
    folder = directory + i
    files_in_folder = glob.glob(os.path.join(folder, "*.csv"))
    global_train_dictionary[i] = pd.concat((pd.read_csv(f).iloc[0:71*cv,:] for f in files_in_folder))

train_frames = [global_train_dictionary['cv2_split1'],global_train_dictionary['cv2_split2'],global_train_dictionary['cv2_split3'],global_train_dictionary['cv2_split4'],global_train_dictionary['cv2_split5']]
global_train = pd.concat(train_frames)   
global_train.shape # (72420,128), where 72420/(71*2) = 510 users
list(global_train.columns)

# get test data (that is, combine all individual's data after the first 71*cv days)
# about 15 seconds
global_test_dictionary = {}
for i in subdirectory:
    folder = directory + i
    files_in_folder = glob.glob(os.path.join(folder, "*.csv"))
    global_test_dictionary[i] = pd.concat((pd.read_csv(f).iloc[71*cv:,:] for f in files_in_folder))
test_frames = [global_test_dictionary['cv2_split1'],global_test_dictionary['cv2_split2'],global_test_dictionary['cv2_split3'],global_test_dictionary['cv2_split4'],global_test_dictionary['cv2_split5']]
global_test = pd.concat(test_frames)   
global_test.shape # (12969,128), and we know there are 389 people who have more than 71*2 days of data
list(global_test.columns)

# if not using function to define train_data and test_data, use these
train_data = global_train.copy()
test_data = global_test.copy()
#####

# helper function to get binary mood
def global_binary_mood(data):
    # drop Index column because Python has its own index
    data = data.drop('Index', axis=1) 
       
    # replacing . with _ in names so that we can reference the columns in python
    data = data.rename(columns={"mood.1": "mood_1", "mood.2": "mood_2", "mood.3": "mood_3", "mood.4":"mood_4", "mood.5":"mood_5", "mood.6":"mood_6", "mood.7":"mood_7"})
    data.mood.value_counts()
    
    # using universal median mood instead of each user's median mood
    data.loc[data.mood < 7, 'mood'] = 0 
    data.loc[data.mood >= 7, 'mood'] = 1 
    data.loc[data.mood_1 < 7, 'mood_1'] = 0 
    data.loc[data.mood_1 >= 7, 'mood_1'] = 1 
    data.loc[data.mood_2 < 7, 'mood_2'] = 0 
    data.loc[data.mood_2 >= 7, 'mood_2'] = 1 
    data.loc[data.mood_3 < 7, 'mood_3'] = 0 
    data.loc[data.mood_3 >= 7, 'mood_3'] = 1 
    data.loc[data.mood_4 < 7, 'mood_4'] = 0 
    data.loc[data.mood_4 >= 7, 'mood_4'] = 1 
    data.loc[data.mood_5 < 7, 'mood_5'] = 0 
    data.loc[data.mood_5 >= 7, 'mood_5'] = 1 
    data.loc[data.mood_6 < 7, 'mood_6'] = 0 
    data.loc[data.mood_6 >= 7, 'mood_6'] = 1 
    data.loc[data.mood_7 < 7, 'mood_7'] = 0 
    data.loc[data.mood_7 >= 7, 'mood_7'] = 1 
    
    return data



def global_train_binary_mood(train = train_data, test = test_data, hidden_units, input_dim, saveAs):
    
    train_data = global_binary_mood(train_data)
    test_data = global_binary_mood(test_data)

    # specify the data
    # everything except mood variable
    X_train = train_data.drop('mood', axis=1)
    # isolate target labels
    Y_train = np.ravel(train_data.mood)
    
    # test data and test labels
    X_test = test_data.drop('mood', axis=1)
    Y_test = np.ravel(test_data.mood)

    X_train.shape # (72420, 126)
    X_test.shape # (12969, 126)
    Y_train.shape
    Y_test.shape
  
    # check distribution of mood in training set 
    # to make sure each class is balanced
    # In global Y_train, 21638 0's and 50782 1's
    # np.unique(Y_train, return_counts = True) 

    # Define the scaler 
    scaler = StandardScaler().fit(X_train)
    
    # Scale the train set
    X_train = scaler.transform(X_train)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    # Initialize the constructor
    # linear stack of layers
    model = Sequential()
    
    
    # Add input layer with 12 hidden units and relu activiation function
    # we have 126 columns in the input dimensions 
    model.add(Dense(hidden_units, input_dim=input_dim, activation='relu'))
    
    # Add two hidden layers
    # this is different architecture from individual model
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    
    # Add an output layer 
    # sigmoid activation function so that your output is a probability
    # This means that this will result in a score between 0 and 1, 
    # indicating how likely the sample is to have the target “1”, or how likely the wine is to be red.
    model.add(Dense(1, activation='sigmoid'))

#    # Model output shape
#    model.output_shape
#    
#    # Model summary
#    model.summary()
#    
#    # Model config
#    model.get_config()
#    
#    # List all weight tensors 
#    model.get_weights()
    
    # compile and fit model
    # alternative optimizers: stochastic gradient descent (SGD), RMSprop
    # binary_crossentropy loss function because we are doing binary classification on red or white wine
    # for multi-class classification you can use categorical_crossentropy loss
    # verbose=1 shows progress bar
    model.compile(loss='binary_crossentropy',
                  optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])
                       
    # batch size is number of samples propagated through network
    # model takes about 3 minutes to run with 8 epochs and batch_size = 40
    model.fit(X_train, Y_train,epochs=8, batch_size=40, verbose=1)
    
    # predict test set labels
    # had to install nomkl in anaconda for this to run
    # this outputs a probability of being in class 1
    Y_pred = model.predict(X_test).ravel()
    
    # in order to calculate AUC, we need both classes 1 and 0 represented in the data
    if 1 in Y_test and 0 in Y_test:
        # false and true positive rates and thresholds
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
        auc_value = auc(fpr, tpr)
        print('AUC: ' + str(auc))
    else:
        fpr, tpr, thresholds, auc_value = ["","","",""] #initialize since we're printing to the csv

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    
    # since the class labels are binary, choose a probability cutoff to make the predictions binary
    Y_pred = Y_pred > 0.50
    # evaluate performance
    score = model.evaluate(X_test, Y_test,verbose=1)
    # loss and accuracy -- but our training data is imbalanced so this is not useful
    # 2x2 Confusion matrix
    # confusion_matrix(Y_test, Y_pred)
    # Precision 
    precision = precision_score(Y_test, Y_pred)
    # Recall
    recall = recall_score(Y_test, Y_pred)
    # print('Precision: ' + str(precision) + ', Recall: ' + str(recall))
    # F1 score - weighted average of precision and recall
    f1 = f1_score(Y_test,Y_pred)
    # Cohen's kappa - classification accuracy normalized by the imbalance of the classes in the data
    cohen = cohen_kappa_score(Y_test, Y_pred)
    
    global_binary_result = {"Number of Test Obs": Y_test.shape[0], "FPR": fpr, "TPR": tpr, "AUC": auc_value, 'Score': score, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Cohen': cohen}
    
    return global_binary_result
