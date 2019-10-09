#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:25:09 2019

@author: jessica

    Binary classification of mood for one individual user 
    in 2017-2018 data
    
    Maintenance:
        9/17/2019: 
            1 if the user's predicted mood is above the median of observed mood across ALL users in training data, 0 if not. Previously, we were comparing to the median of EACH user's mood for each neural network.
            Added AUC curve plotting probabilities of being in mood class 1 or 0 from the neural network. Note that AUC is only reportable if the test set contains both 1 and 0 mood.
"""
####################### BINARY CLASSIFICATION ##################

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

#directory = '/Users/jessica/Downloads/IHS/4_11_19AdditionalCVBlocks/individual_users_neuralNet/cv2/cv2_split1'
#os.chdir('/Users/jessica/Downloads/IHS/4_11_19AdditionalCVBlocks/individual_users_neuralNet/cv2/')
directory = 'cv2_split1'

cv = 2

# initialize where we'll store results
current_result = {'User': None, "FPR": None, "TPR": None, "AUC": None, 'Score': None, 'Precision': None, 'Recall': None, 'F1': None, 'Cohen': None}
results = []

# name csv file and columns
csv_file = "results_binary_split1.csv"   
# # for predicting test set
# csv_columns = ['User', 'Number of Test Obs', 'FPR', 'TPR', 'AUC', 'Score','Precision', 'Recall', 'F1', 'Cohen']
# for predicting training set
csv_columns = ['User', 'Number of Training Obs', 'FPR', 'TPR', 'AUC', 'Score','Precision', 'Recall', 'F1', 'Cohen']

# read in sample data
# data = pd.read_csv('user_303495_data.csv')


############## BINARY CLASSIFICATION ###########

def train_binary_mood(data, hidden_units, input_dim, saveAs):

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

    """
    # had to install nomkl in anaconda for this to run
    # make target labels binary
    # removing observations with moods 4 and 5
    # I'd like to classify high mood (>=7) vs. low mood (<=4)
    # but to avoid deleting data where mood is 5 or 6 i'm classifying above and below the median of all users (see 4_11_19 Midas meeting.pptx)
    # note that the training labels are not balanced
    # 8/14/19 edit: take above and below each user's median mood from training data
    user_median_mood = data.iloc[0:71*cv,:].mood.median()
    data.loc[data.mood < user_median_mood, 'mood'] = 0 
    data.loc[data.mood >= user_median_mood, 'mood'] = 1 
    data.loc[data.mood_1 < user_median_mood, 'mood_1'] = 0 
    data.loc[data.mood_1 >= user_median_mood, 'mood_1'] = 1 
    data.loc[data.mood_2 < user_median_mood, 'mood_2'] = 0 
    data.loc[data.mood_2 >= user_median_mood, 'mood_2'] = 1 
    data.loc[data.mood_3 < user_median_mood, 'mood_3'] = 0 
    data.loc[data.mood_3 >= user_median_mood, 'mood_3'] = 1 
    data.loc[data.mood_4 < user_median_mood, 'mood_4'] = 0 
    data.loc[data.mood_4 >= user_median_mood, 'mood_4'] = 1 
    data.loc[data.mood_5 < user_median_mood, 'mood_5'] = 0 
    data.loc[data.mood_5 >= user_median_mood, 'mood_7'] = 1 
    data.loc[data.mood_6 < user_median_mood, 'mood_6'] = 0 
    data.loc[data.mood_6 >= user_median_mood, 'mood_6'] = 1 
    data.loc[data.mood_7 < user_median_mood, 'mood_7'] = 0 
    data.loc[data.mood_7 >= user_median_mood, 'mood_7'] = 1 

    """

    
    # specify the data
    # everything except mood variable
    X = data.drop('mood', axis=1) 
    
    # Isolate target labels
    Y = np.ravel(data.mood)
    
    
    # Split the data up in train and test sets
    # while preserving time-ordering
    # CV3 : train on 1:73*3 = 219
    # can also try CV2 because maybe the 3rd chunk has fewer entries and thus less variation to test on
    X_train = X.iloc[0:71*cv,:] # take row 0 to row 71*cv, excludes 71*cv
    X_test = X.iloc[71*cv:,:]
    X_train.shape
    X_test.shape
    X.shape # verify X_train.shape[0] and X_test.shape[0] (rows) add up to 356
    Y_train = Y[0:71*cv]
    Y_test = Y[71*cv:]
    Y_train.shape
    Y_test.shape
    #Y.shape # verify Y_train.shape[0] and Y_test.shape[0] (rows) add up to 356
  
    
    # check distribution of mood in training set 
    # to make sure each class is balanced
    np.unique(Y_train, return_counts = True)

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
    
    # Add one hidden layer 
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
                       
    """ ######### COMMENT IN FOR PREDICTING TEST SET LABELS ########
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
    # F1 score - weighted average of precision and recall
    f1 = f1_score(Y_test,Y_pred)
    # Cohen's kappa - classification accuracy normalized by the imbalance of the classes in the data
    cohen = cohen_kappa_score(Y_test, Y_pred)
    
    new_binary_result = {'User': saveAs, "Number of Test Obs": Y_test.shape[0], "FPR": fpr, "TPR": tpr, "AUC": auc_value, 'Score': score, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Cohen': cohen}
    
    return new_binary_result

    """ 

    ######### COMMENT IN FOR PREDICTING TRAINING SET LABELS ########
    # predict training set labels (as a comparison to test set labels, which we will also predict, i.e. change all instances below of X_train and Y_train to X_test and Y_test)
    # had to install nomkl in anaconda for this to run
    # this outputs a probability of being in class 1
    Y_pred = model.predict(X_train).ravel()
    
    # in order to calculate AUC, we need both classes 1 and 0 represented in the data
    if 1 in Y_train and 0 in Y_train:
        # false and true positive rates and thresholds
        fpr, tpr, thresholds = roc_curve(Y_train, Y_pred)
        auc_value = auc(fpr, tpr)
        print('AUC: ' + str(auc))
    else:
        fpr, tpr, thresholds, auc_value = ["","","",""] #initialize since we're printing to the csv
    
    # since the class labels are binary, choose a probability cutoff to make the predictions binary
    Y_pred = Y_pred > 0.50
    # evaluate performance
    score = model.evaluate(X_train, Y_train,verbose=1)
    # loss and accuracy -- but our training data is imbalanced so this is not useful
    # 2x2 Confusion matrix
    # confusion_matrix(Y_train, Y_pred)
    # Precision 
    precision = precision_score(Y_train, Y_pred)
    # Recall
    recall = recall_score(Y_train, Y_pred)
    # F1 score - weighted average of precision and recall
    f1 = f1_score(Y_train,Y_pred)
    # Cohen's kappa - classification accuracy normalized by the imbalance of the classes in the data
    cohen = cohen_kappa_score(Y_train, Y_pred)
    
    new_binary_result = {'User': saveAs, "Number of Training Obs": Y_train.shape[0], "FPR": fpr, "TPR": tpr, "AUC": auc_value, 'Score': score, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Cohen': cohen}
    
    return new_binary_result



##################### RUN THROUGH EACH FILE IN THE DIRECTORY ############ 
            
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.lower().endswith(".csv"):
        full_filepath = os.path.join(directory, filename)
        individual_data = pd.read_csv(full_filepath, sep=',')
  
        # check if we have any test data, and if none, skip.
        if individual_data.shape[0] <= 71*cv:
            print(filename + ' doesn\'t have any test data. Skipping this user')
        else:      
            # run neural net
            print('running neural net on ' + filename)
            new_result = train_binary_mood(data = individual_data, hidden_units = 12, input_dim = 126, saveAs = filename)
            # update the dictionary current_result with the new entry
            # current_result.update(new_result) 
            # print(current_result)
            # add the new entry to the running list of stored results
            # results.append(new_result)
            with open(csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(new_result)
    else:
        continue
    
