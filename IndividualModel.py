from BaseModel import BaseModel
import copy
import numpy as np
from plot_auc import plot_auc
from typing import Dict, List, Any
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam
# standardize the data
from sklearn.preprocessing import StandardScaler
# for binary classification
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
# to write to csv
import csv

class IndividualModel(BaseModel):

    def __init__(self, parameter_config: dict()):
        super().__init__(parameter_config)

        self.template_model = Sequential()

        self.template_model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation))

        for i in range(1,len(self.layers)):
            self.template_model.add(Dense(self.layers[i], activation = self.activation))


        self.template_model.add(Dense(1, activation='sigmoid'))
        self.template_model.compile(
            loss=self.loss,
            optimizer = optimizers.Adam(
                lr=self.lr, 
                beta_1=0.9, 
                beta_2=0.999, 
                epsilon=None, 
                decay=0.0, 
                amsgrad=False
            ),
            metrics=['accuracy'],
        )

    def train(self, user_day_data: Any, validation_data = None)->None:
        self.unique_users = np.unique([x[0] for x in user_day_data.user_day_pairs])
        self.models_dict = {}
        self.scalers_dict = {}
        for user in self.unique_users:
            user_model = copy.deepcopy(self.template_model)
            X_train, Y_train = user_day_data.get_data_for_users([user])
            user_scaler = StandardScaler().fit(X_train)
            X_train = user_scaler.transform(X_train)

            user_model.fit(X_train, Y_train,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

            self.models_dict[user] = user_model
            self.scalers_dict[user] = user_scaler

        # modelFit = self.model.fit(X_dict, Y_dict,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_data = validation_data)
        # return modelFit

    def predict(self, user_day_data: Any)-> None:
        self.predictions_dict = {}

        try:
            for user in self.unique_users:
                try:
                    user_prediction_model = self.models_dict[user]
                    user_prediction_scaler = self.scalers_dict[user]
                    X_test = user_prediction_scaler.transform(user_day_data.X)
                    prediction = user_prediction_model.predict(X_test).ravel()
                    self.predictions_dict[user] = prediction
                except:
                    print('no data found for this user')
                    continue

        # print message if self.unique_users doesn't exist
        except NameError:
            print('missing unique_users (created in train method)')
        else:
            print('error in predict method')
        return self.predictions_dict

    def individual_evaluate(self, user_day_data: Any, predictions_dict: Any, plotAUC = False) -> dict():
        self.metrics_dict = {}

        try:
            for user in self.unique_users:
                try:
                    predictions = self.predictions_dict[user]

                    metrics = evaluate(user_day_data = user_day_data, predictions = predictions, plotAUC = plotAUC)
                    self.metrics_dict[user] = metrics
                    print(metrics)
                except:
                    print('no predictions found for this user')
                    continue

        # print message if self.unique_users doesn't exist
        except NameError:
            print('missing unique_users (created in train method)')
        else:
            print('error in evaluate method')
        return self.metrics_dict


    def reset(self)->None:
        pass
