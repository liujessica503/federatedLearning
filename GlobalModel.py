from BaseModel import BaseModel
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

class GlobalModel(BaseModel):

    def __init__(self, parameter_config: dict()):
        super().__init__(parameter_config)

        self.model = Sequential()

        self.model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation))

        for i in range(1,len(self.layers)):
            self.model.add(Dense(self.layers[i], activation = self.activation))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
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

    def train(self, user_day_data: Any)->None:
        self.scaler = StandardScaler().fit(user_day_data.X)
        
        # Scale the train set
        X_train = self.scaler.transform(user_day_data.X)
        Y_train = user_day_data.y
        
        self.model.fit(X_train, Y_train,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def validate(self, X: Any, Y: Any, validation_data = None)->None:
        modelFit = self.model.fit(X, Y,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_data = validation_data)
        # return an object for running cross-validation purposes
        return modelFit

    def predict(self, user_day_data: Any)->List[int]:
        X_test = self.scaler.transform(user_day_data.X)
        return self.model.predict(X_test).ravel()
        
    def reset(self)->None:
        pass
