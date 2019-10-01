import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam

class GlobalModelServer:

    def __init__(
        self, 
        parameter_config, 
        X_train,
        Y_train, 
        X_eval, 
        Y_eval, 
        X_test,
        Y_test,
    ):
        pass

        # Save data

        self.X_train = X_train
        self.Y_train = Y_train

        self.X_eval = X_eval
        self.Y_eval = Y_eval

        self.X_test = X_test
        self.Y_test = Y_test

        # Save parameters

        self.layers = parameter_config["layers"]
        self.epochs = parameter_config["epochs"]
        self.lr = parameter_config["lr"]
        self.batch_size = parameter_config["batch_size"]

        # Use parameters to build model

        model = Sequential()

    def train(self, epochs = self.epochs):
        model.fit(
            X_train, 
            Y_train, 
            epochs=epochs, 
            batch_size=self.batch_size, 
            verbose=1
        )

    def get_eval(self):
        pass

    def get_test(self):
        pass

