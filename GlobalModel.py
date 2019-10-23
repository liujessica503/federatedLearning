from BaseModel import BaseModel
from plot_auc import plot_auc
from typing import Dict, List, Any
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam
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
        self.model.fit(user_day_data.X, user_day_data.y,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def validate(self, X: Any, Y: Any, validation_data = None)->None:
        modelFit = self.model.fit(X, Y,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_data = validation_data)
        # return an object for running cross-validation purposes
        return modelFit

    def predict(self, user_day_data: Any)->List[int]:
        return self.model.predict(user_day_data.X).ravel()

    def evaluate(self, user_day_data: Any, predictions: List[int], plotAUC = False) -> dict():

        # both 1 and 0 have to be true labels in test set to calculate AUC
        if 1 in predictions and 0 in predictions:
            # false and true positive rates and thresholds
            fpr, tpr, thresholds = roc_curve(user_day_data.y, predictions)
            auc_value = auc(fpr, tpr)
            if plotAUC == True:
                plot_auc(fpr, tpr, auc_value, filename = self.auc_output_path)
        else:
            fpr, tpr, thresholds, auc_value = ["","","",""] #initialize since we're printing to the csv


        # get more metrics
            # since the class labels are binary, choose a probability cutoff to make the predictions binary
        binary_prediction = predictions > 0.50
        # evaluate performance
        score = self.model.evaluate(user_day_data.X, user_day_data.y,verbose=1)
        # Precision 
        precision = precision_score(user_day_data.y, binary_prediction)
        # Recall
        recall = recall_score(user_day_data.y, binary_prediction)
        # print('Precision: ' + str(precision) + ', Recall: ' + str(recall))
        # F1 score - weighted average of precision and recall
        f1 = f1_score(user_day_data.y, binary_prediction)
        # Cohen's kappa - classification accuracy normalized by the imbalance of the classes in the data
        cohen = cohen_kappa_score(user_day_data.y, binary_prediction)

        metrics = {"Number of Test Obs": predictions.shape[0], "FPR": fpr, "TPR": tpr, "AUC": auc_value, 'Score': score, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Cohen': cohen}
        return metrics
        
    def reset(self)->None:
        pass
