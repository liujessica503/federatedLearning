from abc import ABC, abstractmethod
from typing import Any, List
from plot_auc import plot_auc
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

class BaseModel(ABC):

    def __init__(self, parameter_config):

        self.layers = parameter_config["layers"]
        self.input_dim = parameter_config["input_dim"]
        self.activation = parameter_config["activation"]
        self.loss = parameter_config["loss"]
        self.lr = parameter_config["learn_rate"]
        self.epochs = parameter_config["epochs"]
        self.batch_size = parameter_config["batch_size"]
        self.verbose = parameter_config["verbose"]
        self.output_path = parameter_config["output_path"]
        self.auc_output_path = parameter_config["auc_output_path"]
        self.plot_auc = parameter_config["plot_auc"]

    @abstractmethod
    def train(self, X: Any, y: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any)-> List[int]:
        raise NotImplementedError

    def evaluate(self, user_day_data: Any, predictions: List[int], plotAUC = False) -> dict():

        # get more metrics
            # since the class labels are binary, choose a probability cutoff to make the predictions binary
        binary_prediction = predictions > 0.50

        # both 1 and 0 have to be true labels in test set to calculate AUC
        if 1 in binary_prediction and 0 in binary_prediction:
            # false and true positive rates and thresholds
            fpr, tpr, thresholds = roc_curve(user_day_data.y, binary_prediction)
            auc_value = auc(fpr, tpr)
            if plotAUC == True:
                plot_auc(fpr, tpr, auc_value, filename = self.auc_output_path)
        else:
            fpr, tpr, thresholds, auc_value = ["","","",""] #initialize since we're printing to the csv

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

    @abstractmethod
    def reset(self)->None:
        raise NotImplementedError
