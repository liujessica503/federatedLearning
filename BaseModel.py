from abc import ABC, abstractmethod
from typing import Any, List
from plot_auc import plot_auc
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)


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

    def evaluate(
        self,
        user_model: Any,
        test_covariates: Any,
        test_labels: Any,
        predictions: List[int],
        plotAUC=False,
        globalModel=False
    ) -> dict():
        # get more metrics
        # since the class labels are binary, choose a probability cutoff to
        # make the predictions binary
        binary_prediction = predictions > 0.50

        # both 1 and 0 have to be true labels in test set to calculate AUC
        if 1 in test_labels and 0 in test_labels:
            # false and true positive rates and thresholds
            fpr, tpr, thresholds = roc_curve(test_labels, binary_prediction)
            auc_value = auc(fpr, tpr)
            if plotAUC is True:
                plot_auc(fpr, tpr, auc_value, filename=self.auc_output_path)
        else:
            # initialize since we're printing to the csv
            fpr, tpr, auc_value = ["", "", ""]

        # evaluate performance
        # for global, user_model.evaluate should be self.model.evaluate
        if globalModel is True:
            score = self.model.evaluate(test_covariates, test_labels)
        else:
            score = user_model.evaluate(test_covariates, test_labels)
        # Precision
        precision = precision_score(test_labels, binary_prediction)
        # Recall
        recall = recall_score(test_labels, binary_prediction)
        # print('Precision: ' + str(precision) + ', Recall: ' + str(recall))
        # F1 score - weighted average of precision and recall
        f1 = f1_score(test_labels, binary_prediction)
        # Cohen's kappa - classification accuracy normalized by the imbalance
        # of the classes in the data
        cohen = cohen_kappa_score(test_labels, binary_prediction)

        metrics = {
            "Number of Test Obs": test_labels.shape[0],
            "FPR": fpr,
            "TPR": tpr,
            "AUC": auc_value,
            'Score': score,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Cohen': cohen,
        }
        return metrics

    @abstractmethod
    def reset(self)->None:
        raise NotImplementedError
