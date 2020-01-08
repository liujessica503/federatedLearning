import tensorflow as tf
import numpy as np
import random

from abc import ABC, abstractmethod
from typing import Any, List, Dict
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from OutputLayer import OutputLayer
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)

# TODO: Add seed setting


class BaseModel(ABC):

    def __init__(
        self, parameter_config: Dict[str, float]):
    
        self.layers = parameter_config["layers"]
        self.input_dim = parameter_config["input_dim"]
        self.activation = parameter_config["activation"]
        self.lr = parameter_config["learn_rate"]
        self.epochs = parameter_config["epochs"]
        self.batch_size = parameter_config["batch_size"]
        self.verbose = parameter_config["verbose"]
        self.output_path = parameter_config["output_path"]
        self.seed = parameter_config["seed"] * 1234567
        self.np_seed = self.seed * 2
        np.random.seed(self.np_seed)
        tf.random.set_random_seed(self.seed)
        random.seed(self.seed)

        self.model = Sequential()
        self.output_layer = OutputLayer.from_config(
            parameter_config["output_layer"]
        )
        self.model.add(
            Dense(
                self.layers[0],
                input_dim=self.input_dim,
                activation=self.activation,
            )
        )
        for i in range(1, len(self.layers)):
            self.model.add(Dense(self.layers[i], activation=self.activation))
        self.model.add(self.output_layer.layer)
        self.model.compile(
            loss=self.output_layer.loss,
            optimizer=optimizers.Adam(
                lr=self.lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False,
            ),
            metrics=self.output_layer.metrics,
        )
        self.initialization = self.model.get_weights()
        self.is_trained = False

    @abstractmethod
    def train(self, user_day_data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, user_day_data: Any)-> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, user_day_data)->str:
        raise NotImplementedError

    def evaluate(self, test_user_day_data: Any)-> Dict[str, float]:
        if self.output_layer.NAME == "BinaryLayer":
            return self._evaluate_binary(test_user_day_data)
        elif self.output_layer.NAME == "RegressionLayer":
            return self._evaluate_regression(test_user_day_data)
        elif self.output_layer.NAME == "MultiClassLayer":
            return self._evaluate_multiclass(test_user_day_data)

    def _evaluate_binary(self, test_user_day_data: Any)-> Dict[str, float]:

        predictions = self.predict(test_user_day_data)
        test_labels = test_user_day_data.get_y()

        binary_prediction = predictions > 0.50

        # both 1 and 0 have to be true labels in test set to calculate AUC
        if 1 in test_labels and 0 in test_labels:
            # false and true positive rates and thresholds
            fpr, tpr, thresholds = roc_curve(test_labels, predictions)
            auc_value = auc(fpr, tpr)
            fpr = fpr.tolist()
            tpr = tpr.tolist()
        else:
            # initialize since we're printing to the csv
            fpr, tpr, auc_value = [[], [], None]

        # evaluate performance
        score = self.get_score(test_user_day_data)
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

    def _evaluate_regression(self, test_user_day_data: Any)-> Dict[str, float]:
        predictions = self.predict(test_user_day_data)
        test_labels = test_user_day_data.get_y()
        mse = mean_squared_error(test_labels, predictions)
       
        return {
            "Number of Test Obs": test_labels.shape[0],
            "mse": mse
        }

    def _evaluate_multiclass(self, test_user_day_data: Any)-> Dict[str, float]:
        predictions = self.predict(test_user_day_data)
        class_predictions = np.argmax(predictions, axis=1)
        test_labels = test_user_day_data.get_y()
        f1 = f1_score(test_labels, class_predictions, average="weighted")
        accuracy = accuracy_score(test_labels, class_predictions)
        return {
            "Number of Test Obs": test_labels.shape[0],
            "f1": f1,
            "accuracy": accuracy
        }

    '''
    def individual_evaluate(
        self, user_day_data: Any, plotAUC=False
    )->Dict[float, str]:
        self.check_is_trained()

        eval_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        metrics_dict = {}

        for user in eval_users:
            ind_user_day_data = user_day_data.get_subset_for_users([user])
            metrics = self.evaluate(ind_user_day_data)
            try:
                del metrics["FPR"]
                del metrics["TPR"]
            except KeyError:
                pass
            metrics_dict[int(user)] = metrics

        return metrics_dict
    '''

    def reset(self)->None:
        self.model.set_weights(self.initialization)
        self.is_trained = False

    def check_is_trained(self)->None:
        if not self.is_trained:
            raise RuntimeError("Model not yet trained")
