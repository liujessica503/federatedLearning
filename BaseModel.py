import tensorflow as tf
import numpy as np
import random
import os
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import backend as K  # for seed
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
import keras.callbacks
import sys
import json
from functools import partial # for leaky relu activation

# TODO: Add seed setting

class TestCallback(keras.callbacks.Callback):

    # include user for IndividualModel
    def __init__(self, test_user_day_data: Any, parentModel, userID = None):
        self.test_user_day_data = test_user_day_data
        self.parentModel = parentModel
        self.userID = userID

    def on_epoch_end(self, epoch, logs={}):

        if self.parentModel.model_type == 'individual_model':
            metrics = self.parentModel.evaluate(self.test_user_day_data, userID = self.userID)
            metrics['user'] = int(self.userID)

        # for global and federated models
        else:
            metrics = self.parentModel.evaluate(self.test_user_day_data)

        # write the test results to file (append after each epoch)
        callback_file_name = str(self.parentModel.output_path +
            "_(" + self.parentModel.model_type + ")") + "test_per_epoch"

        if self.parentModel.loss_type == 'regression':
            with open(callback_file_name + ".json", "a") as f:
                #f.write(str(metrics['mse']) + '\n')
                json.dump(metrics, f, indent=4)
        elif self.parentModel.loss_type == 'classification':
            with open(callback_file_name + ".json", "a") as f:
                #f.write(str(metrics['AUC']) + '\n')
                json.dump(metrics, f, indent=4)
        print('\nTesting metrics: {}\n'.format(metrics))
        return metrics


class BaseModel(ABC):

    def __init__(self, parameter_config: Dict[str, float]):

        self.output_path = parameter_config["output_path"]
        self.output_layer = OutputLayer.from_config(
                parameter_config["output_layer"]
            )
        self.model_type = parameter_config['model_type']
        self.loss_type = parameter_config["output_layer"]["loss_type"]

        if self.model_type != 'baseline' and self.model_type != 'moving_mean_model':

            import tensorflow as tf
            from keras.models import Sequential
            from keras.layers import Dense
            from keras import optimizers
            from keras import backend as K  # for seed


            self.layers = parameter_config["layers"]
            self.input_dim = parameter_config["input_dim"]
            self.lr = parameter_config["learn_rate"]
            self.epochs = parameter_config["epochs"]
            self.batch_size = parameter_config["batch_size"]
            self.verbose = parameter_config["verbose"]

            # set seeds
            self.seed = parameter_config["seed"] * 1234567
            random.seed(self.seed)  
            self.np_seed = self.seed * 2
            np.random.seed(self.np_seed)
            tf.set_random_seed(self.seed)

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

            self.model = Sequential()
            
            self.model.add(
                Dense(
                    self.layers[0],
                    input_dim=self.input_dim,
                    activation=partial(tf.nn.leaky_relu, alpha=0.01),
                )
            )

            ## dropout on hidden layer
            #self.model.add(Dropout(0.5))

            for i in range(1, len(self.layers)):
                self.model.add(Dense(self.layers[i], activation=partial(tf.nn.leaky_relu, alpha=0.01)))
            self.model.add(self.output_layer.layer)
            self.model.compile(
                loss=self.output_layer.loss,
                optimizer=optimizers.Adam(
                    lr=self.lr,
                    beta_1=0.9,
                    beta_2=0.999,
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
    def predict(self, user_day_data: Any, userID = None) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_score(self, user_day_data)->str:
        raise NotImplementedError

    # including user for IndividualModel
    def evaluate(self, test_user_day_data: Any, userID = None)-> Dict[str, float]:
        if self.output_layer.NAME == "BinaryLayer":
            return self._evaluate_binary(test_user_day_data, userID)
        elif self.output_layer.NAME == "RegressionLayer":
            return self._evaluate_regression(test_user_day_data, userID)
        elif self.output_layer.NAME == "MultiClassLayer":
            return self._evaluate_multiclass(test_user_day_data, userID)

    def _evaluate_binary(self, test_user_day_data: Any, userID = None)-> Dict[str, float]:
        
        # if we're in a callback for the indivdual model
        if self.is_trained == False and userID:
            # get a prediction for a single user
            predictions, test_labels = self.predict(test_user_day_data, userID)
            # evaluate performance
            score = self.get_score(test_user_day_data, userID)

        # if we're done training (not in a callback) for the individual model
        # or for global and federated models
        else: 
            predictions = self.predict(test_user_day_data)
            test_labels = test_user_day_data.get_y()
            # evaluate performance
            score = self.get_score(test_user_day_data)

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
            "AUC": auc_value,
            'Score': score,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Cohen': cohen,
        }
        return metrics

    def _evaluate_regression(self, test_user_day_data: Any, userID = None)-> Dict[str, float]:
        predictions = None
        # for individual model, use individual model's predict function
        if userID:
            # if we're in a callback, get a prediction for a single user
            if self.is_trained == False:
                print('running individual model\'s predict function')
                # use individual model's predict function
                # when we're in a callback, returns prediction and observed test labels 
                prediction, test_labels = self.predict(test_user_day_data, userID)

                # cap predictions to be between 1-10, since our mood is measured on 1-10 scale
                prediction = [10 if x > 10 else 1 if x < 1 else x for x in prediction]
                mse = mean_squared_error(test_labels, prediction)
            # if we're done training (not in a callback), get all the individual predictions
            elif self.is_trained == True:
                predictions = self.predict(test_user_day_data, userID)
                predictions = [10 if x > 10 else 1 if x < 1 else x for x in predictions]
        # for global and federated models, use the respective predict function
        else: 
            predictions = self.predict(test_user_day_data)
            predictions = [10 if x > 10 else 1 if x < 1 else x for x in predictions]
            test_labels = test_user_day_data.get_y()
            mse = mean_squared_error(test_labels, predictions)
       
        return {
            "Number of Test Obs": test_labels.shape[0],
            "mse": mse
        }

    def _evaluate_multiclass(self, test_user_day_data: Any, userID = None)-> Dict[str, float]:
        if userID:
            print('running individual model\'s predict function')
            # use individual model's predict function
            predictions, test_labels = self.predict(test_user_day_data, userID)
        else: 
            # use global / federated model's predict function
            predictions = self.predict(test_user_day_data)
            test_labels = test_user_day_data.get_y()

        class_predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(test_labels, class_predictions, average="weighted")
        accuracy = accuracy_score(test_labels, class_predictions)
        return {
            "Number of Test Obs": test_labels.shape[0],
            "f1": f1,
            "accuracy": accuracy
        }

    def individual_evaluate(self, user_day_data: Any)->Dict[float, str]:
        # self.check_is_trained()

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

    def reset(self)->None:
        self.model.set_weights(self.initialization)
        self.is_trained = False
        # 1/29/2020 trying this -- will only affect if you run multiple experiments without exiting python interpreter
        tf.reset_default_graph()

    # not using this when we are doing callbacks, since for callbacks we will be testing before model has finished training
    def check_is_trained(self)->None:
        if not self.is_trained:
            raise RuntimeError("Model not yet trained")


