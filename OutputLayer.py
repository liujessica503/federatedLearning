from abc import ABC, abstractmethod
from keras.layers import Dense
from sklearn import preprocessing


class OutputLayer(ABC):

    NAME = None

    @classmethod
    def from_config(cls, output_layer_dict):
        if output_layer_dict["loss_type"] == "regression":
            return RegressionLayer(output_layer_dict)
        elif output_layer_dict["loss_type"] == "classification":
            if len(
                output_layer_dict["classification_thresholds"]
            ) == 1:
                return BinaryLayer(output_layer_dict)
            else:
                return MultiClassLayer(output_layer_dict)

    @abstractmethod
    def transform_labels(self, labels):
        raise NotImplementedError


class BinaryLayer(OutputLayer):

    NAME = "BinaryLayer"

    def __init__(self, output_layer_dict):
        self.length = 1
        self.layer = Dense(self.length, activation='sigmoid')
        self.loss = "binary_crossentropy"
        self.metrics = ['accuracy']

    def transform_labels(self, labels):
        return labels

    def fit_one_hot(self, all_labels):
        pass


class MultiClassLayer(OutputLayer):

    NAME = "MultiClassLayer"

    def __init__(self, output_layer_dict):
        self.length = len(output_layer_dict["classification_thresholds"]) + 1
        self.layer = Dense(self.length, activation='softmax')
        self.loss = "categorical_crossentropy"
        self.metrics = ['accuracy']
        self.lb = preprocessing.LabelBinarizer()

    def transform_labels(self, labels):
        return self.lb.transform(labels)

    def fit_one_hot(self, all_labels):
        self.lb.fit(all_labels)


class RegressionLayer(OutputLayer):

    NAME = "RegressionLayer"

    def __init__(self, output_layer_dict):
        self.length = 1
        self.layer = Dense(self.length)
        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]

    def transform_labels(self, labels):
        return labels
