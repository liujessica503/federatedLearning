from abc import ABC
from keras.layers import Dense


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


class BinaryLayer(OutputLayer):

    NAME = "BinaryLayer"

    def __init__(self, output_layer_dict):
        self.layer = Dense(1, activation='sigmoid')
        self.loss = "binary_crossentropy"
        self.metrics = ['accuracy']


class MultiClassLayer(OutputLayer):

    NAME = "MultiClassLayer"

    def __init__(self, output_layer_dict):
        self.layer = Dense(
            len(output_layer_dict["classification_thresholds"]),
            activation='sigmoid',
        )
        self.loss = "categorical_crossentropy"
        self.metrics = ['accuracy']


class RegressionLayer(OutputLayer):

    NAME = "RegressionLayer"

    def __init__(self, output_layer_dict):
        self.layer = Dense(1)
        self.loss = "mean_squared_error"
        self.metrics = ["mae", "mse"]
