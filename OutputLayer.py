from abc import ABC
from keras.layers import Dense


class OutputLayer(ABC):

    @classmethod
    def from_config(cls, output_layer_dict):
        if output_layer_dict["loss_type"] == "classification":
            return ClassificationLayer(output_layer_dict)
        elif output_layer_dict["loss_type"] == "regression":
            return RegressionLayer(output_layer_dict)


class ClassificationLayer(OutputLayer):

    def __init__(self, output_layer_dict):
        self.layer = Dense(
            len(output_layer_dict["classification_thresholds"]),
            activation='sigmoid',
        )
        if len(
            output_layer_dict["classification_thresholds"]
        ) == 1:
            self.loss = "binary_crossentropy"

        else:
            self.loss = "categorical_crossentropy"


class RegressionLayer(OutputLayer):

    def __init__(self, output_layer_dict):
        self.layer = Dense(1, "relu")
        self.loss = "mean_squared_error"
