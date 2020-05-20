import numpy as np
import keras
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Layer
from keras import optimizers
from OutputLayer import OutputLayer
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List

from BaseModel import BaseModel, TestCallback


class PersonalizedInput(Layer):

    def __init__(self, users: List[int], personalized_units=1, input_dim=1):
        super(PersonalizedInput, self).__init__()
        self.len_users = len(users)
        self.personalized_units = personalized_units

    def build(self, input_shape):
        self.personal_embeddings = Embedding(
            input_dim=self.len_users,
            output_dim=self.personalized_units,
            input_length=input_shape[0]
        )

    def call(self, inputs):
        embedding = self.personal_embeddings(inputs[:, -1])
        new_inputs = keras.layers.concatenate(
            [inputs[:, :-1], embedding]
        )
        return new_inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - 1 + self.personalized_units)


class GlobalModelPersonalized(BaseModel):

    def __init__(self, parameter_config: Dict[str, float]):
        super().__init__(parameter_config)
        self.parameter_config = parameter_config
        self.num_personalized_units = 10

    def train(
        self, user_day_data: Any, test_user_day_data: Any, test_callback=0
    )->None:

        K.clear_session()

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        self.model = Sequential()
        self.output_layer = OutputLayer.from_config(
            self.parameter_config["output_layer"]
        )

        unique_users = np.unique(user_day_data.get_users())
        self.reverse_mapping = {}
        for i, v in enumerate(unique_users):
            self.reverse_mapping[v] = i

        self.model.add(
            PersonalizedInput(
                unique_users.tolist(),
                personalized_units=self.num_personalized_units,
                input_dim=self.input_dim + 1,
            )
        )
        self.model.add(
            Dense(
                self.layers[0],
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
                decay=0.0,
                amsgrad=False,
            ),
            metrics=self.output_layer.metrics,
        )

        self.scaler = StandardScaler().fit(user_day_data.get_X())
        # Scale the train set
        X_train = self.scaler.transform(user_day_data.get_X())
        Y_train = user_day_data.get_y()

        self.output_layer.fit_one_hot(Y_train)
        Y_train = self.output_layer.transform_labels(Y_train)

        # if json set test_callback to 1, after each epoch of training,
        # we will evaluate the current model on our test set
        callback_list = []
        if test_callback == 1:
            callback_list = [TestCallback((test_user_day_data), self)]

        # apply the template model, created in the init, to our data
        X_train = np.column_stack(
            (
                X_train,
                np.array(
                    [self.reverse_mapping[
                        x
                    ] for x in user_day_data.get_users()]
                )
            )
        )
        self.model.fit(
            X_train,
            Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callback_list
        )

        self.is_trained = True

    def predict(self, user_day_data: Any, userID = None) -> List[float]:
        X_test = self.scaler.transform(user_day_data.get_X())
        X_test = np.column_stack(
            (
                X_test,
                np.array(
                    [self.reverse_mapping[
                        x
                    ] for x in user_day_data.get_users()]
                )
            )
        )
        return self.model.predict(X_test)

    def get_score(self, user_day_data)->str:
        X_test = self.scaler.transform(user_day_data.get_X())
        X_test = np.column_stack(
            (
                X_test,
                np.array(
                    [self.reverse_mapping[
                        x
                    ] for x in user_day_data.get_users()]
                )
            )
        )
        return self.model.evaluate(X_test, user_day_data.get_y())
