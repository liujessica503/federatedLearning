import csv
import numpy as np
import json
import keras
import sys
import tensorflow as tf

import FedModel

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Layer
from keras import optimizers
from OutputLayer import OutputLayer
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List


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


class FedModelPersonalized(FedModel.FedModel):

    def __init__(self, parameter_config: Dict[str, float]):
        super().__init__(parameter_config)
        self.parameter_config = parameter_config
        self.num_personalized_units = 1

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

        self.unique_users = np.unique(user_day_data.get_users())
        self.reverse_mapping = {}
        for i, v in enumerate(self.unique_users):
            self.reverse_mapping[v] = i

        self.model.add(
            PersonalizedInput(
                self.unique_users.tolist(),
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

        self.model.predict(np.zeros([1, self.input_dim + 1]))
        self.initialization = self.model.get_weights()

        self.output_layer.fit_one_hot(user_day_data.get_y())
        self.scalers_dict = {}
        for user in self.unique_users:
            X_train, Y_train = user_day_data.get_data_for_users([user])
            self.scalers_dict[user] = StandardScaler().fit(X_train)

        self.full_model_weights = self.initialization

        moment_1 = [np.zeros(np.shape(x)) for x in self.full_model_weights]
        moment_2 = [np.zeros(np.shape(x)) for x in self.full_model_weights]

        rounds_per_epoch = int(len(self.unique_users) / self.clients_per_round)

        client_weights = [0] * self.clients_per_round
        client_num_training_points = [0] * self.clients_per_round
        counter = 0
        for epch in range(self.epochs): # 8 epochs
            for rnd in range(rounds_per_epoch): # 6 rounds per epoch = 510 unique users / 80 clients per round

                # select 80 clients
                clients = np.random.choice(
                    self.unique_users, self.clients_per_round
                )

                # write the clients we're sampling to file
                with open(sys.argv[1]) as file:
                    parameter_dict = json.load(file)
                client_file_name = str(parameter_dict["output_path"] +
                    "_(" + parameter_dict['model_type'] + ")") + "clientsRecord.csv"
                with open(client_file_name, mode='a') as csvfile:
                    file_writer = csv.writer(csvfile, delimiter=',')
                    file_writer.writerow(clients)
                # end writing

                for i in range(self.clients_per_round):
                    # if self.verbose > 0:
                    print(epch, rnd, i)
                    self.model.set_weights(self.full_model_weights)
                    X_train, Y_train = user_day_data.get_data_for_users(
                        [clients[i]]
                    )

                    user_scaler = self.scalers_dict[clients[i]]
                    X_train = user_scaler.transform(X_train)
                    X_train = np.column_stack(
                        (
                            X_train,
                            np.array(
                                [self.reverse_mapping[
                                    clients[i]
                                ]] * X_train.shape[0]
                            )
                        )
                    )
                    Y_train = self.output_layer.transform_labels(Y_train)

                    # NOTE: for FedModel batch_size is ignored

                    self.model.fit(
                        X_train,
                        Y_train,
                        epochs=1,
                        verbose=self.verbose,
                        steps_per_epoch=self.local_updates_per_round
                    )
                    client_weights[i] = self.model.get_weights()
                    client_num_training_points[i] = len(Y_train)

                if self.global_aggregator == FedModel.GlobalAggregatorEnum.FEDAVG.value:
                    new_weights = self._fed_avg(
                        client_weights, client_num_training_points
                    )
                elif self.global_aggregator == FedModel.GlobalAggregatorEnum.ADAM.value:
                    counter += 1
                    new_weights, moment_1, moment_2 = self._fed_adam(
                        client_weights=client_weights,
                        client_num_training_points=client_num_training_points,
                        old_weights=self.full_model_weights,
                        moment_1=moment_1,
                        moment_2=moment_2,
                        fed_stepsize=self.fed_stepsize,
                        beta_1=0.9,
                        beta_2=0.999,
                        t=counter,
                    )
                else:
                    raise RuntimeError("global_aggregator not valid")

                for i, client in enumerate(clients):
                    new_weights[0][
                        self.reverse_mapping[client], :
                    ] = client_weights[i][0][
                        self.reverse_mapping[client], :
                    ]

                self.full_model_weights = new_weights

            # if user-inputted json set test_callback to 1,
            # we will evaluate the current model on our test set
            if test_callback == 1:
                self.model.set_weights(self.full_model_weights)
                metrics = self.evaluate(test_user_day_data)
                # import pdb
                # pdb.set_trace()

                with open(sys.argv[1]) as file:
                    parameter_dict = json.load(file)
                    loss_type = parameter_dict["output_layer"]["loss_type"]
                    model_type = parameter_dict['model_type']

                # write the test results to file (append after each epoch)
                callback_file_name = str(parameter_dict["output_path"] +
                    "_(" + parameter_dict['model_type'] + ")") + "test_per_epoch"
                with open(callback_file_name + ".json", "a") as f:
                    json.dump(metrics, f, indent=4)
                print('\nTesting metrics: {}\n'.format(metrics))
            # end code for callback

        self.model.set_weights(self.full_model_weights)
        self.is_trained = True

    def predict(self, user_day_data: Any)->List[float]:
        # self.check_is_trained()
        if self.deployment_location == FedModel.DeploymentLocationsEnum.CLIENT.value:
            predictions = self._predict_on_client(user_day_data)
        elif self.deployment_location == FedModel.DeploymentLocationsEnum.SERVER.value:
            predictions = self._predict_on_server(user_day_data)
        else:
            raise RuntimeError(
                "deployment_location must be either 'client' or 'server' "
            )
        return predictions

    def _predict_on_client(self, user_day_data: Any)->List[float]:
        """When our model is deployed on the client device, we can have the
        client save its own (most recent) standardizer, and then use that
        during prediction time.
        """
        predictions = np.empty(
            [len(user_day_data.get_y()), self.output_layer.length]
        )
        pred_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )

        for user in pred_users:
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            X_test = np.column_stack(
                (
                    X_test,
                    np.array(
                        [self.reverse_mapping[
                            user
                        ]] * X_test.shape[0]
                    )
                )
            )
            prediction = self.model.predict(X_test)
            predictions[
                user_day_data._get_rows_for_users([user]), :
            ] = prediction
        return predictions

    def _predict_on_server(self, user_day_data: Any)->List[float]:
        """When our model is deployed on the server, we do not have access to
        any of the client specific standardizers, so the best we can do is
        standardize the new data with itself.
        """
        scaler = StandardScaler().fit(user_day_data.get_X())
        X_test = scaler.transform(user_day_data.get_X())
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

    def get_score(self, user_day_data: Any)->List[float]:
        # self.check_is_trained()
        if self.deployment_location == FedModel.DeploymentLocationsEnum.CLIENT.value:
            scores = self._get_score_on_client(user_day_data)
        elif self.deployment_location == FedModel.DeploymentLocationsEnum.SERVER.value:
            scores = self._get_score_on_server(user_day_data)
        else:
            raise RuntimeError(
                "deployment_location must be either 'client' or 'server' "
            )
        return scores

    def _get_score_on_client(self, user_day_data: Any)->str:
        """When our model is deployed on the client device, we can have the
        client save its own (most recent) standardizer, and then use that
        during prediction time.
        """
        eval_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        score = ""
        for user in eval_users:
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            X_test = np.column_stack(
                (
                    X_test,
                    np.array(
                        [self.reverse_mapping[
                            user
                        ]] * X_test.shape[0]
                    )
                )
            )
            score = score + str(self.model.evaluate(X_test, Y_test)) + "\n"

        return score

    def _get_score_on_server(self, user_day_data: Any)->str:
        """When our model is deployed on the server, we do not have access to
        any of the client specific standardizers, so the best we can do is
        standardize the new data with itself.
        """
        scaler = StandardScaler().fit(user_day_data.get_X())
        X_test = scaler.transform(user_day_data.get_X())
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
