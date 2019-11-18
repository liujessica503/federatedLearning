import numpy as np

from BaseModel import BaseModel
from enum import Enum
from typing import Any, Dict, List
# standardize the data
from sklearn.preprocessing import StandardScaler


class FedModel(BaseModel):

    def __init__(
        self, parameter_config: Dict[str, float], parameter_overwrite={}
    ):

        super().__init__(parameter_config, parameter_overwrite)
        self.clients_per_round = parameter_config["clients_per_round"]
        self.local_epochs_per_round = parameter_config[
            "local_epochs_per_round"
        ]
        self.deployment_location = parameter_config[
            "deployment_location"
        ]

    def train(self, user_day_data: Any) -> None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        self.scalers_dict = {}
        for user in self.unique_users:
            X_train, Y_train = user_day_data.get_data_for_users([user])
            self.scalers_dict[user] = StandardScaler().fit(X_train)

        self.full_model_weights = self.initialization
        rounds_per_epoch = int(len(self.unique_users) / self.clients_per_round)
        client_weights = [0] * self.clients_per_round
        client_num_training_points = [0] * self.clients_per_round
        for _ in range(self.epochs):
            for __ in range(rounds_per_epoch):
                clients = np.random.choice(
                    self.unique_users, self.clients_per_round
                )
                for i in range(self.clients_per_round):
                    print(_, __, i)
                    self.model.set_weights(self.full_model_weights)
                    X_train, Y_train = user_day_data.get_data_for_users(
                        [clients[i]]
                    )
                    user_scaler = self.scalers_dict[clients[i]]
                    X_train = user_scaler.transform(X_train)

                    self.model.fit(
                        X_train,
                        Y_train,
                        epochs=self.local_epochs_per_round,
                        batch_size=self.batch_size,
                        verbose=self.verbose,
                    )
                    client_weights[i] = self.model.get_weights()
                    client_num_training_points[i] = len(Y_train)

                num_training_points = sum(client_num_training_points)
                for i in range(self.clients_per_round):
                    for j in range(len(client_weights[0])):
                        client_weights[i][j] = client_weights[i][
                            j
                        ] * client_num_training_points[i] / num_training_points

                new_weights = []
                for j in range(len(client_weights[0])):
                    tmp = client_weights[0][j]
                    for i in range(1, self.clients_per_round):
                        tmp = tmp + client_weights[i][j]
                    new_weights.append(tmp)
                self.full_model_weights = new_weights

        self.model.set_weights(self.full_model_weights)
        self.is_trained = True

    def predict(self, user_day_data: Any)->List[float]:
        self.check_is_trained()
        if self.deployment_location == DeploymentLocationsEnum.CLIENT.value:
            predictions = self._predict_on_client(user_day_data)
        elif self.deployment_location == DeploymentLocationsEnum.SERVER.value:
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
        predictions = np.empty(len(user_day_data.get_y()))
        pred_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )

        for user in pred_users:
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            prediction = self.model.predict(X_test).ravel()
            predictions[user_day_data._get_rows_for_users([user])] = prediction
        return predictions

    def _predict_on_server(self, user_day_data: Any)->List[float]:
        """When our model is deployed on the server, we do not have access to
        any of the client specific standardizers, so the best we can do is
        standardize the new data with itself.
        """
        scaler = StandardScaler().fit(user_day_data.get_X())
        X_test = scaler.transform(user_day_data.get_X())
        return self.model.predict(X_test).ravel()

    def get_score(self, user_day_data: Any)->List[float]:
        self.check_is_trained()
        if self.deployment_location == DeploymentLocationsEnum.CLIENT.value:
            scores = self._get_score_on_client(user_day_data)
        elif self.deployment_location == DeploymentLocationsEnum.SERVER.value:
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
            score = score + str(self.model.evaluate(X_test, Y_test)) + "\n"

        return score

    def _get_score_on_server(self, user_day_data: Any)->str:
        """When our model is deployed on the server, we do not have access to
        any of the client specific standardizers, so the best we can do is
        standardize the new data with itself.
        """
        scaler = StandardScaler().fit(user_day_data.get_X())
        X_test = scaler.transform(user_day_data.get_X())
        return self.model.evaluate(X_test, user_day_data.get_y())


class DeploymentLocationsEnum(Enum):
    CLIENT = "client"
    SERVER = "server"
