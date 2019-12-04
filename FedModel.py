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
        self.fed_model_parameters = parameter_config["fed_model_parameters"]
        self.clients_per_round = self.fed_model_parameters["clients_per_round"]
        self.local_updates_per_round = self.fed_model_parameters[
            "local_updates_per_round"
        ]
        self.deployment_location = self.fed_model_parameters[
            "deployment_location"
        ]
        self.global_aggregator = self.fed_model_parameters[
            "global_aggregator"
        ]
        self.fed_stepsize = self.fed_model_parameters["fed_stepsize"]

    def train(self, user_day_data: Any) -> None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
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

                if self.global_aggregator == GlobalAggregatorEnum.FEDAVG.value:
                    new_weights = self._fed_avg(
                        client_weights, client_num_training_points
                    )
                elif self.global_aggregator == GlobalAggregatorEnum.ADAM.value:
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
        return self.model.predict(X_test)

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

    @staticmethod
    def _fed_avg(
        client_weights: List[List[Any]],
        client_num_training_points: List[int],
    )->List[Any]:
        num_training_points = sum(client_num_training_points)
        for i in range(len(client_weights)):
            for j in range(len(client_weights[0])):
                client_weights[i][j] = client_weights[i][
                    j
                ] * client_num_training_points[i] / num_training_points

        new_weights = []
        for j in range(len(client_weights[0])):
            tmp = client_weights[0][j]
            for i in range(1, len(client_weights)):
                tmp = tmp + client_weights[i][j]
            new_weights.append(tmp)

        return new_weights

    @staticmethod
    def _fed_adam(
        client_weights: List[List[Any]],
        client_num_training_points: List[int],
        old_weights: List[Any],
        moment_1: List[Any],
        moment_2: List[Any],
        fed_stepsize: float,
        beta_1: float,
        beta_2: float,
        t: int,
    )->List[Any]:
        n_vec = len(old_weights)
        avg_weights = FedModel._fed_avg(
            client_weights, client_num_training_points
        )
        update_vector = [
            old_weights[i] - avg_weights[i] for i in range(n_vec)
        ]
        moment_1 = [
            beta_1 * moment_1[i] +
            (1 - beta_1) * update_vector[i] for i in range(n_vec)
        ]
        moment_2 = [
            beta_2 * moment_2[i] +
            (1 - beta_2) *
            (update_vector[i] * update_vector[i]) for i in range(n_vec)
        ]
        # bc = bias corrected
        bc_moment_1 = [
            vec / (1 - beta_1**t) for vec in moment_1
        ]
        bc_moment_2 = [
            vec / (1 - beta_2**t) for vec in moment_2
        ]
        sqrt_bc_moment_2 = [
            np.sqrt(vec) + 1e-8 for vec in bc_moment_2
        ]
        new_weights = [
            old_weights[i] - fed_stepsize *
            (bc_moment_1[i] / sqrt_bc_moment_2[i]) for i in range(n_vec)
        ]

        return new_weights, moment_1, moment_2


class DeploymentLocationsEnum(Enum):
    CLIENT = "client"
    SERVER = "server"


class GlobalAggregatorEnum(Enum):
    FEDAVG = "fed_avg"
    ADAM = "adam"
