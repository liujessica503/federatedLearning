from BaseModel import BaseModel, TestCallback
#from TestCallback import TestCallback
import numpy as np
from typing import Any, List


class BaselineModel(BaseModel):

    def train(self, user_day_data: Any, test_user_day_data: Any, test_callback = 0)->None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )

        self.training_mean_dict = {}

        for user in self.unique_users:
            # get user-specific data
            X_train, Y_train = user_day_data.get_data_for_users([user])
            # get the mean of the outcome in the training data
            self.training_mean_dict[user] = np.mean(Y_train)

        self.is_trained = True

    def predict(self, user_day_data: Any)->List[float]:
        predictions = np.empty(
                [len(user_day_data.get_y()), self.output_layer.length]
            )
        pred_users = np.unique(
                [x[0] for x in user_day_data.get_user_day_pairs()]
            )

        for user in pred_users:
            X_test, Y_test = user_day_data.get_data_for_users([user])
            training_mean = self.training_mean_dict[user]
            prediction =  np.array([[training_mean] * len(Y_test)]).transpose()
            predictions[
                user_day_data._get_rows_for_users([user]), :
            ] = prediction
        return predictions

    def get_score(self, user_day_data)->str:
        return ""
