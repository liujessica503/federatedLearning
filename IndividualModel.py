from BaseModel import BaseModel
import numpy as np
from typing import Any, List, Dict
# standardize the data
from sklearn.preprocessing import StandardScaler


class IndividualModel(BaseModel):

    def train(self, user_day_data: Any)->None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        self.model_weights_dict = {}
        self.scalers_dict = {}
        for user in self.unique_users:

            self.model.set_weights(self.initialization)

            # get user-specific data
            X_train, Y_train = user_day_data.get_data_for_users([user])
            user_scaler = StandardScaler().fit(X_train)
            X_train = user_scaler.transform(X_train)

            # apply the template model, created in the init, to our data
            self.model.fit(
                X_train,
                Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )

            self.model_weights_dict[user] = self.model.get_weights()
            self.scalers_dict[user] = user_scaler

    def predict(self, user_day_data: Any)->List[float]:

        # check if unique.users exista and only proceed if so
        try:
            self.unique_users
        except NameError:
            self.unique_users = None
            print('Please run the train method and create unique_users')
            return None

        predictions = np.empty(len(user_day_data.get_y()))
        pred_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )

        for user in pred_users:
            self.model.set_weights(self.model_weights_dict[user])
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            prediction = self.model.predict(X_test).ravel()
            predictions[user_day_data._get_rows_for_users([user])] = prediction
        return predictions

    def get_score(self, user_day_data: Any)->str:
        try:
            self.unique_users
        except NameError:
            self.unique_users = None
            print('Please run the train method and create unique_users')
            return None

        eval_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        score = ""
        for user in eval_users:
            self.model.set_weights(self.model_weights_dict[user])
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            score = score + str(self.model.evaluate(X_test, Y_test)) + "\n"

        return score

    def individual_evaluate(
        self, user_day_data: Any, plotAUC=False
    )->Dict[float, str]:

        # check if unique.users exists and only proceed if so
        try:
            self.unique_users
        except NameError:
            self.unique_users = None
            print('Please run the train method and create unique_users')
            return None

        eval_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        metrics_dict = {}

        for user in eval_users:
            # TODO: Fix risky fake user_day list
            ind_user_day_data = user_day_data.get_subset_for_users([user])
            metrics = self.evaluate(ind_user_day_data)
            metrics_dict[int(user)] = metrics

        return metrics_dict
