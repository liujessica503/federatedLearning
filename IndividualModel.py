from BaseModel import BaseModel
import numpy as np
from typing import Any
# standardize the data
from sklearn.preprocessing import StandardScaler


class IndividualModel(BaseModel):

    def __init__(self, parameter_config: dict, parameter_overwrite={}):
        super().__init__(parameter_config, parameter_overwrite)
        self.initialization = self.model.get_weights()

    def train(self, user_day_data: Any, validation_data=None)->None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.user_day_pairs]
        )
        self.model_weights_dict = {}
        self.scalers_dict = {}
        for user in self.unique_users:
            # deep copy-ing doesn't work in flux
            # user_model = copy.deepcopy(self.template_model)
            # so we clone, build, and compile instead

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
                validation_data=validation_data
            )

            self.model_weights_dict[user] = self.model.get_weights()
            self.scalers_dict[user] = user_scaler

        # modelFit = self.model.fit(X_dict, Y_dict,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_data = validation_data) # noqa
        # return modelFit

    def predict(self, user_day_data: Any)-> None:
        self.predictions_dict = {}

        # check if unique.users exists
        # and only proceed if so
        try:
            self.unique_users
        except NameError:
            self.unique_users = None
            print('Please run the train method and create unique_users')
            return None

        for user in self.unique_users:
            self.model.set_weights(self.model_weights_dict[user])
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            if len(Y_test) > 0:
                X_test = user_prediction_scaler.transform(X_test)
                prediction = self.model.predict(X_test).ravel()
                self.predictions_dict[user] = prediction
            else:
                print('no data found for user ')
        return self.predictions_dict

    def individual_evaluate(
        self, user_day_data: Any, predictions_dict: Any, plotAUC=False
    ) -> dict():
        self.metrics_dict = {}

        # check if unique.users exists
        # and only proceed if so
        try:
            self.unique_users
        except NameError:
            self.unique_users = None
            print('Please run the train method and create unique_users')
            return None

        for user in self.unique_users:
            if user in self.predictions_dict.keys():
                self.model.set_weights(self.model_weights_dict[user])
                user_prediction_scaler = self.scalers_dict[user]
                X_test, Y_test = user_day_data.get_data_for_users([user])
                # if we have test data for this user
                # scale it
                if len(Y_test) > 0:
                    X_test = user_prediction_scaler.transform(X_test)
                    metrics = self.evaluate(
                        user_model=self.model,
                        test_covariates=X_test,
                        test_labels=Y_test,
                        predictions=predictions_dict[user],
                        plotAUC=plotAUC
                    )
                    self.metrics_dict[user] = metrics

        return self.metrics_dict

    def reset(self)->None:
        pass
