from BaseModel import BaseModel
from TestCallback import TestCallback
import numpy as np
from typing import Any, List
# standardize the data
from sklearn.preprocessing import StandardScaler


class IndividualModel(BaseModel):

    def train(self, user_day_data: Any, test_data, test_callback = 0)->None:
        self.unique_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )
        self.model_weights_dict = {}
        self.scalers_dict = {}
        self.output_layer.fit_one_hot(user_day_data.get_y())
        for user in self.unique_users:

            self.model.set_weights(self.initialization)

            # get user-specific data
            X_train, Y_train = user_day_data.get_data_for_users([user])
            user_scaler = StandardScaler().fit(X_train)
            X_train = user_scaler.transform(X_train)
            Y_train = self.output_layer.transform_labels(Y_train)


            ## STILL NEED TO GET X_TEST AND Y_TEST
            
            callback_list = []
            if test_callback == 1:
            # after each epoch of training, 
            # we will evaluate the model on our test set 
                X_test = self.scaler.transform(test_user_day_data.get_X())
                Y_test = test_user_day_data.get_y()
                callback_list = TestCallback((X_test, Y_test))

            # apply the template model, created in the init, to our data    
            self.model.fit(
                X_train,
                Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks= callback_list
            )          

            # for individual, will want to write each individual and each epoch
            # and we’ll want to weigh them by number of training points (but do later)

            self.model_weights_dict[user] = self.model.get_weights()
            self.scalers_dict[user] = user_scaler
        self.is_trained = True

    def predict(self, user_day_data: Any)->List[float]:
        self.check_is_trained()

        predictions = np.empty(
            [len(user_day_data.get_y()), self.output_layer.length]
        )
        pred_users = np.unique(
            [x[0] for x in user_day_data.get_user_day_pairs()]
        )

        for user in pred_users:
            self.model.set_weights(self.model_weights_dict[user])
            user_prediction_scaler = self.scalers_dict[user]
            X_test, Y_test = user_day_data.get_data_for_users([user])
            X_test = user_prediction_scaler.transform(X_test)
            prediction = self.model.predict(X_test)
            predictions[
                user_day_data._get_rows_for_users([user]), :
            ] = prediction
        return predictions

    def get_score(self, user_day_data: Any)->str:
        self.check_is_trained()

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
