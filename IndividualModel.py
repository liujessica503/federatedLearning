from BaseModel import BaseModel, TestCallback
#from TestCallback import TestCallback
import numpy as np
from typing import Any, List
# standardize the data
from sklearn.preprocessing import StandardScaler


class IndividualModel(BaseModel):

    def train(self, user_day_data: Any, test_user_day_data: Any, test_callback = 0)->None:
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
            self.scalers_dict[user] = user_scaler
            X_train = user_scaler.transform(X_train)
            Y_train = self.output_layer.transform_labels(Y_train)
            
            # if json set test_callback to 1, after each epoch of training, 
            # we will evaluate the current model on our test set     
            callback_list = []
            if test_callback == 1:
                X_test, Y_test = test_user_day_data.get_data_for_users([user])
                # check if user has test data -- if they don't, then we're not doing callbacks
                if len(Y_test) > 0:
                    callback_list = [TestCallback((test_user_day_data), parentModel = self, userID = user)]

            #### THE ISSUE IS FOR CALLBACKS, WE NEED TO PASS IN THE USER SO THAT OUR PREDICT METHOD ONLY PREDICTS ON THAT USER
            #### AND NOT ON EVERYONE

            # for individual, will want to write each individual and each epoch
            # and we’ll want to weigh them by number of training points (but do later)

            # apply the template model, created in the init, to our data    
            self.model.fit(
                X_train,
                Y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks= callback_list
            )    

            self.model_weights_dict[user] = self.model.get_weights()           

        self.is_trained = True

    def predict(self, user_day_data: Any, userID = None)->List[float]:
        # self.check_is_trained()


        # if we are predicting during callbacks, need argument userID
        # will return a single prediction for that user
        if self.is_trained == False:
            self.model.set_weights(self.model.get_weights())
            user_prediction_scaler = self.scalers_dict[userID]
            X_test, Y_test = user_day_data.get_data_for_users([userID])
            X_test = user_prediction_scaler.transform(X_test)
            prediction = self.model.predict(X_test)
            return prediction, Y_test


        # if we've already finished training for all of our epochs
        # return a dictionary of predictions containing prediction for each user
        elif self.is_trained == True:
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
        # self.check_is_trained()

        # only the users that we have test data for
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
