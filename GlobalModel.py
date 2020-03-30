from BaseModel import BaseModel, TestCallback
#from TestCallback import TestCallback
from typing import List, Any
# standardize the data
from sklearn.preprocessing import StandardScaler


class GlobalModel(BaseModel):

    def train(self, user_day_data: Any, test_user_day_data: Any, test_callback = 0)->None:
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
        self.model.fit(
            X_train,
            Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks= callback_list
        )         


        self.is_trained = True

    def predict(self, user_day_data: Any)->List[float]:
        # self.check_is_trained()
        X_test = self.scaler.transform(user_day_data.get_X())
        return self.model.predict(X_test)

    def get_score(self, user_day_data: Any)->str:
        # self.check_is_trained()
        X_test = self.scaler.transform(user_day_data.get_X())
        return self.model.evaluate(X_test, user_day_data.get_y())
