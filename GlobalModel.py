from BaseModel import BaseModel
from typing import List, Any
# standardize the data
from sklearn.preprocessing import StandardScaler


class GlobalModel(BaseModel):

    def train(self, user_day_data: Any)->None:
        self.scaler = StandardScaler().fit(user_day_data.get_X())
        # Scale the train set
        X_train = self.scaler.transform(user_day_data.get_X())
        Y_train = user_day_data.get_y()
        self.model.fit(
            X_train,
            Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

    def validate(self, X: Any, Y: Any, validation_data=None)->None:
        modelFit = self.model.fit(
            X,
            Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=validation_data
        )
        # return an object for running cross-validation purposes
        return modelFit

    def predict(self, user_day_data: Any)->List[float]:
        X_test = self.scaler.transform(user_day_data.get_X())
        return self.model.predict(X_test).ravel()

    def get_score(self, user_day_data: Any)->str:
        X_test = self.scaler.transform(user_day_data.get_X())
        return self.model.evaluate(X_test, user_day_data.get_y())

    def reset(self)->None:
        pass
