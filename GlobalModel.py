from BaseModel import BaseModel
from typing import Dict, List, Any
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam

class GlobalModel(BaseModel):

	def __init__(self, parameter_config: dict()):
		super().__init__(parameter_config)

		self.model = Sequential()

		self.model.add(Dense(self.layers[0], input_dim=self.input_dim, activation=self.activation))

		for i in range(1,len(self.layers)):
			self.model.add(Dense(self.layers[i], activation = self.activation))

		self.model.add(Dense(1, activation='sigmoid'))
		self.model.compile(
			loss=self.loss,
			optimizer = optimizers.Adam(
			  	lr=self.lr, 
			  	beta_1=0.9, 
			  	beta_2=0.999, 
			  	epsilon=None, 
			  	decay=0.0, 
			  	amsgrad=False
			),
			metrics=['accuracy'],
		)

	def train(self, X_dict: Any, Y_dict: Any)->None:
		# To make X, concat each dataFrame in X_dict

		self.model.fit(X_dict, Y_dict,epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

	def predict(self, X_dict: Any)->List[int]:
		# To make X, concat all dataFrames in X_dict
		return self.model.predict(X_dict).ravel()

	def evaluate(self) -> None:
		pass

	def write(self) -> None:
		pass

	def reset(self)->None:
		pass
