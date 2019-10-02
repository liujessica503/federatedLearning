from BaseModel import BaseModel
from typing import Dict

class GlobalServerModel(BaseModel):

	def __init__(self, parameter_config: Dict[Any]):
		super.__init__(parameter_config)

		self.model = Sequential()

		model.add(
			Dense(hidden_units=self.layers[0]), 
			input_dim=self.input_dim, 
			activation=self.activation
		)

		for i in range(1,len(self.layers)):
			model.add(Dense(hidden_units=self.layers[i]), activation=self.activation)

		model.add(Dense(1, activation='sigmoid'))
		model.compile(
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

	def train(self, X_dict: Any, y_dict: Any)->None:
		# To make X, concat each dataFrame in X_dict

		model.fit(X, Y ,epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)

	def predict(self, X_dict: Any)->List[int]:
		# To make X, concat all dataFrames in X_dict
		return model.predict(X).ravel()

	def reset(self)->None:
		pass
