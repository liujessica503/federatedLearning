from abc import ABC, abstractmethod
from typing import Any, List

class BaseModel(ABC):

	def __init__(self, parameter_config):

		self.layers = parameter_config["layers"]
		self.input_dim = parameter_config["input_dim"]
		self.activation = parameter_config["activation"]
		self.loss = parameter_config["loss"]
		self.lr = parameter_config["learn_rate"]
		self.epochs = parameter_config["epochs"]
		self.batch_size = parameter_config["batch_size"]
		self.verbose = parameter_config["verbose"]
		self.output_path = parameter_config["output_path"]

	@abstractmethod
	def train(self, X: Any, y: Any) -> None:
		raise NotImplementedError

	@abstractmethod
	def predict(self, X: Any)-> List[int]:
		raise NotImplementedError

	@abstractmethod
	def reset(self)->None:
		raise NotImplementedError
